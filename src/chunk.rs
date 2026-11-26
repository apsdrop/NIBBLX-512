//! Single-chunk (Scalar) and Fused 4-Way Lockstep SIMD (AVX2/NEON) 64 KB chunk engine.

use crate::backend::{self, scalar};
use crate::constants::{
    CASE_EMPTY, CASE_FULL, CASE_PARTIAL, DOM_AD, DOM_CHUNK, DOM_CMT, FLAG_IS_FINAL,
};
use crate::ct_ops;
use crate::state::{MorphicState, SoaState4};
use crate::{MorphicError, MorphicTag, Profile, CHUNK_SIZE, KEY_LEN, MAX_CHUNKS, NONCE_LEN, RATE_BYTES};

/// One-way Davies-Meyer Key-Commitment Context $H_K$ (256-bit) with RAII zeroization.
#[derive(Clone)]
pub struct KeyCommitCtx {
    pub hk: [u64; 4],
}

impl Drop for KeyCommitCtx {
    fn drop(&mut self) {
        ct_ops::secure_zeroize_u64(&mut self.hk);
    }
}

/// Fix 5.1: Derives one-way Key-Commitment Context $H_K$ via $P_{16}$ + truncation + Key Feed-Forward.
pub fn derive_key_commit_ctx(
    key: &[u8; KEY_LEN],
    nonce: &[u8; NONCE_LEN],
    profile: Profile,
) -> KeyCommitCtx {
    let mut k = MorphicState::load_key(key);
    let n = MorphicState::load_nonce(nonce);

    let s0 = (profile.ver_profile_id() << 48) | (DOM_CMT << 40);
    let mut st = MorphicState::new([s0, n[0], n[1], n[2], k[0], k[1], k[2], k[3]]);
    scalar::permute_16(&mut st);

    let hk = [
        st.lanes[0] ^ k[0],
        st.lanes[1] ^ k[1],
        st.lanes[2] ^ k[2],
        st.lanes[3] ^ k[3],
    ];
    ct_ops::secure_zeroize_u64(&mut k);
    KeyCommitCtx { hk }
}

/// Computes 256-bit Associated Data (AD) digest bound to $H_K$, Nonce, and Key.
pub fn compute_ad_digest(
    key: &[u8; KEY_LEN],
    nonce: &[u8; NONCE_LEN],
    ctx: &KeyCommitCtx,
    profile: Profile,
    ad: &[u8],
) -> [u64; 4] {
    let mut k = MorphicState::load_key(key);
    let n = MorphicState::load_nonce(nonce);
    let use_p16 = profile.use_p16_data();

    let domain_word = (DOM_AD << 56) | ((ad.len() as u64) & 0x00FF_FFFF_FFFF_FFFF);
    let mut st = MorphicState::new([
        n[0],
        n[1],
        n[2],
        domain_word,
        ctx.hk[0],
        ctx.hk[1],
        ctx.hk[2],
        ctx.hk[3],
    ]);
    scalar::permute_16(&mut st);
    st.xor_capacity_key(&k);

    if ad.is_empty() {
        st.lanes[7] ^= CASE_EMPTY;
    } else {
        let full_blocks = ad.len() / RATE_BYTES;
        let rem = ad.len() % RATE_BYTES;
        if rem == 0 {
            for b in 0..full_blocks {
                let blk = &ad[b * RATE_BYTES..];
                for i in 0..4 {
                    st.lanes[i] ^= MorphicState::load_le64(&blk[8 * i..]);
                }
                if b + 1 < full_blocks {
                    scalar::permute_data(&mut st, use_p16);
                }
            }
            st.lanes[6] ^= ad.len() as u64;
            st.lanes[7] ^= CASE_FULL;
        } else {
            for b in 0..full_blocks {
                let blk = &ad[b * RATE_BYTES..];
                for i in 0..4 {
                    st.lanes[i] ^= MorphicState::load_le64(&blk[8 * i..]);
                }
                scalar::permute_data(&mut st, use_p16);
            }
            let mut pad = [0u8; RATE_BYTES];
            pad[..rem].copy_from_slice(&ad[full_blocks * RATE_BYTES..]);
            pad[rem] = 0x80;
            for i in 0..4 {
                st.lanes[i] ^= MorphicState::load_le64(&pad[8 * i..]);
            }
            ct_ops::secure_zeroize_bytes(&mut pad);
            st.lanes[6] ^= ad.len() as u64;
            st.lanes[7] ^= CASE_PARTIAL;
        }
    }

    st.xor_capacity_key(&k);
    scalar::permute_16(&mut st);

    let digest = [
        st.lanes[0] ^ k[0],
        st.lanes[1] ^ k[1],
        st.lanes[2] ^ k[2],
        st.lanes[3] ^ k[3],
    ];
    ct_ops::secure_zeroize_u64(&mut k);
    digest
}

fn process_chunk_scalar(
    key: &[u8; KEY_LEN],
    nonce: &[u8; NONCE_LEN],
    ctx: &KeyCommitCtx,
    ad_digest: &[u64; 4],
    profile: Profile,
    chunk_index: u64,
    is_final: bool,
    is_decrypt: bool,
    buf: &mut [u8],
) -> Result<MorphicTag, MorphicError> {
    if buf.len() > CHUNK_SIZE || chunk_index >= MAX_CHUNKS {
        return Err(MorphicError::InvalidParameter);
    }

    let mut k = MorphicState::load_key(key);
    let n = MorphicState::load_nonce(nonce);
    let use_p16 = profile.use_p16_data();

    let flags = if is_final { FLAG_IS_FINAL } else { 0u64 };
    let domain_word = (DOM_CHUNK << 56) | (flags << 48) | (chunk_index & 0x0000_FFFF_FFFF_FFFF);

    let mut st = MorphicState::new([
        n[0],
        n[1],
        n[2],
        domain_word,
        ctx.hk[0],
        ctx.hk[1],
        ctx.hk[2],
        ctx.hk[3],
    ]);
    scalar::permute_16(&mut st);
    st.xor_capacity_key(&k);

    for i in 0..4 {
        st.lanes[i] ^= ad_digest[i];
    }
    scalar::permute_data(&mut st, use_p16);

    if buf.is_empty() {
        st.lanes[7] ^= CASE_EMPTY;
    } else {
        let full_blocks = buf.len() / RATE_BYTES;
        let rem = buf.len() % RATE_BYTES;

        if rem == 0 {
            for b in 0..full_blocks {
                let blk = &mut buf[b * RATE_BYTES..(b + 1) * RATE_BYTES];
                for i in 0..4 {
                    let w = MorphicState::load_le64(&blk[8 * i..]);
                    if !is_decrypt {
                        let ct = w ^ st.lanes[i];
                        MorphicState::store_le64(&mut blk[8 * i..], ct);
                        st.lanes[i] = ct;
                    } else {
                        let pt = w ^ st.lanes[i];
                        MorphicState::store_le64(&mut blk[8 * i..], pt);
                        st.lanes[i] = w;
                    }
                }
                if b + 1 < full_blocks {
                    scalar::permute_data(&mut st, use_p16);
                }
            }
            st.lanes[6] ^= buf.len() as u64;
            st.lanes[7] ^= CASE_FULL;
        } else {
            for b in 0..full_blocks {
                let blk = &mut buf[b * RATE_BYTES..(b + 1) * RATE_BYTES];
                for i in 0..4 {
                    let w = MorphicState::load_le64(&blk[8 * i..]);
                    if !is_decrypt {
                        let ct = w ^ st.lanes[i];
                        MorphicState::store_le64(&mut blk[8 * i..], ct);
                        st.lanes[i] = ct;
                    } else {
                        let pt = w ^ st.lanes[i];
                        MorphicState::store_le64(&mut blk[8 * i..], pt);
                        st.lanes[i] = w;
                    }
                }
                scalar::permute_data(&mut st, use_p16);
            }
            st.duplex_partial_block(&mut buf[full_blocks * RATE_BYTES..], is_decrypt);
            st.lanes[6] ^= buf.len() as u64;
            st.lanes[7] ^= CASE_PARTIAL;
        }
    }

    st.xor_capacity_key(&k);
    scalar::permute_16(&mut st);

    let tag = squeeze_tag(&st.lanes, &k, profile);
    ct_ops::secure_zeroize_u64(&mut k);
    Ok(tag)
}

#[inline(always)]
fn squeeze_tag(lanes: &[u64; 8], k: &[u64; 4], profile: Profile) -> MorphicTag {
    match profile {
        Profile::Compact128 => {
            let mut t = [0u8; 16];
            MorphicState::store_le64(&mut t[0..8], lanes[0] ^ k[0]);
            MorphicState::store_le64(&mut t[8..16], lanes[1] ^ k[1]);
            MorphicTag::Compact128(t)
        }
        Profile::EnterpriseCmt256 | Profile::HighSecCmt256 => {
            let mut t = [0u8; 32];
            for i in 0..4 {
                MorphicState::store_le64(&mut t[8 * i..], lanes[i] ^ k[i]);
            }
            MorphicTag::Commit256(t)
        }
    }
}

/// Encrypts one chunk (<= 64 KB) in place using the scalar backend.
pub fn encrypt_chunk(
    key: &[u8; KEY_LEN],
    nonce: &[u8; NONCE_LEN],
    ctx: &KeyCommitCtx,
    ad_digest: &[u64; 4],
    profile: Profile,
    chunk_index: u64,
    is_final: bool,
    buf: &mut [u8],
) -> Result<MorphicTag, MorphicError> {
    process_chunk_scalar(
        key,
        nonce,
        ctx,
        ad_digest,
        profile,
        chunk_index,
        is_final,
        false,
        buf,
    )
}

/// Decrypts one chunk (<= 64 KB) in place using the scalar backend.
pub fn decrypt_chunk(
    key: &[u8; KEY_LEN],
    nonce: &[u8; NONCE_LEN],
    ctx: &KeyCommitCtx,
    ad_digest: &[u64; 4],
    profile: Profile,
    chunk_index: u64,
    is_final: bool,
    buf: &mut [u8],
) -> Result<MorphicTag, MorphicError> {
    process_chunk_scalar(
        key,
        nonce,
        ctx,
        ad_digest,
        profile,
        chunk_index,
        is_final,
        true,
        buf,
    )
}

/// Fused 4-Way Lockstep SIMD processor for 4 exact 64 KB chunks (`CASE_FULL`, 2,048 blocks).
fn process_4_full_chunks_simd<const IS_DECRYPT: bool>(
    key: &[u8; KEY_LEN],
    nonce: &[u8; NONCE_LEN],
    ctx: &KeyCommitCtx,
    ad_digest: &[u64; 4],
    profile: Profile,
    base_chunk_index: u64,
    last_chunk_is_final: bool,
    chunks: [&mut [u8]; 4],
) -> Result<[MorphicTag; 4], MorphicError> {
    if base_chunk_index > MAX_CHUNKS - 4 {
        return Err(MorphicError::InvalidParameter);
    }
    for c in chunks.iter() {
        if c.len() != CHUNK_SIZE {
            return Err(MorphicError::InvalidParameter);
        }
    }

    let mut k = MorphicState::load_key(key);
    let n = MorphicState::load_nonce(nonce);
    let use_p16 = profile.use_p16_data();

    let mut soa = SoaState4::zeroed();
    for q in 0..4 {
        let flags = if q == 3 && last_chunk_is_final {
            FLAG_IS_FINAL
        } else {
            0u64
        };
        let domain_word = (DOM_CHUNK << 56)
            | (flags << 48)
            | ((base_chunk_index + q as u64) & 0x0000_FFFF_FFFF_FFFF);

        soa.lanes[0][q] = n[0];
        soa.lanes[1][q] = n[1];
        soa.lanes[2][q] = n[2];
        soa.lanes[3][q] = domain_word;
        soa.lanes[4][q] = ctx.hk[0];
        soa.lanes[5][q] = ctx.hk[1];
        soa.lanes[6][q] = ctx.hk[2];
        soa.lanes[7][q] = ctx.hk[3];
    }

    backend::process_4_full_chunks::<IS_DECRYPT>(&mut soa, chunks, &k, ad_digest, use_p16);

    let mut tags = [MorphicTag::Compact128([0u8; 16]); 4];
    for q in 0..4 {
        let lanes_q = [
            soa.lanes[0][q],
            soa.lanes[1][q],
            soa.lanes[2][q],
            soa.lanes[3][q],
            0,
            0,
            0,
            0,
        ];
        tags[q] = squeeze_tag(&lanes_q, &k, profile);
    }

    ct_ops::secure_zeroize_u64(&mut k);
    Ok(tags)
}

/// Encrypts 4 full 64 KB chunks simultaneously via fused register-resident AVX2/NEON kernel.
pub fn encrypt_4_full_chunks(
    key: &[u8; KEY_LEN],
    nonce: &[u8; NONCE_LEN],
    ctx: &KeyCommitCtx,
    ad_digest: &[u64; 4],
    profile: Profile,
    base_chunk_index: u64,
    last_chunk_is_final: bool,
    chunks: [&mut [u8]; 4],
) -> Result<[MorphicTag; 4], MorphicError> {
    process_4_full_chunks_simd::<false>(
        key,
        nonce,
        ctx,
        ad_digest,
        profile,
        base_chunk_index,
        last_chunk_is_final,
        chunks,
    )
}

/// Decrypts 4 full 64 KB chunks simultaneously via fused register-resident AVX2/NEON kernel.
pub fn decrypt_4_full_chunks(
    key: &[u8; KEY_LEN],
    nonce: &[u8; NONCE_LEN],
    ctx: &KeyCommitCtx,
    ad_digest: &[u64; 4],
    profile: Profile,
    base_chunk_index: u64,
    last_chunk_is_final: bool,
    chunks: [&mut [u8]; 4],
) -> Result<[MorphicTag; 4], MorphicError> {
    process_4_full_chunks_simd::<true>(
        key,
        nonce,
        ctx,
        ad_digest,
        profile,
        base_chunk_index,
        last_chunk_is_final,
        chunks,
    )
}
