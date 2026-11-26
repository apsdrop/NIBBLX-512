//! Hybrid 4-Way SIMD + Scalar Tail Streaming & In-Place AEAD orchestrator.
//!
//! Hardened with borrowed `&'k [u8; KEY_LEN]` key references in `RootFinalizer`,
//! `StreamEncryptor`, and `StreamDecryptor` (zero raw key copies stored in structs),
//! plus permanent auth-failure poisoning on `StreamDecryptor`.

use crate::backend::scalar;
use crate::chunk::{
    compute_ad_digest, decrypt_4_full_chunks, decrypt_chunk, derive_key_commit_ctx,
    encrypt_4_full_chunks, encrypt_chunk, KeyCommitCtx,
};
use crate::constants::{DOM_ROOT, DOM_ROOT_EOF};
use crate::ct_ops;
use crate::state::MorphicState;
use crate::{
    MorphicError, MorphicTag, Profile, CHUNK_SIZE, KEY_LEN, MAX_CHUNKS, NONCE_LEN, RATE_BYTES,
};

/// Streaming Root Finalization Sponge (O(1) RAM, borrowed key lifetime, zero raw key copy).
pub struct RootFinalizer<'k> {
    st: MorphicState,
    key: &'k [u8; KEY_LEN],
    profile: Profile,
    pending: [u8; RATE_BYTES],
    pending_len: usize,
    chunk_count: u64,
    total_bytes: u64,
}

impl<'k> RootFinalizer<'k> {
    pub fn new(
        key: &'k [u8; KEY_LEN],
        nonce: &[u8; NONCE_LEN],
        ctx: &KeyCommitCtx,
        ad_digest: &[u64; 4],
        profile: Profile,
    ) -> Self {
        let mut k = MorphicState::load_key(key);
        let n = MorphicState::load_nonce(nonce);

        let domain_word = DOM_ROOT << 56;
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
        ct_ops::secure_zeroize_u64(&mut k);

        for i in 0..4 {
            st.lanes[i] ^= ad_digest[i];
        }
        scalar::permute_data(&mut st, profile.use_p16_data());

        Self {
            st,
            key,
            profile,
            pending: [0u8; RATE_BYTES],
            pending_len: 0,
            chunk_count: 0,
            total_bytes: 0,
        }
    }

    pub fn absorb_chunk_tag(
        &mut self,
        chunk_tag: &MorphicTag,
        chunk_byte_len: usize,
    ) -> Result<(), MorphicError> {
        let tag_bytes = chunk_tag.as_bytes();
        if tag_bytes.len() != self.profile.tag_len() || self.chunk_count >= MAX_CHUNKS {
            return Err(MorphicError::InvalidParameter);
        }

        if self.pending_len == RATE_BYTES {
            for i in 0..4 {
                self.st.lanes[i] ^= MorphicState::load_le64(&self.pending[8 * i..]);
            }
            scalar::permute_data(&mut self.st, self.profile.use_p16_data());
            self.pending_len = 0;
        }

        self.pending[self.pending_len..self.pending_len + tag_bytes.len()]
            .copy_from_slice(tag_bytes);
        self.pending_len += tag_bytes.len();
        self.chunk_count += 1;
        self.total_bytes = self.total_bytes.wrapping_add(chunk_byte_len as u64);
        Ok(())
    }

    /// Finalizes with 3 strictly disjoint Capacity lanes ($S_5, S_6, S_7$) + single $P_{16}$.
    pub(crate) fn finalize_in_place(&mut self) -> MorphicTag {
        if self.pending_len == RATE_BYTES {
            for i in 0..4 {
                self.st.lanes[i] ^= MorphicState::load_le64(&self.pending[8 * i..]);
            }
        } else if self.pending_len > 0 {
            let mut pad = [0u8; RATE_BYTES];
            pad[..self.pending_len].copy_from_slice(&self.pending[..self.pending_len]);
            pad[self.pending_len] = 0x80;
            for i in 0..4 {
                self.st.lanes[i] ^= MorphicState::load_le64(&pad[8 * i..]);
            }
            ct_ops::secure_zeroize_bytes(&mut pad);
        }

        let mut k = MorphicState::load_key(self.key);
        self.st.lanes[5] ^= self.chunk_count;
        self.st.lanes[6] ^= self.total_bytes;
        self.st.lanes[7] ^= DOM_ROOT_EOF;
        self.st.xor_capacity_key(&k);
        scalar::permute_16(&mut self.st);

        let tag = match self.profile {
            Profile::Compact128 => {
                let mut t = [0u8; 16];
                MorphicState::store_le64(&mut t[0..8], self.st.lanes[0] ^ k[0]);
                MorphicState::store_le64(&mut t[8..16], self.st.lanes[1] ^ k[1]);
                MorphicTag::Compact128(t)
            }
            Profile::EnterpriseCmt256 | Profile::HighSecCmt256 => {
                let mut t = [0u8; 32];
                for i in 0..4 {
                    MorphicState::store_le64(&mut t[8 * i..], self.st.lanes[i] ^ k[i]);
                }
                MorphicTag::Commit256(t)
            }
        };
        ct_ops::secure_zeroize_u64(&mut k);
        tag
    }

    pub fn finalize(mut self) -> MorphicTag {
        self.finalize_in_place()
    }
}

impl<'k> Drop for RootFinalizer<'k> {
    fn drop(&mut self) {
        ct_ops::secure_zeroize_bytes(&mut self.pending);
    }
}

/// Zero-copy O(1)-RAM Multi-GB Stream Encryptor (borrowed key lifetime, zero raw key copy).
pub struct StreamEncryptor<'k> {
    key: &'k [u8; KEY_LEN],
    nonce: [u8; NONCE_LEN],
    ctx: KeyCommitCtx,
    ad_digest: [u64; 4],
    profile: Profile,
    root: RootFinalizer<'k>,
    next_chunk_idx: u64,
    finalized: bool,
}

impl<'k> StreamEncryptor<'k> {
    pub fn new(
        key: &'k [u8; KEY_LEN],
        nonce: &[u8; NONCE_LEN],
        ad: &[u8],
        profile: Profile,
    ) -> Self {
        let ctx = derive_key_commit_ctx(key, nonce, profile);
        let ad_digest = compute_ad_digest(key, nonce, &ctx, profile, ad);
        let root = RootFinalizer::new(key, nonce, &ctx, &ad_digest, profile);
        Self {
            key,
            nonce: *nonce,
            ctx,
            ad_digest,
            profile,
            root,
            next_chunk_idx: 0,
            finalized: false,
        }
    }

    /// Encrypts 4 full 64 KB chunks simultaneously via AVX2/NEON 4-way SIMD.
    pub fn encrypt_4_chunks(
        &mut self,
        chunks: [&mut [u8]; 4],
        last_chunk_is_final: bool,
    ) -> Result<[MorphicTag; 4], MorphicError> {
        if self.finalized {
            return Err(MorphicError::StreamAlreadyFinalized);
        }
        let tags = encrypt_4_full_chunks(
            self.key,
            &self.nonce,
            &self.ctx,
            &self.ad_digest,
            self.profile,
            self.next_chunk_idx,
            last_chunk_is_final,
            chunks,
        )?;
        for t in tags.iter() {
            self.root.absorb_chunk_tag(t, CHUNK_SIZE)?;
        }
        self.next_chunk_idx += 4;
        if last_chunk_is_final {
            self.finalized = true;
        }
        Ok(tags)
    }

    /// Encrypts a single chunk (<= 64 KB) via the scalar backend.
    pub fn encrypt_next_chunk(
        &mut self,
        chunk_buf: &mut [u8],
        is_final: bool,
    ) -> Result<MorphicTag, MorphicError> {
        if self.finalized {
            return Err(MorphicError::StreamAlreadyFinalized);
        }
        if !is_final && chunk_buf.len() != CHUNK_SIZE {
            return Err(MorphicError::InvalidParameter);
        }
        let ctag = encrypt_chunk(
            self.key,
            &self.nonce,
            &self.ctx,
            &self.ad_digest,
            self.profile,
            self.next_chunk_idx,
            is_final,
            chunk_buf,
        )?;
        self.root.absorb_chunk_tag(&ctag, chunk_buf.len())?;
        self.next_chunk_idx += 1;
        if is_final {
            self.finalized = true;
        }
        Ok(ctag)
    }

    pub fn finish(mut self) -> Result<MorphicTag, MorphicError> {
        if !self.finalized {
            return Err(MorphicError::InvalidParameter);
        }
        Ok(self.root.finalize_in_place())
    }
}

impl<'k> Drop for StreamEncryptor<'k> {
    fn drop(&mut self) {
        ct_ops::secure_zeroize_u64(&mut self.ad_digest);
    }
}

/// Zero-copy O(1)-RAM Multi-GB Stream Decryptor (borrowed key lifetime, per-chunk CMT-4 verify,
/// permanently poisoned on first authentication failure to prevent retry oracles).
pub struct StreamDecryptor<'k> {
    key: &'k [u8; KEY_LEN],
    nonce: [u8; NONCE_LEN],
    ctx: KeyCommitCtx,
    ad_digest: [u64; 4],
    profile: Profile,
    root: RootFinalizer<'k>,
    next_chunk_idx: u64,
    finalized: bool,
    poisoned: bool,
}

impl<'k> StreamDecryptor<'k> {
    pub fn new(
        key: &'k [u8; KEY_LEN],
        nonce: &[u8; NONCE_LEN],
        ad: &[u8],
        profile: Profile,
    ) -> Self {
        let ctx = derive_key_commit_ctx(key, nonce, profile);
        let ad_digest = compute_ad_digest(key, nonce, &ctx, profile, ad);
        let root = RootFinalizer::new(key, nonce, &ctx, &ad_digest, profile);
        Self {
            key,
            nonce: *nonce,
            ctx,
            ad_digest,
            profile,
            root,
            next_chunk_idx: 0,
            finalized: false,
            poisoned: false,
        }
    }

    pub fn decrypt_4_chunks(
        &mut self,
        mut chunks: [&mut [u8]; 4],
        expected_tags: &[MorphicTag; 4],
        last_chunk_is_final: bool,
    ) -> Result<(), MorphicError> {
        if self.finalized || self.poisoned {
            return Err(MorphicError::StreamAlreadyFinalized);
        }
        for t in expected_tags.iter() {
            if t.as_bytes().len() != self.profile.tag_len() {
                return Err(MorphicError::InvalidParameter);
            }
        }

        let computed = {
            let [c0, c1, c2, c3] = &mut chunks;
            decrypt_4_full_chunks(
                self.key,
                &self.nonce,
                &self.ctx,
                &self.ad_digest,
                self.profile,
                self.next_chunk_idx,
                last_chunk_is_final,
                [*c0, *c1, *c2, *c3],
            )?
        };

        let mut all_ok = true;
        for q in 0..4 {
            all_ok &= ct_ops::ct_eq_slice(computed[q].as_bytes(), expected_tags[q].as_bytes());
        }
        if !all_ok {
            self.poisoned = true;
            for c in chunks.iter_mut() {
                ct_ops::secure_zeroize_bytes(c);
            }
            return Err(MorphicError::AuthenticationFailed);
        }

        for t in computed.iter() {
            self.root.absorb_chunk_tag(t, CHUNK_SIZE)?;
        }
        self.next_chunk_idx += 4;
        if last_chunk_is_final {
            self.finalized = true;
        }
        Ok(())
    }

    pub fn decrypt_next_chunk(
        &mut self,
        chunk_buf: &mut [u8],
        expected_chunk_tag: &MorphicTag,
        is_final: bool,
    ) -> Result<(), MorphicError> {
        if self.finalized || self.poisoned {
            return Err(MorphicError::StreamAlreadyFinalized);
        }
        if expected_chunk_tag.as_bytes().len() != self.profile.tag_len() {
            return Err(MorphicError::InvalidParameter);
        }
        if !is_final && chunk_buf.len() != CHUNK_SIZE {
            return Err(MorphicError::InvalidParameter);
        }

        let computed = decrypt_chunk(
            self.key,
            &self.nonce,
            &self.ctx,
            &self.ad_digest,
            self.profile,
            self.next_chunk_idx,
            is_final,
            chunk_buf,
        )?;

        if !ct_ops::ct_eq_slice(computed.as_bytes(), expected_chunk_tag.as_bytes()) {
            self.poisoned = true;
            ct_ops::secure_zeroize_bytes(chunk_buf);
            return Err(MorphicError::AuthenticationFailed);
        }

        self.root.absorb_chunk_tag(&computed, chunk_buf.len())?;
        self.next_chunk_idx += 1;
        if is_final {
            self.finalized = true;
        }
        Ok(())
    }

    pub fn finish_and_verify(
        mut self,
        expected_master_tag: &MorphicTag,
    ) -> Result<(), MorphicError> {
        if self.poisoned {
            return Err(MorphicError::StreamAlreadyFinalized);
        }
        if !self.finalized || expected_master_tag.as_bytes().len() != self.profile.tag_len() {
            return Err(MorphicError::InvalidParameter);
        }
        let computed = self.root.finalize_in_place();
        if ct_ops::ct_eq_slice(computed.as_bytes(), expected_master_tag.as_bytes()) {
            Ok(())
        } else {
            self.poisoned = true;
            Err(MorphicError::AuthenticationFailed)
        }
    }
}

impl<'k> Drop for StreamDecryptor<'k> {
    fn drop(&mut self) {
        ct_ops::secure_zeroize_u64(&mut self.ad_digest);
    }
}

/// Hybrid in-place AEAD encryption: 4-way SIMD batches + scalar tail.
pub fn encrypt_in_place(
    key: &[u8; KEY_LEN],
    nonce: &[u8; NONCE_LEN],
    ad: &[u8],
    profile: Profile,
    buffer: &mut [u8],
) -> Result<MorphicTag, MorphicError> {
    let ctx = derive_key_commit_ctx(key, nonce, profile);
    let mut ad_digest = compute_ad_digest(key, nonce, &ctx, profile, ad);
    let mut root = RootFinalizer::new(key, nonce, &ctx, &ad_digest, profile);

    if buffer.is_empty() {
        let ctag = encrypt_chunk(key, nonce, &ctx, &ad_digest, profile, 0, true, buffer)?;
        root.absorb_chunk_tag(&ctag, 0)?;
    } else {
        let total_chunks = ((buffer.len() - 1) / CHUNK_SIZE + 1) as u64;
        let batch_bytes = CHUNK_SIZE * 4;
        let mut idx = 0u64;
        let mut rest = &mut *buffer;

        while rest.len() >= batch_bytes {
            let (batch, tail) = rest.split_at_mut(batch_bytes);
            let (c0, r1) = batch.split_at_mut(CHUNK_SIZE);
            let (c1, r2) = r1.split_at_mut(CHUNK_SIZE);
            let (c2, c3) = r2.split_at_mut(CHUNK_SIZE);

            let last_is_final = (idx + 4) == total_chunks;
            let tags = encrypt_4_full_chunks(
                key,
                nonce,
                &ctx,
                &ad_digest,
                profile,
                idx,
                last_is_final,
                [c0, c1, c2, c3],
            )?;
            for t in tags.iter() {
                root.absorb_chunk_tag(t, CHUNK_SIZE)?;
            }
            idx += 4;
            rest = tail;
        }

        for chunk in rest.chunks_mut(CHUNK_SIZE) {
            let is_final = (idx + 1) == total_chunks;
            let ctag = encrypt_chunk(key, nonce, &ctx, &ad_digest, profile, idx, is_final, chunk)?;
            root.absorb_chunk_tag(&ctag, chunk.len())?;
            idx += 1;
        }
    }

    ct_ops::secure_zeroize_u64(&mut ad_digest);
    Ok(root.finalize())
}

/// Hybrid in-place AEAD decryption with strict verify-before-release & volatile wipe on any failure.
pub fn decrypt_in_place(
    key: &[u8; KEY_LEN],
    nonce: &[u8; NONCE_LEN],
    ad: &[u8],
    profile: Profile,
    buffer: &mut [u8],
    expected_tag: &MorphicTag,
) -> Result<(), MorphicError> {
    if expected_tag.as_bytes().len() != profile.tag_len() {
        return Err(MorphicError::InvalidParameter);
    }

    let ctx = derive_key_commit_ctx(key, nonce, profile);
    let mut ad_digest = compute_ad_digest(key, nonce, &ctx, profile, ad);
    let mut root = RootFinalizer::new(key, nonce, &ctx, &ad_digest, profile);

    let process_res: Result<(), MorphicError> = (|| {
        if buffer.is_empty() {
            let ctag = decrypt_chunk(key, nonce, &ctx, &ad_digest, profile, 0, true, buffer)?;
            root.absorb_chunk_tag(&ctag, 0)?;
        } else {
            let total_chunks = ((buffer.len() - 1) / CHUNK_SIZE + 1) as u64;
            let batch_bytes = CHUNK_SIZE * 4;
            let mut idx = 0u64;
            let mut rest: &mut [u8] = &mut *buffer;

            while rest.len() >= batch_bytes {
                let (batch, tail) = rest.split_at_mut(batch_bytes);
                let (c0, r1) = batch.split_at_mut(CHUNK_SIZE);
                let (c1, r2) = r1.split_at_mut(CHUNK_SIZE);
                let (c2, c3) = r2.split_at_mut(CHUNK_SIZE);

                let last_is_final = (idx + 4) == total_chunks;
                let tags = decrypt_4_full_chunks(
                    key,
                    nonce,
                    &ctx,
                    &ad_digest,
                    profile,
                    idx,
                    last_is_final,
                    [c0, c1, c2, c3],
                )?;
                for t in tags.iter() {
                    root.absorb_chunk_tag(t, CHUNK_SIZE)?;
                }
                idx += 4;
                rest = tail;
            }

            for chunk in rest.chunks_mut(CHUNK_SIZE) {
                let is_final = (idx + 1) == total_chunks;
                let ctag =
                    decrypt_chunk(key, nonce, &ctx, &ad_digest, profile, idx, is_final, chunk)?;
                root.absorb_chunk_tag(&ctag, chunk.len())?;
                idx += 1;
            }
        }
        Ok(())
    })();

    ct_ops::secure_zeroize_u64(&mut ad_digest);

    if let Err(e) = process_res {
        ct_ops::secure_zeroize_bytes(buffer);
        return Err(e);
    }

    let computed = root.finalize();

    if ct_ops::ct_eq_slice(computed.as_bytes(), expected_tag.as_bytes()) {
        Ok(())
    } else {
        ct_ops::secure_zeroize_bytes(buffer);
        Err(MorphicError::AuthenticationFailed)
    }
}

/// Pure scalar reference encryption (used for SIMD cross-verification).
pub fn encrypt_in_place_scalar(
    key: &[u8; KEY_LEN],
    nonce: &[u8; NONCE_LEN],
    ad: &[u8],
    profile: Profile,
    buffer: &mut [u8],
) -> Result<MorphicTag, MorphicError> {
    let ctx = derive_key_commit_ctx(key, nonce, profile);
    let mut ad_digest = compute_ad_digest(key, nonce, &ctx, profile, ad);
    let mut root = RootFinalizer::new(key, nonce, &ctx, &ad_digest, profile);

    if buffer.is_empty() {
        let ctag = encrypt_chunk(key, nonce, &ctx, &ad_digest, profile, 0, true, buffer)?;
        root.absorb_chunk_tag(&ctag, 0)?;
    } else {
        let total_chunks = ((buffer.len() - 1) / CHUNK_SIZE + 1) as u64;
        for (i, chunk) in buffer.chunks_mut(CHUNK_SIZE).enumerate() {
            let idx = i as u64;
            let is_final = (idx + 1) == total_chunks;
            let ctag = encrypt_chunk(key, nonce, &ctx, &ad_digest, profile, idx, is_final, chunk)?;
            root.absorb_chunk_tag(&ctag, chunk.len())?;
        }
    }

    ct_ops::secure_zeroize_u64(&mut ad_digest);
    Ok(root.finalize())
}
