//! Pure NIBBLX-512 $P_{16}$ Memory-Hard (Hybrid Cache-Timing Safe) & Iterated KDF (`#![no_std]`).
//!
//! Zero external ciphers or hashes: derives a 256-bit key strictly from NIBBLX-512 $P_{16}$
//! using 3-case domain-separated password absorption, Argon2id-style hybrid memory-hard mixing
//! (Pass 0: data-independent addressing for cache-timing resistance; Pass >= 1: data-dependent
//! addressing for GPU/ASIC trade-off resistance), division-free Lemire fast-range indexing,
//! and volatile zeroization of all scratchpad blocks and stack temporaries.

use crate::backend::scalar;
use crate::constants::{CASE_EMPTY, CASE_FULL, CASE_PARTIAL, DOM_KDF, DOM_KDF_FINAL, VER_KDF};
use crate::ct_ops;
use crate::state::MorphicState;
use crate::{MorphicError, KEY_LEN, NONCE_LEN, RATE_BYTES};

/// Deterministic public mixer for Pass-0 data-independent indexing (SplitMix64 finalizer).
#[inline(always)]
fn splitmix64_public(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Division-free constant-time range reduction mapping `x in [0, 2^64)` into `[0, m)`.
/// Uses 128-bit multiply-high (`mul`/`umulh`) — avoids variable-time hardware `DIV`.
#[inline(always)]
fn fast_range_u64(x: u64, m: usize) -> usize {
    (((x as u128).wrapping_mul(m as u128)) >> 64) as usize
}

/// Derives a 256-bit master key from `password` and a 192-bit `salt` using a caller-owned
/// memory-hard scratchpad `mem_blocks` (each block = 64 bytes = `[u64; 8]`).
///
/// - Recommended interactive parameters: `mem_blocks.len() == 262_144` (16 MiB), `passes == 3`.
/// - Pass 0 uses strictly public data-independent indexing (resistant to cache-timing leaks);
///   subsequent passes (`pass >= 1`) use division-free data-dependent indexing.
/// - All `mem_blocks` and internal stack temporaries are volatile-zeroized before returning.
pub fn derive_key_memory_hard(
    password: &[u8],
    salt: &[u8; NONCE_LEN],
    mem_blocks: &mut [[u64; 8]],
    passes: u32,
) -> Result<[u8; KEY_LEN], MorphicError> {
    if mem_blocks.is_empty() || passes == 0 {
        return Err(MorphicError::InvalidParameter);
    }

    let s = MorphicState::load_nonce(salt);
    let m_len = mem_blocks.len() as u64;
    let init_s0 = (VER_KDF << 48) | (DOM_KDF << 40) | (passes as u64);
    let mut st = MorphicState::new([
        init_s0,
        s[0],
        s[1],
        s[2],
        m_len,
        password.len() as u64,
        0,
        0,
    ]);
    scalar::permute_16(&mut st);

    // 3-case collision-free password absorption
    absorb_password(&mut st, password);
    scalar::permute_16(&mut st);

    // Phase 1: Fill entire scratchpad sequentially via P16
    for block in mem_blocks.iter_mut() {
        scalar::permute_16(&mut st);
        *block = st.lanes;
    }

    // Phase 2: Hybrid memory-hard mixing passes (Pass 0: data-independent; Pass >= 1: data-dependent)
    let m = mem_blocks.len();
    let mut ref_blk = [0u64; 8];
    for pass in 0..passes {
        st.lanes[7] ^= (pass as u64).wrapping_add(1);
        for i in 0..m {
            let j = if pass == 0 {
                let pub_seed = ((pass as u64) << 32) ^ (i as u64) ^ (m_len.rotate_left(17));
                fast_range_u64(splitmix64_public(pub_seed), m)
            } else {
                fast_range_u64(st.lanes[0], m)
            };
            ref_blk = mem_blocks[j];
            for l in 0..8 {
                st.lanes[l] ^= ref_blk[l];
            }
            scalar::permute_16(&mut st);
            for l in 0..8 {
                mem_blocks[i][l] ^= st.lanes[l];
            }
        }
    }
    ct_ops::secure_zeroize_u64(&mut ref_blk);

    // Phase 3: Final Davies-Meyer squeeze with salt + DOM_KDF_FINAL
    let mut pre = [st.lanes[0], st.lanes[1], st.lanes[2], st.lanes[3]];
    st.lanes[4] ^= s[0];
    st.lanes[5] ^= s[1];
    st.lanes[6] ^= s[2];
    st.lanes[7] ^= DOM_KDF_FINAL;
    scalar::permute_16(&mut st);

    let mut out = [0u8; KEY_LEN];
    for i in 0..4 {
        MorphicState::store_le64(&mut out[8 * i..], st.lanes[i] ^ pre[i]);
    }
    ct_ops::secure_zeroize_u64(&mut pre);

    // Wipe entire memory-hard scratchpad via volatile writes
    for block in mem_blocks.iter_mut() {
        ct_ops::secure_zeroize_u64(block);
    }

    Ok(out)
}

/// Zero-allocation iterated $P_{16}$ KDF for `#![no_std]` embedded devices or C-FFI callers
/// without a large heap scratchpad.
pub fn derive_key_iterated(
    password: &[u8],
    salt: &[u8; NONCE_LEN],
    iterations: u32,
) -> Result<[u8; KEY_LEN], MorphicError> {
    if iterations == 0 {
        return Err(MorphicError::InvalidParameter);
    }

    let s = MorphicState::load_nonce(salt);
    let init_s0 = (VER_KDF << 48) | (DOM_KDF << 40);
    let mut st = MorphicState::new([
        init_s0,
        s[0],
        s[1],
        s[2],
        0,
        password.len() as u64,
        iterations as u64,
        0,
    ]);
    scalar::permute_16(&mut st);

    absorb_password(&mut st, password);
    scalar::permute_16(&mut st);

    let mut acc = [st.lanes[0], st.lanes[1], st.lanes[2], st.lanes[3]];
    for iter in 0..iterations {
        st.lanes[7] ^= (iter as u64).wrapping_add(1);
        scalar::permute_16(&mut st);
        for i in 0..4 {
            acc[i] ^= st.lanes[i];
        }
    }

    st.lanes[4] ^= s[0];
    st.lanes[5] ^= s[1];
    st.lanes[6] ^= s[2];
    st.lanes[7] ^= DOM_KDF_FINAL;
    scalar::permute_16(&mut st);

    let mut out = [0u8; KEY_LEN];
    for i in 0..4 {
        MorphicState::store_le64(&mut out[8 * i..], st.lanes[i] ^ acc[i]);
    }
    ct_ops::secure_zeroize_u64(&mut acc);
    Ok(out)
}

fn absorb_password(st: &mut MorphicState, password: &[u8]) {
    if password.is_empty() {
        st.lanes[7] ^= CASE_EMPTY;
        return;
    }

    let full_blocks = password.len() / RATE_BYTES;
    let rem = password.len() % RATE_BYTES;

    if rem == 0 {
        for b in 0..full_blocks {
            let blk = &password[b * RATE_BYTES..];
            for i in 0..4 {
                st.lanes[i] ^= MorphicState::load_le64(&blk[8 * i..]);
            }
            if b + 1 < full_blocks {
                scalar::permute_16(st);
            }
        }
        st.lanes[6] ^= password.len() as u64;
        st.lanes[7] ^= CASE_FULL;
    } else {
        for b in 0..full_blocks {
            let blk = &password[b * RATE_BYTES..];
            for i in 0..4 {
                st.lanes[i] ^= MorphicState::load_le64(&blk[8 * i..]);
            }
            scalar::permute_16(st);
        }
        let mut pad = [0u8; RATE_BYTES];
        pad[..rem].copy_from_slice(&password[full_blocks * RATE_BYTES..]);
        pad[rem] = 0x80;
        for i in 0..4 {
            st.lanes[i] ^= MorphicState::load_le64(&pad[8 * i..]);
        }
        ct_ops::secure_zeroize_bytes(&mut pad);
        st.lanes[6] ^= password.len() as u64;
        st.lanes[7] ^= CASE_PARTIAL;
    }
}
