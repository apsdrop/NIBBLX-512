//! # NIBBLX-512 v3.1.1 ("Seal-Pack", `#![no_std]` Production Crate)
//!
//! Zero-dependency, 512-bit bijective ARX-Chi sponge AEAD cipher v3.1.0 featuring:
//! - Full Double-Rotate ARX ($\mathcal{A}_{v3}$), PF1 non-linear pre-fold `W = Y + ROR(Y, 17)`
//!   + 4-way interleaved 5-step `SHL`-folded `CT-MR2_v3.1` with `black_box`-hardened masks
//! - $P_{14}$ data duplexing and $P_{16}$ Init, Finalization, $H_K$, KDF, and HighSec
//! - Fused register-resident 4-Way SIMD (`x86_64` AVX2 with `#![no_std]` CPUID+XGETBV auto-dispatch
//!   & `aarch64` NEON) + zero-spill Scalar tail

#![no_std]
#![forbid(unsafe_op_in_unsafe_fn)]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

#[cfg(test)]
extern crate std;

pub mod backend;
pub mod chunk;
pub mod constants;
pub mod ct_ops;
pub mod ffi;
pub mod kdf;
pub mod state;
pub mod stream;

use core::fmt;

pub use stream::{
    decrypt_in_place, encrypt_in_place, RootFinalizer, StreamDecryptor, StreamEncryptor,
};

/// Key length in bytes (256 bits).
pub const KEY_LEN: usize = 32;
/// Nonce length in bytes (192 bits).
pub const NONCE_LEN: usize = 24;
/// Sponge rate in bytes (256 bits = 4 x 64-bit lanes).
pub const RATE_BYTES: usize = 32;
/// Independent parallel chunk size in bytes (64 KB).
pub const CHUNK_SIZE: usize = 65_536;
/// Maximum number of chunks per nonce ($2^{48}$ chunks = 16 Exabytes).
pub const MAX_CHUNKS: u64 = 1u64 << 48;

/// Cryptographic Profile & Protocol Version (v3.1.0, cryptographically bound into $H_K$).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// `0x0311`: 128-bit Auth Tag (64-bit birthday key-commitment), $P_{14}$ data rounds.
    Compact128,
    /// `0x0312`: 256-bit CMT-4 Key-Committing Tag ($2^{128}$ multi-key bound), $P_{14}$ data rounds.
    EnterpriseCmt256,
    /// `0x0313`: 256-bit CMT-4 Key-Committing Tag + $P_{16}$ everywhere.
    HighSecCmt256,
}

impl Profile {
    #[inline(always)]
    pub const fn ver_profile_id(self) -> u64 {
        match self {
            Profile::Compact128 => constants::VER_COMPACT_128,
            Profile::EnterpriseCmt256 => constants::VER_ENTERPRISE_256,
            Profile::HighSecCmt256 => constants::VER_HIGHSEC_256,
        }
    }

    #[inline(always)]
    pub const fn tag_len(self) -> usize {
        match self {
            Profile::Compact128 => 16,
            Profile::EnterpriseCmt256 | Profile::HighSecCmt256 => 32,
        }
    }

    #[inline(always)]
    pub const fn use_p16_data(self) -> bool {
        matches!(self, Profile::HighSecCmt256)
    }

    #[inline(always)]
    pub const fn from_ver_id(id: u16) -> Option<Self> {
        match id as u64 {
            constants::VER_COMPACT_128 => Some(Profile::Compact128),
            constants::VER_ENTERPRISE_256 => Some(Profile::EnterpriseCmt256),
            constants::VER_HIGHSEC_256 => Some(Profile::HighSecCmt256),
            _ => None,
        }
    }
}

/// Profile-consistent authentication tag (16 bytes Compact or 32 bytes Enterprise CMT-4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphicTag {
    Compact128([u8; 16]),
    Commit256([u8; 32]),
}

impl MorphicTag {
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            MorphicTag::Compact128(b) => b,
            MorphicTag::Commit256(b) => b,
        }
    }
}

/// Error type for NIBBLX-512 v3.1.0 operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphicError {
    InvalidParameter,
    AuthenticationFailed,
    StreamAlreadyFinalized,
}

impl fmt::Display for MorphicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MorphicError::InvalidParameter => {
                write!(f, "invalid parameter, profile mismatch, or chunk limit exceeded")
            }
            MorphicError::AuthenticationFailed => {
                write!(f, "AEAD authentication tag verification failed")
            }
            MorphicError::StreamAlreadyFinalized => {
                write!(f, "stream already finalized")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec;

    #[test]
    fn test_verified_v31_kat_enterprise_and_compact() {
        let mut key = [0u8; KEY_LEN];
        for i in 0..KEY_LEN {
            key[i] = i as u8;
        }
        let mut nonce = [0u8; NONCE_LEN];
        for i in 0..NONCE_LEN {
            nonce[i] = (0xA0 + i) as u8;
        }
        let ad = b"MORPHIC-SPONGE-ARX-512-V31";
        let mut msg = [0u8; 48];
        for i in 0..48 {
            msg[i] = (0x30 + i) as u8;
        }
        let orig = msg;

        let ent_tag =
            encrypt_in_place(&key, &nonce, ad, Profile::EnterpriseCmt256, &mut msg).unwrap();
        let expected_ent_ct16: [u8; 16] = [
            0x77, 0xCE, 0xF8, 0x15, 0x22, 0xF6, 0xB2, 0x26, 0xA8, 0x9A, 0x51, 0xD2, 0x17, 0x6F,
            0x36, 0x47,
        ];
        let expected_ent_tag32: [u8; 32] = [
            0x7E, 0xCC, 0x05, 0xCF, 0x48, 0x88, 0x92, 0xE5, 0x88, 0x7D, 0xB9, 0x09, 0xA7, 0xFE,
            0x4E, 0xD9, 0x36, 0x5C, 0xC7, 0x20, 0x51, 0x39, 0x72, 0xE0, 0xBD, 0xB2, 0xEC, 0x38,
            0xB2, 0x83, 0xA0, 0xA5,
        ];
        assert_eq!(&msg[..16], &expected_ent_ct16);
        assert_eq!(ent_tag, MorphicTag::Commit256(expected_ent_tag32));
        decrypt_in_place(&key, &nonce, ad, Profile::EnterpriseCmt256, &mut msg, &ent_tag).unwrap();
        assert_eq!(msg, orig);

        let cmp_tag = encrypt_in_place(&key, &nonce, ad, Profile::Compact128, &mut msg).unwrap();
        let expected_cmp_ct16: [u8; 16] = [
            0xF5, 0xA0, 0x47, 0x88, 0xF4, 0x56, 0x0B, 0xCF, 0xF2, 0x73, 0x76, 0x38, 0x12, 0x3E,
            0x7C, 0x90,
        ];
        let expected_cmp_tag16: [u8; 16] = [
            0x64, 0x03, 0x86, 0x6E, 0x4F, 0xE5, 0x8D, 0xD5, 0x5D, 0x94, 0x1B, 0xBD, 0xEA, 0xB4,
            0x7D, 0xF4,
        ];
        assert_eq!(&msg[..16], &expected_cmp_ct16);
        assert_eq!(cmp_tag, MorphicTag::Compact128(expected_cmp_tag16));
        decrypt_in_place(&key, &nonce, ad, Profile::Compact128, &mut msg, &cmp_tag).unwrap();
        assert_eq!(msg, orig);
    }

    #[test]
    fn test_reduced_round_avalanche_p1_to_p6() {
        // Verifies full 512-bit diffusion by Round 4 for both 1-bit (e0) and 2-bit even-weight (e0 ^ e2)
        let trials = 256u64;
        let mut sum_1bit = [0u64; 6];
        let mut sum_2bit = [0u64; 6];

        for t in 0..trials {
            let seed = t.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            let mut a = state::MorphicState::new([
                seed,
                seed.rotate_left(7),
                seed.rotate_left(17),
                seed.rotate_left(29),
                seed ^ 0xA5A5_A5A5_5A5A_5A5A,
                seed.rotate_right(11),
                seed.rotate_right(23),
                seed.rotate_right(37),
            ]);
            let mut b1 = a.clone();
            let mut b2 = a.clone();
            b1.lanes[0] ^= 1u64;
            b2.lanes[5] ^= (1u64 << 0) | (1u64 << 2);

            for r in 0..6 {
                backend::scalar::permute_single_round(&mut a, r);
                backend::scalar::permute_single_round(&mut b1, r);
                backend::scalar::permute_single_round(&mut b2, r);
                let mut w1 = 0u32;
                let mut w2 = 0u32;
                for i in 0..8 {
                    w1 += (a.lanes[i] ^ b1.lanes[i]).count_ones();
                    w2 += (a.lanes[i] ^ b2.lanes[i]).count_ones();
                }
                sum_1bit[r] += w1 as u64;
                sum_2bit[r] += w2 as u64;
            }
        }

        // Round 1 must already diffuse strongly even for 2-bit even-weight e0 ^ e2 (PF1 fix)
        let avg_2bit_r1 = (sum_2bit[0] as f64) / (trials as f64);
        assert!(avg_2bit_r1 > 80.0, "PF1 failed R1 diffusion: {}", avg_2bit_r1);

        // Rounds 4, 5, 6 must saturate near 256 / 512 for both 1-bit and 2-bit even-weight
        for r in 3..6 {
            let avg1 = (sum_1bit[r] as f64) / (trials as f64);
            let avg2 = (sum_2bit[r] as f64) / (trials as f64);
            assert!(avg1 > 240.0 && avg1 < 272.0, "1-bit R{} avg={}", r + 1, avg1);
            assert!(avg2 > 240.0 && avg2 < 272.0, "2-bit R{} avg={}", r + 1, avg2);
        }
    }

    #[test]
    fn test_7_chunk_hybrid_simd_vs_pure_scalar() {
        let key = [0x3Cu8; KEY_LEN];
        let nonce = [0x7Au8; NONCE_LEN];
        let ad = b"7-chunk-hybrid-simd-v31";

        let total_len = CHUNK_SIZE * 6 + 1337;
        let orig: vec::Vec<u8> = (0..total_len)
            .map(|i| ((i.wrapping_mul(131) ^ (i >> 7) ^ 0x5A) & 0xFF) as u8)
            .collect();

        for &profile in &[
            Profile::EnterpriseCmt256,
            Profile::Compact128,
            Profile::HighSecCmt256,
        ] {
            let mut hybrid_buf = orig.clone();
            let mut scalar_buf = orig.clone();

            let hybrid_tag =
                encrypt_in_place(&key, &nonce, ad, profile, &mut hybrid_buf).unwrap();
            let scalar_tag =
                stream::encrypt_in_place_scalar(&key, &nonce, ad, profile, &mut scalar_buf)
                    .unwrap();

            assert_eq!(hybrid_buf, scalar_buf);
            assert_eq!(hybrid_tag, scalar_tag);

            decrypt_in_place(&key, &nonce, ad, profile, &mut hybrid_buf, &hybrid_tag).unwrap();
            assert_eq!(hybrid_buf, orig);
        }
    }

    #[test]
    fn test_7_chunk_framed_streaming_4way_and_tamper_wipe() {
        let key = [0x91u8; KEY_LEN];
        let nonce = [0x24u8; NONCE_LEN];
        let ad = b"framed-stream-7-chunks-v31";

        let mut chunks: vec::Vec<vec::Vec<u8>> = (0..6)
            .map(|c| vec![(0x10 + c) as u8; CHUNK_SIZE])
            .collect();
        chunks.push(vec![0x77u8; 4096]);
        let orig = chunks.clone();

        let mut enc = StreamEncryptor::new(&key, &nonce, ad, Profile::EnterpriseCmt256);
        let tags_0_3 = {
            let (first4, _) = chunks.split_at_mut(4);
            let (c0, r1) = first4.split_at_mut(1);
            let (c1, r2) = r1.split_at_mut(1);
            let (c2, c3) = r2.split_at_mut(1);
            enc.encrypt_4_chunks(
                [&mut c0[0], &mut c1[0], &mut c2[0], &mut c3[0]],
                false,
            )
            .unwrap()
        };
        let t4 = enc.encrypt_next_chunk(&mut chunks[4], false).unwrap();
        let t5 = enc.encrypt_next_chunk(&mut chunks[5], false).unwrap();
        let t6 = enc.encrypt_next_chunk(&mut chunks[6], true).unwrap();
        let master = enc.finish().unwrap();

        let mut flat: vec::Vec<u8> = orig.iter().flatten().copied().collect();
        let flat_master =
            stream::encrypt_in_place_scalar(&key, &nonce, ad, Profile::EnterpriseCmt256, &mut flat)
                .unwrap();
        assert_eq!(master, flat_master);

        let mut dec = StreamDecryptor::new(&key, &nonce, ad, Profile::EnterpriseCmt256);
        {
            let (first4, _) = chunks.split_at_mut(4);
            let (c0, r1) = first4.split_at_mut(1);
            let (c1, r2) = r1.split_at_mut(1);
            let (c2, c3) = r2.split_at_mut(1);
            dec.decrypt_4_chunks(
                [&mut c0[0], &mut c1[0], &mut c2[0], &mut c3[0]],
                &tags_0_3,
                false,
            )
            .unwrap();
        }
        dec.decrypt_next_chunk(&mut chunks[4], &t4, false).unwrap();
        dec.decrypt_next_chunk(&mut chunks[5], &t5, false).unwrap();
        dec.decrypt_next_chunk(&mut chunks[6], &t6, true).unwrap();
        dec.finish_and_verify(&master).unwrap();
        assert_eq!(chunks, orig);
    }

    #[test]
    fn test_kdf_memory_hard_and_iterated() {
        let salt = [0xABu8; NONCE_LEN];
        let mut salt2 = salt;
        salt2[0] ^= 0x01;

        let mut mem1 = vec![[0u64; 8]; 256];
        let k1 = kdf::derive_key_memory_hard(b"correct-horse", &salt, &mut mem1, 2).unwrap();
        assert!(mem1.iter().all(|blk| *blk == [0u64; 8]));

        let mut mem2 = vec![[0u64; 8]; 256];
        let k1_again = kdf::derive_key_memory_hard(b"correct-horse", &salt, &mut mem2, 2).unwrap();
        assert_eq!(k1, k1_again);

        let mut mem3 = vec![[0u64; 8]; 256];
        let k_diff_salt =
            kdf::derive_key_memory_hard(b"correct-horse", &salt2, &mut mem3, 2).unwrap();
        assert_ne!(k1, k_diff_salt);

        let ki = kdf::derive_key_iterated(b"correct-horse", &salt, 64).unwrap();
        assert_ne!(ki, [0u8; KEY_LEN]);
    }

    #[test]
    fn test_c_ffi_round_trip_and_auth_wipe() {
        let key = [0x77u8; KEY_LEN];
        let nonce = [0x88u8; NONCE_LEN];
        let ad = b"ffi-ad";
        let orig = *b"hello morphic sponge arx-512 v3.1 via ffi!";
        let mut buf = orig;
        let mut tag = [0u8; 32];

        unsafe {
            let rc = ffi::morphic_v3_encrypt_in_place(
                key.as_ptr(),
                nonce.as_ptr(),
                ad.as_ptr(),
                ad.len(),
                0x0312,
                buf.as_mut_ptr(),
                buf.len(),
                tag.as_mut_ptr(),
                tag.len(),
            );
            assert_eq!(rc, ffi::MORPHIC_OK);

            let rc2 = ffi::morphic_v3_decrypt_in_place(
                key.as_ptr(),
                nonce.as_ptr(),
                ad.as_ptr(),
                ad.len(),
                0x0312,
                buf.as_mut_ptr(),
                buf.len(),
                tag.as_ptr(),
                tag.len(),
            );
            assert_eq!(rc2, ffi::MORPHIC_OK);
            assert_eq!(buf, orig);

            tag[0] ^= 0x01;
            let rc3 = ffi::morphic_v3_decrypt_in_place(
                key.as_ptr(),
                nonce.as_ptr(),
                ad.as_ptr(),
                ad.len(),
                0x0312,
                buf.as_mut_ptr(),
                buf.len(),
                tag.as_ptr(),
                tag.len(),
            );
            assert_eq!(rc3, ffi::MORPHIC_ERR_AUTH_FAILED);
            assert!(buf.iter().all(|&b| b == 0));
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_direct_runtime_equivalence() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut s_avx = state::SoaState4::zeroed();
        let mut s_sca = state::SoaState4::zeroed();
        for l in 0..8 {
            for c in 0..4 {
                let v = (c as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ ((l as u64) << 59)
                    ^ ((c as u64) << 62);
                s_avx.lanes[l][c] = v;
                s_sca.lanes[l][c] = v;
            }
        }
        unsafe {
            backend::avx2::permute_14_4way(&mut s_avx);
        }
        backend::scalar::permute_14_4way(&mut s_sca);
        assert_eq!(s_avx.lanes, s_sca.lanes);

        unsafe {
            backend::avx2::permute_16_4way(&mut s_avx);
        }
        backend::scalar::permute_16_4way(&mut s_sca);
        assert_eq!(s_avx.lanes, s_sca.lanes);
    }

    #[test]
    fn test_stream_decryptor_poisoning_and_kdf_non_pow2() {
        let key = [0x33u8; KEY_LEN];
        let nonce = [0x44u8; NONCE_LEN];
        let ad = b"poison-test-v311";

        let mut buf = vec![0xAAu8; 128];
        let mut enc = StreamEncryptor::new(&key, &nonce, ad, Profile::EnterpriseCmt256);
        let tag = enc.encrypt_next_chunk(&mut buf, true).unwrap();
        let _master = enc.finish().unwrap();

        let mut bad_tag_bytes = [0u8; 32];
        bad_tag_bytes.copy_from_slice(tag.as_bytes());
        bad_tag_bytes[31] ^= 0x01;
        let bad_tag = MorphicTag::Commit256(bad_tag_bytes);

        let mut dec = StreamDecryptor::new(&key, &nonce, ad, Profile::EnterpriseCmt256);
        let mut ct_copy = buf.clone();
        let err1 = dec.decrypt_next_chunk(&mut ct_copy, &bad_tag, true);
        assert_eq!(err1, Err(MorphicError::AuthenticationFailed));
        assert!(ct_copy.iter().all(|&b| b == 0));

        // Subsequent call on poisoned decryptor MUST be rejected even with valid tag
        let mut ct_retry = buf.clone();
        let err2 = dec.decrypt_next_chunk(&mut ct_retry, &tag, true);
        assert_eq!(err2, Err(MorphicError::StreamAlreadyFinalized));

        // Non-power-of-2 memory-hard KDF scratchpad (300 blocks) via division-free fast_range
        let mut mem = vec![[0u64; 8]; 300];
        let k_a = kdf::derive_key_memory_hard(b"pw-non-pow2", &nonce, &mut mem, 2).unwrap();
        assert!(mem.iter().all(|blk| *blk == [0u64; 8]));
        let mut mem2 = vec![[0u64; 8]; 300];
        let k_b = kdf::derive_key_memory_hard(b"pw-non-pow2", &nonce, &mut mem2, 2).unwrap();
        assert_eq!(k_a, k_b);
    }
}
