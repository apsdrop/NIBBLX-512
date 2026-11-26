//! Backend dispatcher for Scalar, x86-64 AVX2 (compile-time + `#![no_std]` runtime CPUID),
//! and ARM64 NEON permutations & fused 4-chunk kernels.

pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(target_arch = "aarch64")]
pub mod neon;

use crate::state::SoaState4;

/// Zero-dependency `#![no_std]` runtime AVX2 + OSXSAVE + XCR0[2:1] detector (cached in 1 `AtomicU8`).
///
/// Only compiled on `x86_64` when `target_feature = "avx2"` is NOT already enabled at compile time.
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
#[inline(always)]
pub fn has_avx2_runtime() -> bool {
    use core::sync::atomic::{AtomicU8, Ordering};

    // 0 = uninitialized, 1 = no AVX2, 2 = AVX2 + OS YMM state verified
    static AVX2_STATE: AtomicU8 = AtomicU8::new(0);

    let cached = AVX2_STATE.load(Ordering::Relaxed);
    if cached != 0 {
        return cached == 2;
    }

    #[target_feature(enable = "xsave")]
    unsafe fn read_xcr0() -> u64 {
        unsafe { core::arch::x86_64::_xgetbv(0) }
    }

    let supported = unsafe {
        let leaf0 = core::arch::x86_64::__cpuid(0);
        if leaf0.eax < 7 {
            false
        } else {
            let leaf1 = core::arch::x86_64::__cpuid(1);
            let osxsave = (leaf1.ecx & (1 << 27)) != 0;
            let avx = (leaf1.ecx & (1 << 28)) != 0;
            if !osxsave || !avx {
                false
            } else {
                // Verify OS saves/restores both XMM (bit 1) and YMM (bit 2) registers
                let xcr0 = read_xcr0();
                if (xcr0 & 0x06) != 0x06 {
                    false
                } else {
                    let leaf7 = core::arch::x86_64::__cpuid_count(7, 0);
                    (leaf7.ebx & (1 << 5)) != 0
                }
            }
        }
    };

    AVX2_STATE.store(if supported { 2 } else { 1 }, Ordering::Relaxed);
    supported
}

/// Optimal 4-way 10-round permutation dispatch (`#![no_std]` safe).
#[inline(always)]
pub fn permute_14_4way(state: &mut SoaState4) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { avx2::permute_14_4way(state) };
    }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    {
        if has_avx2_runtime() {
            // SAFETY: CPUID + OSXSAVE + XCR0[2:1] verified AVX2 support.
            unsafe { avx2::permute_14_4way(state) };
        } else {
            scalar::permute_14_4way(state);
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe { neon::permute_14_4way(state) };
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        scalar::permute_14_4way(state);
    }
}

/// Optimal 4-way 12-round permutation dispatch (`#![no_std]` safe).
#[inline(always)]
pub fn permute_16_4way(state: &mut SoaState4) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { avx2::permute_16_4way(state) };
    }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    {
        if has_avx2_runtime() {
            // SAFETY: CPUID + OSXSAVE + XCR0[2:1] verified AVX2 support.
            unsafe { avx2::permute_16_4way(state) };
        } else {
            scalar::permute_16_4way(state);
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe { neon::permute_16_4way(state) };
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        scalar::permute_16_4way(state);
    }
}

#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "aarch64", target_feature = "neon")
)))]
fn process_4_full_chunks_scalar<const IS_DECRYPT: bool>(
    soa: &mut SoaState4,
    chunks: [&mut [u8]; 4],
    k: &[u64; 4],
    ad: &[u64; 4],
    use_p16: bool,
) {
    use crate::constants::CASE_FULL;
    use crate::state::MorphicState;
    use crate::{CHUNK_SIZE, RATE_BYTES};

    let full_blocks = CHUNK_SIZE / RATE_BYTES;
    for q in 0..4 {
        let mut st = MorphicState::new([
            soa.lanes[0][q],
            soa.lanes[1][q],
            soa.lanes[2][q],
            soa.lanes[3][q],
            soa.lanes[4][q],
            soa.lanes[5][q],
            soa.lanes[6][q],
            soa.lanes[7][q],
        ]);
        scalar::permute_16(&mut st);
        st.xor_capacity_key(k);
        for i in 0..4 {
            st.lanes[i] ^= ad[i];
        }
        scalar::permute_data(&mut st, use_p16);
        for b in 0..full_blocks {
            let off = b * RATE_BYTES;
            let blk = &mut chunks[q][off..off + RATE_BYTES];
            for i in 0..4 {
                let w = MorphicState::load_le64(&blk[8 * i..]);
                if !IS_DECRYPT {
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
        st.lanes[6] ^= CHUNK_SIZE as u64;
        st.lanes[7] ^= CASE_FULL;
        st.xor_capacity_key(k);
        scalar::permute_16(&mut st);
        for i in 0..4 {
            soa.lanes[i][q] = st.lanes[i];
        }
    }
}

/// Dispatches fused register-resident 4-chunk 64 KB processing to AVX2 (compile-time or
/// `#![no_std]` runtime CPUID), NEON, or Scalar fallback.
#[inline(always)]
pub fn process_4_full_chunks<const IS_DECRYPT: bool>(
    soa: &mut SoaState4,
    chunks: [&mut [u8]; 4],
    k: &[u64; 4],
    ad: &[u64; 4],
    use_p16: bool,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        if use_p16 {
            unsafe { avx2::process_4_full_chunks::<IS_DECRYPT, true>(soa, chunks, k, ad) };
        } else {
            unsafe { avx2::process_4_full_chunks::<IS_DECRYPT, false>(soa, chunks, k, ad) };
        }
    }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    {
        if has_avx2_runtime() {
            // SAFETY: CPUID + OSXSAVE + XCR0[2:1] verified AVX2 support.
            if use_p16 {
            unsafe { avx2::process_4_full_chunks::<IS_DECRYPT, true>(soa, chunks, k, ad) };
        } else {
            unsafe { avx2::process_4_full_chunks::<IS_DECRYPT, false>(soa, chunks, k, ad) };
        }
        } else {
            process_4_full_chunks_scalar::<IS_DECRYPT>(soa, chunks, k, ad, use_p16);
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        if use_p16 {
            unsafe { neon::process_4_full_chunks::<IS_DECRYPT, true>(soa, chunks, k, ad) };
        } else {
            unsafe { neon::process_4_full_chunks::<IS_DECRYPT, false>(soa, chunks, k, ad) };
        }
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        process_4_full_chunks_scalar::<IS_DECRYPT>(soa, chunks, k, ad, use_p16);
    }
}
