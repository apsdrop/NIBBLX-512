//! x86-64 AVX2 4-Way SoA v3.1.0 ("Seal-Pack") parallel permutation & fused 64 KB chunk kernel.
//!
//! Implements PF1 non-linear pre-fold `W = Y + ROR(Y, 17)` + 4-way interleaved SHL-fold,
//! $P_{14}$ data duplexing, and $P_{16}$ Init/Final/HighSec with 0 secret branches.

#![cfg(target_arch = "x86_64")]

use core::arch::x86_64::{
    __m256i, _mm256_add_epi64, _mm256_and_si256, _mm256_andnot_si256, _mm256_cmpgt_epi64,
    _mm256_loadu_si256, _mm256_or_si256, _mm256_permute2x128_si256, _mm256_set1_epi64x,
    _mm256_setzero_si256, _mm256_slli_epi64, _mm256_srli_epi64, _mm256_storeu_si256,
    _mm256_unpackhi_epi64, _mm256_unpacklo_epi64, _mm256_xor_si256,
};
use core::hint::black_box;

use crate::constants::{CASE_FULL, RC};
use crate::state::SoaState4;
use crate::{CHUNK_SIZE, RATE_BYTES};

macro_rules! ror64_x4 {
    ($x:expr, $r:literal, $l:literal) => {
        _mm256_or_si256(_mm256_srli_epi64::<$r>($x), _mm256_slli_epi64::<$l>($x))
    };
}

macro_rules! rol64_x4 {
    ($x:expr, $l:literal, $r:literal) => {
        _mm256_or_si256(_mm256_slli_epi64::<$l>($x), _mm256_srli_epi64::<$r>($x))
    };
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn transpose4x4_u64(a: &mut __m256i, b: &mut __m256i, c: &mut __m256i, d: &mut __m256i) {
    let t0 = _mm256_unpacklo_epi64(*a, *b);
    let t1 = _mm256_unpackhi_epi64(*a, *b);
    let t2 = _mm256_unpacklo_epi64(*c, *d);
    let t3 = _mm256_unpackhi_epi64(*c, *d);
    *a = _mm256_permute2x128_si256::<0x20>(t0, t2);
    *b = _mm256_permute2x128_si256::<0x20>(t1, t3);
    *c = _mm256_permute2x128_si256::<0x31>(t0, t2);
    *d = _mm256_permute2x128_si256::<0x31>(t1, t3);
}

/// 4-way interleaved PF1 non-linear pre-fold + SHL-fold + MSB-only `_mm256_cmpgt_epi64`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn ct_mr2_v31_quad_x4(
    x0: &mut __m256i,
    x1: &mut __m256i,
    x2: &mut __m256i,
    x3: &mut __m256i,
    y0: __m256i,
    y1: __m256i,
    y2: __m256i,
    y3: __m256i,
    zero: __m256i,
) {
    // PF1 non-linear pre-fold: W = Y + ROR(Y, 17)
    let mut w0 = _mm256_add_epi64(y0, ror64_x4!(y0, 17, 47));
    let mut w1 = _mm256_add_epi64(y1, ror64_x4!(y1, 17, 47));
    let mut w2 = _mm256_add_epi64(y2, ror64_x4!(y2, 17, 47));
    let mut w3 = _mm256_add_epi64(y3, ror64_x4!(y3, 17, 47));

    w0 = _mm256_xor_si256(w0, _mm256_slli_epi64::<32>(w0));
    w1 = _mm256_xor_si256(w1, _mm256_slli_epi64::<32>(w1));
    w2 = _mm256_xor_si256(w2, _mm256_slli_epi64::<32>(w2));
    w3 = _mm256_xor_si256(w3, _mm256_slli_epi64::<32>(w3));

    w0 = _mm256_xor_si256(w0, _mm256_slli_epi64::<16>(w0));
    w1 = _mm256_xor_si256(w1, _mm256_slli_epi64::<16>(w1));
    w2 = _mm256_xor_si256(w2, _mm256_slli_epi64::<16>(w2));
    w3 = _mm256_xor_si256(w3, _mm256_slli_epi64::<16>(w3));

    w0 = _mm256_xor_si256(w0, _mm256_slli_epi64::<8>(w0));
    w1 = _mm256_xor_si256(w1, _mm256_slli_epi64::<8>(w1));
    w2 = _mm256_xor_si256(w2, _mm256_slli_epi64::<8>(w2));
    w3 = _mm256_xor_si256(w3, _mm256_slli_epi64::<8>(w3));

    w0 = _mm256_xor_si256(w0, _mm256_slli_epi64::<4>(w0));
    w1 = _mm256_xor_si256(w1, _mm256_slli_epi64::<4>(w1));
    w2 = _mm256_xor_si256(w2, _mm256_slli_epi64::<4>(w2));
    w3 = _mm256_xor_si256(w3, _mm256_slli_epi64::<4>(w3));

    w0 = _mm256_xor_si256(w0, _mm256_slli_epi64::<2>(w0));
    w1 = _mm256_xor_si256(w1, _mm256_slli_epi64::<2>(w1));
    w2 = _mm256_xor_si256(w2, _mm256_slli_epi64::<2>(w2));
    w3 = _mm256_xor_si256(w3, _mm256_slli_epi64::<2>(w3));

    macro_rules! apply_morph_x4 {
        ($dst:expr, $w:expr) => {{
            let mut a = ror64_x4!(*$dst, 11, 53);
            let m0 = _mm256_cmpgt_epi64(zero, $w);
            let m1 = _mm256_cmpgt_epi64(zero, _mm256_slli_epi64::<1>($w));
            let d0 = _mm256_xor_si256(a, ror64_x4!(a, 17, 47));
            a = _mm256_xor_si256(a, _mm256_and_si256(d0, m0));
            let d1 = _mm256_xor_si256(a, ror64_x4!(a, 31, 33));
            *$dst = _mm256_xor_si256(a, _mm256_and_si256(d1, m1));
        }};
    }

    apply_morph_x4!(x0, w0);
    apply_morph_x4!(x1, w1);
    apply_morph_x4!(x2, w2);
    apply_morph_x4!(x3, w3);
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn arx_mix_v3_x4(v: &mut [__m256i; 8]) {
    let t0 = _mm256_add_epi64(v[0], v[4]);
    v[0] = rol64_x4!(t0, 7, 57);
    v[4] = _mm256_xor_si256(ror64_x4!(v[4], 19, 45), t0);

    let t1 = _mm256_add_epi64(v[1], v[5]);
    v[1] = rol64_x4!(t1, 13, 51);
    v[5] = _mm256_xor_si256(ror64_x4!(v[5], 27, 37), t1);

    let t2 = _mm256_add_epi64(v[2], v[6]);
    v[2] = rol64_x4!(t2, 29, 35);
    v[6] = _mm256_xor_si256(ror64_x4!(v[6], 41, 23), t2);

    let t3 = _mm256_add_epi64(v[3], v[7]);
    v[3] = rol64_x4!(t3, 43, 21);
    v[7] = _mm256_xor_si256(ror64_x4!(v[7], 53, 11), t3);
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn chi_5_3_inplace_x4(v: &mut [__m256i; 8]) {
    let mut t0 = v[0];
    let mut t1 = v[1];

    v[0] = _mm256_xor_si256(v[0], _mm256_andnot_si256(v[1], v[2]));
    v[1] = _mm256_xor_si256(v[1], _mm256_andnot_si256(v[2], v[3]));
    v[2] = _mm256_xor_si256(v[2], _mm256_andnot_si256(v[3], v[4]));
    v[3] = _mm256_xor_si256(v[3], _mm256_andnot_si256(v[4], t0));
    v[4] = _mm256_xor_si256(v[4], _mm256_andnot_si256(t0, t1));

    t0 = v[5];
    t1 = v[6];
    v[5] = _mm256_xor_si256(v[5], _mm256_andnot_si256(v[6], v[7]));
    v[6] = _mm256_xor_si256(v[6], _mm256_andnot_si256(v[7], t0));
    v[7] = _mm256_xor_si256(v[7], _mm256_andnot_si256(t0, t1));
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn pi_v3_x4(v: &mut [__m256i; 8]) {
    let orig = *v;
    v[5] = orig[0];
    v[6] = orig[1];
    v[7] = orig[2];
    v[0] = orig[3];
    v[2] = orig[4];
    v[1] = orig[5];
    v[4] = orig[6];
    v[3] = orig[7];
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn round_even_x4<const R: usize>(v: &mut [__m256i; 8], zero: __m256i) {
    let rc = RC[R];
    v[0] = _mm256_xor_si256(v[0], _mm256_set1_epi64x(rc as i64));
    v[4] = _mm256_xor_si256(v[4], _mm256_set1_epi64x(rc.rotate_right(32) as i64));

    unsafe {
        arx_mix_v3_x4(v);

        let (y0, y1, y2, y3) = (v[5], v[6], v[7], v[4]);
        let [r0, r1, r2, r3, c0, c1, c2, c3] = v;
        ct_mr2_v31_quad_x4(r0, r1, r2, r3, y0, y1, y2, y3, zero);

        let (u0, u1, u2, u3) = (*r3, *r0, *r1, *r2);
        ct_mr2_v31_quad_x4(c0, c1, c2, c3, u0, u1, u2, u3, zero);

        chi_5_3_inplace_x4(v);
        pi_v3_x4(v);
    }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn round_odd_x4<const R: usize>(v: &mut [__m256i; 8]) {
    let rc = RC[R];
    v[0] = _mm256_xor_si256(v[0], _mm256_set1_epi64x(rc as i64));
    v[4] = _mm256_xor_si256(v[4], _mm256_set1_epi64x(rc.rotate_right(32) as i64));

    unsafe {
        arx_mix_v3_x4(v);
        chi_5_3_inplace_x4(v);
        pi_v3_x4(v);
    }
}

/// 14-round in-register AVX2 permutation ($P_{14}$, kept `#[inline]` for µop-cache residency).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn permute_14_inreg(v: &mut [__m256i; 8], zero: __m256i) {
    unsafe {
        round_even_x4::<0>(v, zero);
        round_odd_x4::<1>(v);
        round_even_x4::<2>(v, zero);
        round_odd_x4::<3>(v);
        round_even_x4::<4>(v, zero);
        round_odd_x4::<5>(v);
        round_even_x4::<6>(v, zero);
        round_odd_x4::<7>(v);
        round_even_x4::<8>(v, zero);
        round_odd_x4::<9>(v);
        round_even_x4::<10>(v, zero);
        round_odd_x4::<11>(v);
        round_even_x4::<12>(v, zero);
        round_odd_x4::<13>(v);
    }
}

/// 16-round in-register AVX2 permutation ($P_{16}$).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn permute_16_inreg(v: &mut [__m256i; 8], zero: __m256i) {
    unsafe {
        permute_14_inreg(v, zero);
        round_even_x4::<14>(v, zero);
        round_odd_x4::<15>(v);
    }
}

/// 4-way AVX2 v3.1 14-round permutation ($P_{14}$).
///
/// # Safety
/// Caller must guarantee CPU supports `avx2`.
#[target_feature(enable = "avx2")]
pub unsafe fn permute_14_4way(soa: &mut SoaState4) {
    let zero = _mm256_setzero_si256();
    let mut v = [zero; 8];
    unsafe {
        for i in 0..8 {
            v[i] = _mm256_loadu_si256(soa.lanes[i].as_ptr() as *const __m256i);
        }
        permute_14_inreg(&mut v, zero);
        for i in 0..8 {
            _mm256_storeu_si256(soa.lanes[i].as_mut_ptr() as *mut __m256i, v[i]);
            v[i] = zero;
        }
    }
    black_box(v);
}

/// 4-way AVX2 v3.1 16-round permutation ($P_{16}$).
///
/// # Safety
/// Caller must guarantee CPU supports `avx2`.
#[target_feature(enable = "avx2")]
pub unsafe fn permute_16_4way(soa: &mut SoaState4) {
    let zero = _mm256_setzero_si256();
    let mut v = [zero; 8];
    unsafe {
        for i in 0..8 {
            v[i] = _mm256_loadu_si256(soa.lanes[i].as_ptr() as *const __m256i);
        }
        permute_16_inreg(&mut v, zero);
        for i in 0..8 {
            _mm256_storeu_si256(soa.lanes[i].as_mut_ptr() as *mut __m256i, v[i]);
            v[i] = zero;
        }
    }
    black_box(v);
}

/// Fused register-resident AVX2 v3.1 kernel for 4 full 64 KB chunks.
///
/// # Safety
/// Caller must guarantee CPU supports `avx2` and every slice in `chunks` has `len() == CHUNK_SIZE`.
#[target_feature(enable = "avx2")]
pub unsafe fn process_4_full_chunks<const IS_DECRYPT: bool, const USE_P16: bool>(
    soa: &mut SoaState4,
    chunks: [&mut [u8]; 4],
    k: &[u64; 4],
    ad: &[u64; 4],
) {
    let zero = _mm256_setzero_si256();
    let mut v = [zero; 8];
    let p0 = chunks[0].as_mut_ptr();
    let p1 = chunks[1].as_mut_ptr();
    let p2 = chunks[2].as_mut_ptr();
    let p3 = chunks[3].as_mut_ptr();

    unsafe {
        for i in 0..8 {
            v[i] = _mm256_loadu_si256(soa.lanes[i].as_ptr() as *const __m256i);
        }

        permute_16_inreg(&mut v, zero);
        for i in 0..4 {
            v[4 + i] = _mm256_xor_si256(v[4 + i], _mm256_set1_epi64x(k[i] as i64));
        }
        for i in 0..4 {
            v[i] = _mm256_xor_si256(v[i], _mm256_set1_epi64x(ad[i] as i64));
        }
        if USE_P16 {
            permute_16_inreg(&mut v, zero);
        } else {
            permute_14_inreg(&mut v, zero);
        }

        let full_blocks = CHUNK_SIZE / RATE_BYTES;
        for b in 0..full_blocks {
            let off = b * RATE_BYTES;
            let mut w0 = _mm256_loadu_si256(p0.add(off) as *const __m256i);
            let mut w1 = _mm256_loadu_si256(p1.add(off) as *const __m256i);
            let mut w2 = _mm256_loadu_si256(p2.add(off) as *const __m256i);
            let mut w3 = _mm256_loadu_si256(p3.add(off) as *const __m256i);

            transpose4x4_u64(&mut w0, &mut w1, &mut w2, &mut w3);

            if !IS_DECRYPT {
                w0 = _mm256_xor_si256(w0, v[0]);
                v[0] = w0;
                w1 = _mm256_xor_si256(w1, v[1]);
                v[1] = w1;
                w2 = _mm256_xor_si256(w2, v[2]);
                v[2] = w2;
                w3 = _mm256_xor_si256(w3, v[3]);
                v[3] = w3;
            } else {
                let ct0 = w0;
                let ct1 = w1;
                let ct2 = w2;
                let ct3 = w3;
                w0 = _mm256_xor_si256(ct0, v[0]);
                v[0] = ct0;
                w1 = _mm256_xor_si256(ct1, v[1]);
                v[1] = ct1;
                w2 = _mm256_xor_si256(ct2, v[2]);
                v[2] = ct2;
                w3 = _mm256_xor_si256(ct3, v[3]);
                v[3] = ct3;
            }

            transpose4x4_u64(&mut w0, &mut w1, &mut w2, &mut w3);

            _mm256_storeu_si256(p0.add(off) as *mut __m256i, w0);
            _mm256_storeu_si256(p1.add(off) as *mut __m256i, w1);
            _mm256_storeu_si256(p2.add(off) as *mut __m256i, w2);
            _mm256_storeu_si256(p3.add(off) as *mut __m256i, w3);

            if b + 1 < full_blocks {
                if USE_P16 {
                    permute_16_inreg(&mut v, zero);
                } else {
                    permute_14_inreg(&mut v, zero);
                }
            }
        }

        v[6] = _mm256_xor_si256(v[6], _mm256_set1_epi64x(CHUNK_SIZE as i64));
        v[7] = _mm256_xor_si256(v[7], _mm256_set1_epi64x(CASE_FULL as i64));
        for i in 0..4 {
            v[4 + i] = _mm256_xor_si256(v[4 + i], _mm256_set1_epi64x(k[i] as i64));
        }
        permute_16_inreg(&mut v, zero);

        for i in 0..4 {
            _mm256_storeu_si256(soa.lanes[i].as_mut_ptr() as *mut __m256i, v[i]);
        }
        for i in 0..8 {
            v[i] = zero;
        }
    }
    black_box(v);
}
