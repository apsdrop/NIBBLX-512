//! ARM64 NEON 4-Way (2 x 128-bit) v3.1.0 ("Seal-Pack") parallel permutation & fused 64 KB kernel.
//!
//! Implements PF1 non-linear pre-fold `W = Y + ROR(Y, 17)` + 4-way interleaved `vshlq_n_u64` fold,
//! $P_{14}$ data duplexing, and $P_{16}$ Init/Final/HighSec.

#![cfg(target_arch = "aarch64")]

use core::arch::aarch64::{
    int64x2_t, uint64x2_t, vaddq_u64, vandq_u64, vbicq_u64, vcltq_s64, vdupq_n_s64, vdupq_n_u64,
    veorq_u64, vld1q_u64, vreinterpretq_s64_u64, vshlq_n_u64, vsriq_n_u64, vst1q_u64, vtrn1q_u64,
    vtrn2q_u64,
};
use core::hint::black_box;

use crate::constants::{CASE_FULL, RC};
use crate::state::SoaState4;
use crate::{CHUNK_SIZE, RATE_BYTES};

macro_rules! ror64_x2 {
    ($x:expr, $r:literal, $l:literal) => {
        vsriq_n_u64::<$r>(vshlq_n_u64::<$l>($x), $x)
    };
}

macro_rules! rol64_x2 {
    ($x:expr, $l:literal, $r:literal) => {
        vsriq_n_u64::<$r>(vshlq_n_u64::<$l>($x), $x)
    };
}

/// 4-way interleaved PF1 non-linear pre-fold + `vshlq_n_u64` fold + MSB-only `vcltq_s64`.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn ct_mr2_v31_quad_x2(
    x0: &mut uint64x2_t,
    x1: &mut uint64x2_t,
    x2: &mut uint64x2_t,
    x3: &mut uint64x2_t,
    y0: uint64x2_t,
    y1: uint64x2_t,
    y2: uint64x2_t,
    y3: uint64x2_t,
    zero_s: int64x2_t,
) {
    // PF1 non-linear pre-fold: W = Y + ROR(Y, 17)
    let mut w0 = vaddq_u64(y0, ror64_x2!(y0, 17, 47));
    let mut w1 = vaddq_u64(y1, ror64_x2!(y1, 17, 47));
    let mut w2 = vaddq_u64(y2, ror64_x2!(y2, 17, 47));
    let mut w3 = vaddq_u64(y3, ror64_x2!(y3, 17, 47));

    w0 = veorq_u64(w0, vshlq_n_u64::<32>(w0));
    w1 = veorq_u64(w1, vshlq_n_u64::<32>(w1));
    w2 = veorq_u64(w2, vshlq_n_u64::<32>(w2));
    w3 = veorq_u64(w3, vshlq_n_u64::<32>(w3));

    w0 = veorq_u64(w0, vshlq_n_u64::<16>(w0));
    w1 = veorq_u64(w1, vshlq_n_u64::<16>(w1));
    w2 = veorq_u64(w2, vshlq_n_u64::<16>(w2));
    w3 = veorq_u64(w3, vshlq_n_u64::<16>(w3));

    w0 = veorq_u64(w0, vshlq_n_u64::<8>(w0));
    w1 = veorq_u64(w1, vshlq_n_u64::<8>(w1));
    w2 = veorq_u64(w2, vshlq_n_u64::<8>(w2));
    w3 = veorq_u64(w3, vshlq_n_u64::<8>(w3));

    w0 = veorq_u64(w0, vshlq_n_u64::<4>(w0));
    w1 = veorq_u64(w1, vshlq_n_u64::<4>(w1));
    w2 = veorq_u64(w2, vshlq_n_u64::<4>(w2));
    w3 = veorq_u64(w3, vshlq_n_u64::<4>(w3));

    w0 = veorq_u64(w0, vshlq_n_u64::<2>(w0));
    w1 = veorq_u64(w1, vshlq_n_u64::<2>(w1));
    w2 = veorq_u64(w2, vshlq_n_u64::<2>(w2));
    w3 = veorq_u64(w3, vshlq_n_u64::<2>(w3));

    macro_rules! apply_morph_x2 {
        ($dst:expr, $w:expr) => {{
            let mut a = ror64_x2!(*$dst, 11, 53);
            let m0 = vcltq_s64(vreinterpretq_s64_u64($w), zero_s);
            let m1 = vcltq_s64(vreinterpretq_s64_u64(vshlq_n_u64::<1>($w)), zero_s);
            let d0 = veorq_u64(a, ror64_x2!(a, 17, 47));
            a = veorq_u64(a, vandq_u64(d0, m0));
            let d1 = veorq_u64(a, ror64_x2!(a, 31, 33));
            *$dst = veorq_u64(a, vandq_u64(d1, m1));
        }};
    }

    apply_morph_x2!(x0, w0);
    apply_morph_x2!(x1, w1);
    apply_morph_x2!(x2, w2);
    apply_morph_x2!(x3, w3);
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn arx_mix_v3_x2(v: &mut [uint64x2_t; 8]) {
    let t0 = vaddq_u64(v[0], v[4]);
    v[0] = rol64_x2!(t0, 7, 57);
    v[4] = veorq_u64(ror64_x2!(v[4], 19, 45), t0);

    let t1 = vaddq_u64(v[1], v[5]);
    v[1] = rol64_x2!(t1, 13, 51);
    v[5] = veorq_u64(ror64_x2!(v[5], 27, 37), t1);

    let t2 = vaddq_u64(v[2], v[6]);
    v[2] = rol64_x2!(t2, 29, 35);
    v[6] = veorq_u64(ror64_x2!(v[6], 41, 23), t2);

    let t3 = vaddq_u64(v[3], v[7]);
    v[3] = rol64_x2!(t3, 43, 21);
    v[7] = veorq_u64(ror64_x2!(v[7], 53, 11), t3);
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn chi_5_3_inplace_x2(v: &mut [uint64x2_t; 8]) {
    let mut t0 = v[0];
    let mut t1 = v[1];

    v[0] = veorq_u64(v[0], vbicq_u64(v[2], v[1]));
    v[1] = veorq_u64(v[1], vbicq_u64(v[3], v[2]));
    v[2] = veorq_u64(v[2], vbicq_u64(v[4], v[3]));
    v[3] = veorq_u64(v[3], vbicq_u64(t0, v[4]));
    v[4] = veorq_u64(v[4], vbicq_u64(t1, t0));

    t0 = v[5];
    t1 = v[6];
    v[5] = veorq_u64(v[5], vbicq_u64(v[7], v[6]));
    v[6] = veorq_u64(v[6], vbicq_u64(t0, v[7]));
    v[7] = veorq_u64(v[7], vbicq_u64(t1, t0));
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn pi_v3_x2(v: &mut [uint64x2_t; 8]) {
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

#[target_feature(enable = "neon")]
#[inline]
unsafe fn round_even_x2<const R: usize>(v: &mut [uint64x2_t; 8], zero_s: int64x2_t) {
    let rc = RC[R];
    v[0] = veorq_u64(v[0], vdupq_n_u64(rc));
    v[4] = veorq_u64(v[4], vdupq_n_u64(rc.rotate_right(32)));

    unsafe {
        arx_mix_v3_x2(v);

        let (y0, y1, y2, y3) = (v[5], v[6], v[7], v[4]);
        let [r0, r1, r2, r3, c0, c1, c2, c3] = v;
        ct_mr2_v31_quad_x2(r0, r1, r2, r3, y0, y1, y2, y3, zero_s);

        let (u0, u1, u2, u3) = (*r3, *r0, *r1, *r2);
        ct_mr2_v31_quad_x2(c0, c1, c2, c3, u0, u1, u2, u3, zero_s);

        chi_5_3_inplace_x2(v);
        pi_v3_x2(v);
    }
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn round_odd_x2<const R: usize>(v: &mut [uint64x2_t; 8]) {
    let rc = RC[R];
    v[0] = veorq_u64(v[0], vdupq_n_u64(rc));
    v[4] = veorq_u64(v[4], vdupq_n_u64(rc.rotate_right(32)));

    unsafe {
        arx_mix_v3_x2(v);
        chi_5_3_inplace_x2(v);
        pi_v3_x2(v);
    }
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn run_14_x2(v: &mut [uint64x2_t; 8], zero_s: int64x2_t) {
    unsafe {
        round_even_x2::<0>(v, zero_s);
        round_odd_x2::<1>(v);
        round_even_x2::<2>(v, zero_s);
        round_odd_x2::<3>(v);
        round_even_x2::<4>(v, zero_s);
        round_odd_x2::<5>(v);
        round_even_x2::<6>(v, zero_s);
        round_odd_x2::<7>(v);
        round_even_x2::<8>(v, zero_s);
        round_odd_x2::<9>(v);
        round_even_x2::<10>(v, zero_s);
        round_odd_x2::<11>(v);
        round_even_x2::<12>(v, zero_s);
        round_odd_x2::<13>(v);
    }
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn run_16_x2(v: &mut [uint64x2_t; 8], zero_s: int64x2_t) {
    unsafe {
        run_14_x2(v, zero_s);
        round_even_x2::<14>(v, zero_s);
        round_odd_x2::<15>(v);
    }
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn duplex_block_x2<const IS_DECRYPT: bool>(
    v: &mut [uint64x2_t; 8],
    pa: *mut u8,
    pb: *mut u8,
    off: usize,
) {
    unsafe {
        let a01 = vld1q_u64(pa.add(off) as *const u64);
        let b01 = vld1q_u64(pb.add(off) as *const u64);
        let a23 = vld1q_u64(pa.add(off + 16) as *const u64);
        let b23 = vld1q_u64(pb.add(off + 16) as *const u64);

        let mut w0 = vtrn1q_u64(a01, b01);
        let mut w1 = vtrn2q_u64(a01, b01);
        let mut w2 = vtrn1q_u64(a23, b23);
        let mut w3 = vtrn2q_u64(a23, b23);

        if !IS_DECRYPT {
            w0 = veorq_u64(w0, v[0]);
            v[0] = w0;
            w1 = veorq_u64(w1, v[1]);
            v[1] = w1;
            w2 = veorq_u64(w2, v[2]);
            v[2] = w2;
            w3 = veorq_u64(w3, v[3]);
            v[3] = w3;
        } else {
            let ct0 = w0;
            let ct1 = w1;
            let ct2 = w2;
            let ct3 = w3;
            w0 = veorq_u64(ct0, v[0]);
            v[0] = ct0;
            w1 = veorq_u64(ct1, v[1]);
            v[1] = ct1;
            w2 = veorq_u64(ct2, v[2]);
            v[2] = ct2;
            w3 = veorq_u64(ct3, v[3]);
            v[3] = ct3;
        }

        vst1q_u64(pa.add(off) as *mut u64, vtrn1q_u64(w0, w1));
        vst1q_u64(pb.add(off) as *mut u64, vtrn2q_u64(w0, w1));
        vst1q_u64(pa.add(off + 16) as *mut u64, vtrn1q_u64(w2, w3));
        vst1q_u64(pb.add(off + 16) as *mut u64, vtrn2q_u64(w2, w3));
    }
}

/// 4-way ARM64 NEON v3.1 14-round permutation ($P_{14}$).
///
/// # Safety
/// Caller must guarantee `aarch64` NEON availability.
#[target_feature(enable = "neon")]
pub unsafe fn permute_14_4way(soa: &mut SoaState4) {
    let zero_u = vdupq_n_u64(0);
    let zero_s = vdupq_n_s64(0);
    let mut lo = [zero_u; 8];
    let mut hi = [zero_u; 8];

    unsafe {
        for i in 0..8 {
            let ptr = soa.lanes[i].as_ptr();
            lo[i] = vld1q_u64(ptr);
            hi[i] = vld1q_u64(ptr.add(2));
        }
        run_14_x2(&mut lo, zero_s);
        run_14_x2(&mut hi, zero_s);
        for i in 0..8 {
            let ptr = soa.lanes[i].as_mut_ptr();
            vst1q_u64(ptr, lo[i]);
            vst1q_u64(ptr.add(2), hi[i]);
            lo[i] = zero_u;
            hi[i] = zero_u;
        }
    }
    black_box((lo, hi));
}

/// 4-way ARM64 NEON v3.1 16-round permutation ($P_{16}$).
///
/// # Safety
/// Caller must guarantee `aarch64` NEON availability.
#[target_feature(enable = "neon")]
pub unsafe fn permute_16_4way(soa: &mut SoaState4) {
    let zero_u = vdupq_n_u64(0);
    let zero_s = vdupq_n_s64(0);
    let mut lo = [zero_u; 8];
    let mut hi = [zero_u; 8];

    unsafe {
        for i in 0..8 {
            let ptr = soa.lanes[i].as_ptr();
            lo[i] = vld1q_u64(ptr);
            hi[i] = vld1q_u64(ptr.add(2));
        }
        run_16_x2(&mut lo, zero_s);
        run_16_x2(&mut hi, zero_s);
        for i in 0..8 {
            let ptr = soa.lanes[i].as_mut_ptr();
            vst1q_u64(ptr, lo[i]);
            vst1q_u64(ptr.add(2), hi[i]);
            lo[i] = zero_u;
            hi[i] = zero_u;
        }
    }
    black_box((lo, hi));
}

/// Fused register-resident ARM64 NEON v3.1 kernel for 4 full 64 KB chunks.
///
/// # Safety
/// Caller must guarantee `aarch64` NEON availability and every slice has `len() == CHUNK_SIZE`.
#[target_feature(enable = "neon")]
pub unsafe fn process_4_full_chunks<const IS_DECRYPT: bool, const USE_P16: bool>(
    soa: &mut SoaState4,
    chunks: [&mut [u8]; 4],
    k: &[u64; 4],
    ad: &[u64; 4],
) {
    let zero_u = vdupq_n_u64(0);
    let zero_s = vdupq_n_s64(0);
    let kv = [
        vdupq_n_u64(k[0]),
        vdupq_n_u64(k[1]),
        vdupq_n_u64(k[2]),
        vdupq_n_u64(k[3]),
    ];
    let adv = [
        vdupq_n_u64(ad[0]),
        vdupq_n_u64(ad[1]),
        vdupq_n_u64(ad[2]),
        vdupq_n_u64(ad[3]),
    ];

    let mut lo = [zero_u; 8];
    let mut hi = [zero_u; 8];
    let p0 = chunks[0].as_mut_ptr();
    let p1 = chunks[1].as_mut_ptr();
    let p2 = chunks[2].as_mut_ptr();
    let p3 = chunks[3].as_mut_ptr();

    unsafe {
        for i in 0..8 {
            let ptr = soa.lanes[i].as_ptr();
            lo[i] = vld1q_u64(ptr);
            hi[i] = vld1q_u64(ptr.add(2));
        }

        run_16_x2(&mut lo, zero_s);
        run_16_x2(&mut hi, zero_s);
        for i in 0..4 {
            lo[4 + i] = veorq_u64(lo[4 + i], kv[i]);
            hi[4 + i] = veorq_u64(hi[4 + i], kv[i]);
            lo[i] = veorq_u64(lo[i], adv[i]);
            hi[i] = veorq_u64(hi[i], adv[i]);
        }
        if USE_P16 {
            run_16_x2(&mut lo, zero_s);
            run_16_x2(&mut hi, zero_s);
        } else {
            run_14_x2(&mut lo, zero_s);
            run_14_x2(&mut hi, zero_s);
        }

        let full_blocks = CHUNK_SIZE / RATE_BYTES;
        for b in 0..full_blocks {
            let off = b * RATE_BYTES;
            duplex_block_x2::<IS_DECRYPT>(&mut lo, p0, p1, off);
            duplex_block_x2::<IS_DECRYPT>(&mut hi, p2, p3, off);

            if b + 1 < full_blocks {
                if USE_P16 {
                    run_16_x2(&mut lo, zero_s);
                    run_16_x2(&mut hi, zero_s);
                } else {
                    run_14_x2(&mut lo, zero_s);
                    run_14_x2(&mut hi, zero_s);
                }
            }
        }

        let len_v = vdupq_n_u64(CHUNK_SIZE as u64);
        let case_v = vdupq_n_u64(CASE_FULL);
        lo[6] = veorq_u64(lo[6], len_v);
        hi[6] = veorq_u64(hi[6], len_v);
        lo[7] = veorq_u64(lo[7], case_v);
        hi[7] = veorq_u64(hi[7], case_v);
        for i in 0..4 {
            lo[4 + i] = veorq_u64(lo[4 + i], kv[i]);
            hi[4 + i] = veorq_u64(hi[4 + i], kv[i]);
        }
        run_16_x2(&mut lo, zero_s);
        run_16_x2(&mut hi, zero_s);

        for i in 0..4 {
            let ptr = soa.lanes[i].as_mut_ptr();
            vst1q_u64(ptr, lo[i]);
            vst1q_u64(ptr.add(2), hi[i]);
        }
        for i in 0..8 {
            lo[i] = zero_u;
            hi[i] = zero_u;
        }
    }
    black_box((lo, hi));
}
