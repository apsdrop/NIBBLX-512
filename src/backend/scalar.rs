//! Verified v3.1.0 ("Seal-Pack") zero-spill Scalar backend:
//! - Double-Rotate ARX ($\mathcal{A}_{v3}$)
//! - PF1 non-linear pre-fold `W = Y.wrapping_add(Y.rotate_right(17))` + 4-way interleaved SHL-fold
//! - `black_box`-hardened MSB-only `SAR 63` masks (`0` secret-dependent branches)
//! - 3-scratch in-place Split-Chi ($\chi_{5,3}$) + non-affine $\pi_{v3}$
//! - $P_{14}$ (data duplex) and $P_{16}$ (init, finalization, $H_K$, KDF, HighSec)

use core::hint::black_box;

use crate::constants::{
    ARX_RHO, ARX_SIGMA, MORPH_BASE_ROT, MORPH_STRIDE_0, MORPH_STRIDE_1, PREFOLD_ROT, RC,
};
use crate::state::{MorphicState, SoaState4};

/// 4-way interleaved PF1 non-linear pre-fold + SHL-fold + `black_box`-hardened SAR 63 masks.
#[inline(always)]
fn ct_mr2_v31_quad(
    x0: &mut u64,
    x1: &mut u64,
    x2: &mut u64,
    x3: &mut u64,
    y0: u64,
    y1: u64,
    y2: u64,
    y3: u64,
) {
    // PF1 non-linear pre-fold: breaks even-weight XOR-parity bypass
    let mut w0 = y0.wrapping_add(y0.rotate_right(PREFOLD_ROT));
    let mut w1 = y1.wrapping_add(y1.rotate_right(PREFOLD_ROT));
    let mut w2 = y2.wrapping_add(y2.rotate_right(PREFOLD_ROT));
    let mut w3 = y3.wrapping_add(y3.rotate_right(PREFOLD_ROT));

    // 5-step interleaved SHL-fold (32, 16, 8, 4, 2)
    w0 ^= w0 << 32;
    w1 ^= w1 << 32;
    w2 ^= w2 << 32;
    w3 ^= w3 << 32;

    w0 ^= w0 << 16;
    w1 ^= w1 << 16;
    w2 ^= w2 << 16;
    w3 ^= w3 << 16;

    w0 ^= w0 << 8;
    w1 ^= w1 << 8;
    w2 ^= w2 << 8;
    w3 ^= w3 << 8;

    w0 ^= w0 << 4;
    w1 ^= w1 << 4;
    w2 ^= w2 << 4;
    w3 ^= w3 << 4;

    w0 ^= w0 << 2;
    w1 ^= w1 << 2;
    w2 ^= w2 << 2;
    w3 ^= w3 << 2;

    let mut a0 = x0.rotate_right(MORPH_BASE_ROT);
    let mut a1 = x1.rotate_right(MORPH_BASE_ROT);
    let mut a2 = x2.rotate_right(MORPH_BASE_ROT);
    let mut a3 = x3.rotate_right(MORPH_BASE_ROT);

    // Stage 1: MSB bit 63 via signed SAR 63 (black_box prevents branchification)
    let m0_0 = black_box(((w0 as i64) >> 63) as u64);
    let m0_1 = black_box(((w1 as i64) >> 63) as u64);
    let m0_2 = black_box(((w2 as i64) >> 63) as u64);
    let m0_3 = black_box(((w3 as i64) >> 63) as u64);

    a0 ^= (a0 ^ a0.rotate_right(MORPH_STRIDE_0)) & m0_0;
    a1 ^= (a1 ^ a1.rotate_right(MORPH_STRIDE_0)) & m0_1;
    a2 ^= (a2 ^ a2.rotate_right(MORPH_STRIDE_0)) & m0_2;
    a3 ^= (a3 ^ a3.rotate_right(MORPH_STRIDE_0)) & m0_3;

    // Stage 2: Bit 62 via (w << 1) signed SAR 63
    let m1_0 = black_box((((w0 << 1) as i64) >> 63) as u64);
    let m1_1 = black_box((((w1 << 1) as i64) >> 63) as u64);
    let m1_2 = black_box((((w2 << 1) as i64) >> 63) as u64);
    let m1_3 = black_box((((w3 << 1) as i64) >> 63) as u64);

    a0 ^= (a0 ^ a0.rotate_right(MORPH_STRIDE_1)) & m1_0;
    a1 ^= (a1 ^ a1.rotate_right(MORPH_STRIDE_1)) & m1_1;
    a2 ^= (a2 ^ a2.rotate_right(MORPH_STRIDE_1)) & m1_2;
    a3 ^= (a3 ^ a3.rotate_right(MORPH_STRIDE_1)) & m1_3;

    *x0 = a0;
    *x1 = a1;
    *x2 = a2;
    *x3 = a3;
}

/// True 3-scratch in-place Split-Chi ($\chi_{5,3}$).
#[inline(always)]
fn chi_5_3_inplace(s: &mut [u64; 8]) {
    let mut t0 = s[0];
    let mut t1 = s[1];

    let mut t2 = (!s[1]) & s[2];
    s[0] ^= t2;
    t2 = (!s[2]) & s[3];
    s[1] ^= t2;
    t2 = (!s[3]) & s[4];
    s[2] ^= t2;
    t2 = (!s[4]) & t0;
    s[3] ^= t2;
    t2 = (!t0) & t1;
    s[4] ^= t2;

    t0 = s[5];
    t1 = s[6];
    t2 = (!s[6]) & s[7];
    s[5] ^= t2;
    t2 = (!s[7]) & t0;
    s[6] ^= t2;
    t2 = (!t0) & t1;
    s[7] ^= t2;
}

/// Non-Affine Pair-Breaking Lane Permutation ($\pi_{v3} = (5, 6, 7, 0, 2, 1, 4, 3)$).
#[inline(always)]
fn pi_v3(s: &mut [u64; 8]) {
    let orig = *s;
    s[5] = orig[0];
    s[6] = orig[1];
    s[7] = orig[2];
    s[0] = orig[3];
    s[2] = orig[4];
    s[1] = orig[5];
    s[4] = orig[6];
    s[3] = orig[7];
}

/// Full Double-Rotate ARX ($\mathcal{A}_{v3}$).
#[inline(always)]
fn arx_mix_v3(s: &mut [u64; 8]) {
    let t0 = s[0].wrapping_add(s[4]);
    s[0] = t0.rotate_left(ARX_SIGMA[0]);
    s[4] = s[4].rotate_right(ARX_RHO[0]) ^ t0;

    let t1 = s[1].wrapping_add(s[5]);
    s[1] = t1.rotate_left(ARX_SIGMA[1]);
    s[5] = s[5].rotate_right(ARX_RHO[1]) ^ t1;

    let t2 = s[2].wrapping_add(s[6]);
    s[2] = t2.rotate_left(ARX_SIGMA[2]);
    s[6] = s[6].rotate_right(ARX_RHO[2]) ^ t2;

    let t3 = s[3].wrapping_add(s[7]);
    s[3] = t3.rotate_left(ARX_SIGMA[3]);
    s[7] = s[7].rotate_right(ARX_RHO[3]) ^ t3;
}

#[inline(always)]
fn round_even<const R: usize>(s: &mut [u64; 8]) {
    let rc = RC[R];
    s[0] ^= rc;
    s[4] ^= rc.rotate_right(32);

    arx_mix_v3(s);

    let (y0, y1, y2, y3) = (s[5], s[6], s[7], s[4]);
    let [r0, r1, r2, r3, c0, c1, c2, c3] = s;
    ct_mr2_v31_quad(r0, r1, r2, r3, y0, y1, y2, y3);

    let (u0, u1, u2, u3) = (*r3, *r0, *r1, *r2);
    ct_mr2_v31_quad(c0, c1, c2, c3, u0, u1, u2, u3);

    chi_5_3_inplace(s);
    pi_v3(s);
}

#[inline(always)]
fn round_odd<const R: usize>(s: &mut [u64; 8]) {
    let rc = RC[R];
    s[0] ^= rc;
    s[4] ^= rc.rotate_right(32);

    arx_mix_v3(s);
    chi_5_3_inplace(s);
    pi_v3(s);
}

/// 14-round data duplex permutation ($P_{14}$).
#[inline(always)]
pub fn permute_14(state: &mut MorphicState) {
    let s = &mut state.lanes;
    round_even::<0>(s);
    round_odd::<1>(s);
    round_even::<2>(s);
    round_odd::<3>(s);
    round_even::<4>(s);
    round_odd::<5>(s);
    round_even::<6>(s);
    round_odd::<7>(s);
    round_even::<8>(s);
    round_odd::<9>(s);
    round_even::<10>(s);
    round_odd::<11>(s);
    round_even::<12>(s);
    round_odd::<13>(s);
}

/// 16-round full permutation ($P_{16}$) for Init, Finalization, $H_K$, KDF, and HighSec.
#[inline(always)]
pub fn permute_16(state: &mut MorphicState) {
    permute_14(state);
    let s = &mut state.lanes;
    round_even::<14>(s);
    round_odd::<15>(s);
}

#[inline(always)]
pub fn permute_data(state: &mut MorphicState, use_p16: bool) {
    if use_p16 {
        permute_16(state);
    } else {
        permute_14(state);
    }
}

/// Executes a single round `r` (`0..16`), used for reduced-round avalanche verification.
pub fn permute_single_round(state: &mut MorphicState, r: usize) {
    let s = &mut state.lanes;
    match r {
        0 => round_even::<0>(s),
        1 => round_odd::<1>(s),
        2 => round_even::<2>(s),
        3 => round_odd::<3>(s),
        4 => round_even::<4>(s),
        5 => round_odd::<5>(s),
        6 => round_even::<6>(s),
        7 => round_odd::<7>(s),
        8 => round_even::<8>(s),
        9 => round_odd::<9>(s),
        10 => round_even::<10>(s),
        11 => round_odd::<11>(s),
        12 => round_even::<12>(s),
        13 => round_odd::<13>(s),
        14 => round_even::<14>(s),
        15 => round_odd::<15>(s),
        _ => {}
    }
}

pub fn permute_14_4way(soa: &mut SoaState4) {
    for c in 0..4 {
        let mut st = MorphicState::new([
            soa.lanes[0][c],
            soa.lanes[1][c],
            soa.lanes[2][c],
            soa.lanes[3][c],
            soa.lanes[4][c],
            soa.lanes[5][c],
            soa.lanes[6][c],
            soa.lanes[7][c],
        ]);
        permute_14(&mut st);
        for l in 0..8 {
            soa.lanes[l][c] = st.lanes[l];
        }
    }
}

pub fn permute_16_4way(soa: &mut SoaState4) {
    for c in 0..4 {
        let mut st = MorphicState::new([
            soa.lanes[0][c],
            soa.lanes[1][c],
            soa.lanes[2][c],
            soa.lanes[3][c],
            soa.lanes[4][c],
            soa.lanes[5][c],
            soa.lanes[6][c],
            soa.lanes[7][c],
        ]);
        permute_16(&mut st);
        for l in 0..8 {
            soa.lanes[l][c] = st.lanes[l];
        }
    }
}
