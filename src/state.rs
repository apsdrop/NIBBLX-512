//! 64-byte aligned internal sponge states (Scalar & 4-Way SoA SIMD) with RAII zeroization.

use crate::ct_ops;

/// 64-byte cache-line-aligned 512-bit scalar sponge state.
#[repr(C, align(64))]
#[derive(Clone)]
pub struct MorphicState {
    pub lanes: [u64; 8],
}

impl MorphicState {
    #[inline(always)]
    pub const fn new(lanes: [u64; 8]) -> Self {
        Self { lanes }
    }

    #[inline(always)]
    pub fn load_le64(bytes: &[u8]) -> u64 {
        let mut arr = [0u8; 8];
        arr.copy_from_slice(&bytes[..8]);
        u64::from_le_bytes(arr)
    }

    #[inline(always)]
    pub fn store_le64(out: &mut [u8], val: u64) {
        out[..8].copy_from_slice(&val.to_le_bytes());
    }

    #[inline(always)]
    pub fn load_key(key: &[u8; 32]) -> [u64; 4] {
        [
            Self::load_le64(&key[0..8]),
            Self::load_le64(&key[8..16]),
            Self::load_le64(&key[16..24]),
            Self::load_le64(&key[24..32]),
        ]
    }

    #[inline(always)]
    pub fn load_nonce(nonce: &[u8; 24]) -> [u64; 3] {
        [
            Self::load_le64(&nonce[0..8]),
            Self::load_le64(&nonce[8..16]),
            Self::load_le64(&nonce[16..24]),
        ]
    }

    /// Feed-Forward Key Hardening into Capacity lanes (S4..S7).
    #[inline(always)]
    pub fn xor_capacity_key(&mut self, key: &[u64; 4]) {
        self.lanes[4] ^= key[0];
        self.lanes[5] ^= key[1];
        self.lanes[6] ^= key[2];
        self.lanes[7] ^= key[3];
    }

    /// Fix 2: Surgical word-level partial-block duplex transition (1 <= `tail.len()` <= 31).
    /// Overwrites ONLY the active `tail.len()` bytes with ciphertext, XORs `0x80` at exact
    /// byte index `tail.len()`, and leaves all trailing Rate bytes untouched.
    #[inline(always)]
    pub fn duplex_partial_block(&mut self, tail: &mut [u8], is_decrypt: bool) {
        let rem = tail.len();
        let full_lanes = rem / 8;
        let tail_bytes = rem % 8;

        for i in 0..full_lanes {
            let w = Self::load_le64(&tail[8 * i..]);
            if !is_decrypt {
                let ct = w ^ self.lanes[i];
                Self::store_le64(&mut tail[8 * i..], ct);
                self.lanes[i] = ct;
            } else {
                let pt = w ^ self.lanes[i];
                Self::store_le64(&mut tail[8 * i..], pt);
                self.lanes[i] = w;
            }
        }

        if tail_bytes > 0 {
            let mask = (1u64 << (8 * tail_bytes)) - 1u64;
            let mut in_w = 0u64;
            for j in 0..tail_bytes {
                in_w |= (tail[full_lanes * 8 + j] as u64) << (8 * j);
            }
            let out_w = (in_w ^ self.lanes[full_lanes]) & mask;
            for j in 0..tail_bytes {
                tail[full_lanes * 8 + j] = (out_w >> (8 * j)) as u8;
            }
            let ct_w = if is_decrypt { in_w } else { out_w };
            self.lanes[full_lanes] = (self.lanes[full_lanes] & !mask) | (ct_w & mask);
        }

        self.lanes[full_lanes] ^= 0x80u64 << (8 * tail_bytes);
    }
}

impl Drop for MorphicState {
    fn drop(&mut self) {
        ct_ops::secure_zeroize_u64(&mut self.lanes);
    }
}

/// 64-byte aligned 4-Way Structure-of-Arrays (SoA) state for AVX2 / NEON lockstep execution.
/// `lanes[l][c]` holds register `S_l` of parallel chunk `c` (`c in 0..4`).
#[repr(C, align(64))]
pub struct SoaState4 {
    pub lanes: [[u64; 4]; 8],
}

impl SoaState4 {
    #[inline(always)]
    pub const fn zeroed() -> Self {
        Self { lanes: [[0u64; 4]; 8] }
    }

    #[inline(always)]
    pub fn xor_capacity_key_all(&mut self, key: &[u64; 4]) {
        for i in 0..4 {
            let k = key[i];
            let lane = &mut self.lanes[4 + i];
            lane[0] ^= k;
            lane[1] ^= k;
            lane[2] ^= k;
            lane[3] ^= k;
        }
    }
}

impl Drop for SoaState4 {
    fn drop(&mut self) {
        for lane in self.lanes.iter_mut() {
            ct_ops::secure_zeroize_u64(lane);
        }
    }
}
