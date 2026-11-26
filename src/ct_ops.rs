//! Constant-time tag verification (`black_box` hardened) and volatile memory zeroization.

use core::hint::black_box;
use core::ptr::write_volatile;
use core::sync::atomic::{compiler_fence, Ordering};

/// Constant-time equality comparison for 16-byte or 32-byte tags (zero early exits).
#[inline(never)]
pub fn ct_eq_slice(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    let mut i = 0usize;
    while i < a.len() {
        diff |= black_box(a[i]) ^ black_box(b[i]);
        i += 1;
    }
    compiler_fence(Ordering::SeqCst);
    black_box(diff) == 0
}

/// Volatile byte-slice zeroization resistant to dead-store elimination.
#[inline(never)]
pub fn secure_zeroize_bytes(buf: &mut [u8]) {
    for slot in buf.iter_mut() {
        // SAFETY: `slot` is a valid, aligned, mutable reference to a `u8`.
        unsafe {
            write_volatile(slot as *mut u8, 0u8);
        }
    }
    compiler_fence(Ordering::SeqCst);
}

/// Volatile `u64`-slice zeroization resistant to dead-store elimination.
#[inline(never)]
pub fn secure_zeroize_u64(buf: &mut [u64]) {
    for slot in buf.iter_mut() {
        // SAFETY: `slot` is a valid, aligned, mutable reference to a `u64`.
        unsafe {
            write_volatile(slot as *mut u64, 0u64);
        }
    }
    compiler_fence(Ordering::SeqCst);
}
