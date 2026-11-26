//! C11 Foreign Function Interface (FFI) for NIBBLX-512 v3.1.1.
//!
//! Matches `include/nibblx512.h`. Safe for C/C++, Python `ctypes`, JNI, and OS kernels.

use core::slice;

use crate::ct_ops;
use crate::kdf;
use crate::{decrypt_in_place, encrypt_in_place, MorphicError, MorphicTag, Profile, KEY_LEN, NONCE_LEN};

pub const MORPHIC_OK: i32 = 0;
pub const MORPHIC_ERR_INVALID_PARAM: i32 = -1;
pub const MORPHIC_ERR_AUTH_FAILED: i32 = -2;

#[inline(always)]
fn map_err(e: MorphicError) -> i32 {
    match e {
        MorphicError::AuthenticationFailed => MORPHIC_ERR_AUTH_FAILED,
        MorphicError::InvalidParameter | MorphicError::StreamAlreadyFinalized => {
            MORPHIC_ERR_INVALID_PARAM
        }
    }
}

/// In-place AEAD encryption via C ABI.
///
/// # Safety
/// Caller must pass valid non-null, non-overlapping pointers for `key` (32B), `nonce` (24B),
/// `tag_out` (`tag_cap` >= 16/32), and valid non-overlapping pointers whenever `ad_len > 0`
/// or `buf_len > 0`.
#[no_mangle]
pub unsafe extern "C" fn morphic_v3_encrypt_in_place(
    key: *const u8,
    nonce: *const u8,
    ad: *const u8,
    ad_len: usize,
    profile_id: u16,
    buf: *mut u8,
    buf_len: usize,
    tag_out: *mut u8,
    tag_cap: usize,
) -> i32 {
    if key.is_null() || nonce.is_null() || tag_out.is_null() {
        return MORPHIC_ERR_INVALID_PARAM;
    }
    if (ad_len > 0 && ad.is_null()) || (buf_len > 0 && buf.is_null()) {
        return MORPHIC_ERR_INVALID_PARAM;
    }
    let profile = match Profile::from_ver_id(profile_id) {
        Some(p) => p,
        None => return MORPHIC_ERR_INVALID_PARAM,
    };
    if tag_cap < profile.tag_len() {
        return MORPHIC_ERR_INVALID_PARAM;
    }

    // SAFETY: Non-null checks, length invariants, and caller non-overlapping precondition verified.
    let (k_ref, n_ref, ad_slice, data_slice, tag_slice) = unsafe {
        let k = &*(key as *const [u8; KEY_LEN]);
        let n = &*(nonce as *const [u8; NONCE_LEN]);
        let a = if ad_len == 0 { &[] } else { slice::from_raw_parts(ad, ad_len) };
        let d = if buf_len == 0 { &mut [] } else { slice::from_raw_parts_mut(buf, buf_len) };
        let t = slice::from_raw_parts_mut(tag_out, profile.tag_len());
        (k, n, a, d, t)
    };

    match encrypt_in_place(k_ref, n_ref, ad_slice, profile, data_slice) {
        Ok(tag) => {
            tag_slice.copy_from_slice(tag.as_bytes());
            MORPHIC_OK
        }
        Err(e) => map_err(e),
    }
}

/// In-place AEAD decryption with strict verify-before-release & volatile wipe on auth failure.
///
/// # Safety
/// Caller must pass valid non-null, non-overlapping pointers for `key` (32B), `nonce` (24B),
/// `expected_tag` (`tag_len`), and valid non-overlapping pointers whenever `ad_len > 0`
/// or `buf_len > 0`.
#[no_mangle]
pub unsafe extern "C" fn morphic_v3_decrypt_in_place(
    key: *const u8,
    nonce: *const u8,
    ad: *const u8,
    ad_len: usize,
    profile_id: u16,
    buf: *mut u8,
    buf_len: usize,
    expected_tag: *const u8,
    tag_len: usize,
) -> i32 {
    if key.is_null() || nonce.is_null() || expected_tag.is_null() {
        return MORPHIC_ERR_INVALID_PARAM;
    }
    if (ad_len > 0 && ad.is_null()) || (buf_len > 0 && buf.is_null()) {
        return MORPHIC_ERR_INVALID_PARAM;
    }
    let profile = match Profile::from_ver_id(profile_id) {
        Some(p) => p,
        None => return MORPHIC_ERR_INVALID_PARAM,
    };
    if tag_len != profile.tag_len() {
        return MORPHIC_ERR_INVALID_PARAM;
    }

    // SAFETY: Non-null checks, length invariants, and caller non-overlapping precondition verified.
    let (k_ref, n_ref, ad_slice, data_slice, tag) = unsafe {
        let k = &*(key as *const [u8; KEY_LEN]);
        let n = &*(nonce as *const [u8; NONCE_LEN]);
        let a = if ad_len == 0 { &[] } else { slice::from_raw_parts(ad, ad_len) };
        let d = if buf_len == 0 { &mut [] } else { slice::from_raw_parts_mut(buf, buf_len) };
        let t = match profile {
            Profile::Compact128 => {
                let mut arr = [0u8; 16];
                arr.copy_from_slice(slice::from_raw_parts(expected_tag, 16));
                MorphicTag::Compact128(arr)
            }
            Profile::EnterpriseCmt256 | Profile::HighSecCmt256 => {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(slice::from_raw_parts(expected_tag, 32));
                MorphicTag::Commit256(arr)
            }
        };
        (k, n, a, d, t)
    };

    match decrypt_in_place(k_ref, n_ref, ad_slice, profile, data_slice, &tag) {
        Ok(()) => MORPHIC_OK,
        Err(e) => map_err(e),
    }
}

/// Derives a 256-bit key via pure NIBBLX-512 $P_{16}$ iterated KDF.
///
/// # Safety
/// Caller must pass valid non-null, non-overlapping pointers for `salt24` (24B), `out_key32` (32B),
/// and `password` whenever `password_len > 0`.
#[no_mangle]
pub unsafe extern "C" fn morphic_v3_derive_key(
    password: *const u8,
    password_len: usize,
    salt24: *const u8,
    iterations: u32,
    out_key32: *mut u8,
) -> i32 {
    if salt24.is_null() || out_key32.is_null() || (password_len > 0 && password.is_null()) {
        return MORPHIC_ERR_INVALID_PARAM;
    }

    // SAFETY: Non-null checks and caller non-overlapping precondition verified above.
    let (pw, salt, out) = unsafe {
        let p = if password_len == 0 { &[] } else { slice::from_raw_parts(password, password_len) };
        let s = &*(salt24 as *const [u8; NONCE_LEN]);
        let o = slice::from_raw_parts_mut(out_key32, KEY_LEN);
        (p, s, o)
    };

    match kdf::derive_key_iterated(pw, salt, iterations) {
        Ok(mut k) => {
            out.copy_from_slice(&k);
            ct_ops::secure_zeroize_bytes(&mut k);
            MORPHIC_OK
        }
        Err(e) => map_err(e),
    }
}

/// Compiler-barrier-protected volatile memory zeroization via C ABI.
///
/// # Safety
/// Caller must pass a valid writable buffer of `len` bytes when `len > 0`.
#[no_mangle]
pub unsafe extern "C" fn morphic_v3_secure_zeroize(ptr: *mut u8, len: usize) {
    if !ptr.is_null() && len > 0 {
        // SAFETY: Non-null and `len > 0` checked above.
        let s = unsafe { slice::from_raw_parts_mut(ptr, len) };
        ct_ops::secure_zeroize_bytes(s);
    }
}
