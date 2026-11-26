#ifndef NIBBLX512_H
#define NIBBLX512_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MORPHIC_KEY_LEN            32U
#define MORPHIC_NONCE_LEN          24U
#define MORPHIC_TAG_LEN_COMPACT    16U
#define MORPHIC_TAG_LEN_ENTERPRISE 32U

/* Protocol Version + Profile IDs (v3.1.x) */
#define MORPHIC_PROFILE_COMPACT_128    0x0311U
#define MORPHIC_PROFILE_ENTERPRISE_256 0x0312U
#define MORPHIC_PROFILE_HIGHSEC_256    0x0313U

/* Return Codes */
#define MORPHIC_OK                 0
#define MORPHIC_ERR_INVALID_PARAM -1
#define MORPHIC_ERR_AUTH_FAILED   -2

/**
 * Encrypts `buf[0..buf_len]` in place and writes 16B (Compact) or 32B (Enterprise/HighSec) tag.
 * Automatically uses 4-Way AVX2 on x86-64 or NEON on ARM64 via #![no_std] runtime/compile dispatch.
 * Precondition: `key`, `nonce`, `ad`, `buf`, and `tag_out` must not overlap in memory.
 */
int32_t morphic_v3_encrypt_in_place(
    const uint8_t *key,
    const uint8_t *nonce,
    const uint8_t *ad,
    size_t ad_len,
    uint16_t profile_id,
    uint8_t *buf,
    size_t buf_len,
    uint8_t *tag_out,
    size_t tag_cap
);

/**
 * Decrypts `buf[0..buf_len]` in place with strict verify-before-release.
 * On authentication failure, `buf` is immediately wiped via volatile writes and MORPHIC_ERR_AUTH_FAILED is returned.
 * Precondition: `key`, `nonce`, `ad`, `buf`, and `expected_tag` must not overlap in memory.
 */
int32_t morphic_v3_decrypt_in_place(
    const uint8_t *key,
    const uint8_t *nonce,
    const uint8_t *ad,
    size_t ad_len,
    uint16_t profile_id,
    uint8_t *buf,
    size_t buf_len,
    const uint8_t *expected_tag,
    size_t tag_len
);

/**
 * Pure NIBBLX-512 P16 iterated KDF (zero external hash/cipher dependencies).
 * Precondition: `password`, `salt24`, and `out_key32` must not overlap in memory.
 */
int32_t morphic_v3_derive_key(
    const uint8_t *password,
    size_t password_len,
    const uint8_t *salt24,
    uint32_t iterations,
    uint8_t *out_key32
);

/**
 * Compiler-barrier-protected volatile memory wipe.
 */
void morphic_v3_secure_zeroize(uint8_t *ptr, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* NIBBLX512_H */
