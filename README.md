# NIBBLX-512 (`nibblx512`) — v3.1.1 ("Seal-Pack")

> **Status & Security Notice:** NIBBLX-512 is an independent, custom cryptographic design and research implementation authored by **Akash Patra**. Although it has been verified against internal algebraic bounds, constant-time timing audits (`dudect`), assembly branch checks, and 32 GB of PractRand statistical testing, it has **not** yet undergone independent third-party academic cryptanalysis. Treat it as experimental research software until formal external peer review is completed.

**NIBBLX-512** is a zero-dependency, `#![no_std]` 512-bit bijective ARX-Chi sponge Authenticated Encryption with Associated Data (AEAD) cipher written in pure Rust. Every primitive in the crate — AEAD encryption/authentication, key commitment ($H_K$), memory-hard & iterated KDF, C11 FFI, and O(1)-RAM multi-GB streaming CLI — is built exclusively from the NIBBLX-512 permutation with zero external crate dependencies, zero lookup tables, and zero secret-dependent branches.

## Key Features
- **Zero External Dependencies (`#![no_std]` core):** No external crates, no S-Boxes, no RAM lookup tables.
- **Bijective 512-bit Permutation ($P_{14}$ / $P_{16}$):**
  - Asymmetric `SplitMix64` round constants ($\iota_r$, 16 rounds).
  - Full Double-Rotate ARX ($\mathcal{A}_{v3}$).
  - **PF1 non-linear pre-fold** (`W = Y.wrapping_add(Y.rotate_right(17))`) + 5-step interleaved `SHL`-folded `CT-MR2_v3.1` with `core::hint::black_box`-hardened masks.
  - 3-scratch in-place bijective Split-Chi ($\chi_{5,3}$) + non-affine pair-breaking lane permutation $\pi_{v3} = (5, 6, 7, 0, 2, 1, 4, 3)$.
- **Fused Register-Resident 4-Way SIMD:** Automatic `#![no_std]` runtime CPUID + OSXSAVE + XGETBV (`XCR0[2:1]`) AVX2 dispatch on `x86_64`, native `aarch64` NEON kernel, and zero-spill scalar fallback.
- **Key-Committing AEAD (CMT-4) & Streaming Hardening:** One-way Davies-Meyer Key-Commitment Context ($H_K$), 3-case collision-free padding, borrowed-key `&'k [u8; 32]` streaming structs, permanent `StreamDecryptor` poisoning on first auth failure, and volatile RAII zeroization.
- **Pure NIBBLX-512 Hybrid Memory-Hard KDF:** Argon2id-style hybrid addressing (Pass 0 strictly data-independent for cache-timing resistance; Pass $\ge 1$ data-dependent) with division-free 128-bit Lemire fast-range indexing.
- **O(1)-RAM `.mrf3` Streaming CLI & C11 FFI:** Safe password input via `NIBBLX_PASS` / `--pass-stdin`, atomic `.nibblx.tmp` rename after Master Tag verification, and C header `include/nibblx512.h`.

## Security Bounds & Threat Model

| Property | Bound / Specification |
| :--- | :--- |
| Internal State / Rate / Capacity | 512-bit state (`r = 256`, `c = 256`) |
| Generic Sponge AEAD Security (Confidentiality & Integrity) | **128-bit** ($2^{c/2}$ for $c = 256$) |
| Key-Recovery Resistance | **256-bit** classical (**128-bit** quantum Grover) |
| Key Commitment (`EnterpriseCmt256` / `HighSecCmt256`) | **128-bit** CMT-4 multi-key commitment (32-byte tag) |
| Key Commitment (`Compact128`) | **64-bit** birthday commitment (16-byte tag) |
| Nonce Size & Requirement | **192-bit** (24 bytes) — **Nonce-respecting AEAD** |
| Max Stream Length per Nonce | $2^{48}$ chunks × 64 KB = **16 Exabytes** |

- **Nonce-Respecting Requirement:** NIBBLX-512 is **not** nonce-misuse resistant. Reusing `(Key, Nonce)` with identical `AD` leaks the XOR of the first 32 plaintext bytes of each chunk. With 192-bit random nonces, birthday collision probability stays below $2^{-32}$ up to $2^{80}$ messages per key.
- **Defense-in-Depth (`CT-MR2_v3.1`):** Even under worst-case `CT-MR2` bypass, $P_{14}$ and $P_{16}$ retain conservative analytical active-slice bounds ($\ge 51$ and $\ge 59$ active $\chi_{5,3}$ slices, respectively). Full formal specification is in [`SPEC.md`](SPEC.md).

## Measured Performance (`EnterpriseCmt256`, 64 MB payload)
Measured on x86-64 (WSL2, `rustc --release`, `opt-level = 3`, `lto = "fat"`):

| Backend | Throughput | Cycles / Byte | Speedup |
| :--- | ---: | ---: | ---: |
| 1-Core Scalar | 154.0 MB/s | 14.84 cpb | 1.00× |
| 1-Core 4-Way AVX2 SIMD | 405.1 MB/s | 5.64 cpb | 2.63× |
| 8-Core Parallel AVX2 SIMD | 1,286.5 MB/s | — | 8.35× |

## Verified Audit Results (v3.1.1)

| Audit Check | Scope | Result |
| :--- | :--- | :--- |
| `cargo clippy --all-targets --release -- -D warnings` | Full crate, CLI, examples, tests | **0 warnings, 0 errors** |
| `cargo test --release` | v3.1 KATs, $P_1 \dots P_6$ avalanche, AVX2==Scalar, poisoning, KDF, FFI | **8 / 8 PASS** |
| `objdump` Branch Audit | Compiled `permute_14_inreg` (x86-64 AVX2) | **0 branches** |
| `dudect` Suite 1 (Scalar 64 B) | 1,000,000 samples (Fixed vs. Random) | `Max \|t\| = 1.6362` (**PASS** `< 4.5`) |
| `dudect` Suite 2 (4-Way SIMD 256 KB) | 4,000 samples (~1 GB timed) | `Max \|t\| = 2.3362` (**PASS** `< 4.5`) |
| `dudect` Suite 3 (Tag Verify `ct_eq_slice`) | 500,000 samples (byte-0 vs. byte-31 mismatch) | `Max \|t\| = 0.7665` (**PASS** `< 4.5`) |
| PractRand 0.95 (`RNG_test stdin64`) | **32 GB** ($2^{35}$ bytes) raw ciphertext stream | **0 anomalies in 325 tests** |

## Quick Start

### Rust Library (`#![no_std]`)
```rust
use nibblx512::{decrypt_in_place, encrypt_in_place, Profile};

let key = [0x42u8; 32];
let nonce = [0x24u8; 24];
let ad = b"header-metadata";
let mut buf = *b"Hello from NIBBLX-512 v3.1.1!";

let tag = encrypt_in_place(&key, &nonce, ad, Profile::EnterpriseCmt256, &mut buf).unwrap();
decrypt_in_place(&key, &nonce, ad, Profile::EnterpriseCmt256, &mut buf, &tag).unwrap();
```

### Streaming File CLI (`nibblx_cli`)
```bash
# Recommended: pass password via environment variable or --pass-stdin (never in argv)
NIBBLX_PASS="MyStrongPass" cargo run --release --bin nibblx_cli -- encrypt input.bin output.mrf3
NIBBLX_PASS="MyStrongPass" cargo run --release --bin nibblx_cli -- decrypt output.mrf3 restored.bin
```

### Run the Full Verification Suite
```bash
cargo build --release --all-targets
./verify_all.sh
```

## License
Dual-licensed under either of:
- MIT License ([`LICENSE-MIT`](LICENSE-MIT))
- Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE))

at your option.
