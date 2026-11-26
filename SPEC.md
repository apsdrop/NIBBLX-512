# NIBBLX-512 v3.1.1 ("Seal-Pack") — Formal Specification

## 1. Core Parameters
- **Internal State:** 512 bits = 8 × 64-bit Little-Endian registers `(S0..S7)`
- **Rate (`r`):** 256 bits (`S0..S3`, 32 bytes)
- **Capacity (`c`):** 256 bits (`S4..S7`, 32 bytes)
- **Key:** 256 bits (32 bytes) | **Nonce:** 192 bits (24 bytes) | **Chunk Size:** 64 KB (`65,536` bytes, max $2^{48}$ chunks = 16 Exabytes)
- **Profiles (`VER_PROFILE_16`):**
  - `0x0311` (`Compact128`): 16-byte Tag (64-bit birthday key commitment), $P_{14}$ data rounds, $P_{16}$ Init/Final/$H_K$
  - `0x0312` (`EnterpriseCmt256`): 32-byte CMT-4 Key-Committing Tag (128-bit multi-key bound), $P_{14}$ data rounds, $P_{16}$ Init/Final/$H_K$
  - `0x0313` (`HighSecCmt256`): 32-byte CMT-4 Key-Committing Tag, $P_{16}$ everywhere
  - `0x031F` (`VER_KDF`): Pure NIBBLX-512 $P_{16}$ Hybrid Memory-Hard & Iterated KDF

## 2. Bijective Permutation $P_{14}$ / $P_{16}$
Each round $r \in \{0..N_r-1\}$ executes 5 bijective sub-layers:
1. **Asymmetric Round Constant ($\iota_r$):** $S_0 \leftarrow S_0 \oplus RC_r$, $S_4 \leftarrow S_4 \oplus (RC_r \ggg 32)$ where $RC_r = \text{SplitMix64}((r+1) \cdot \text{0x9E3779B97F4A7C15} \bmod 2^{64})$ for $r \in \{0..15\}$.
2. **Double-Rotate ARX ($\mathcal{A}_{v3}$):** For pairs $(S_k, S_{k+4})$, $k \in \{0..3\}$:
   $t = S_k \boxplus S_{k+4}$, $S_k \leftarrow t \lll \sigma_k$, $S_{k+4} \leftarrow (S_{k+4} \ggg \rho_k) \oplus t$,
   with $\sigma = (7, 13, 29, 43)$ and $\rho = (19, 27, 41, 53)$.
3. **PF1 Non-Linear Pre-Fold + Folded `CT-MR2_v3.1` (Even Rounds Only):**
   $W_0 = Y \boxplus (Y \ggg 17)$; $W_5 = \text{SHL-Fold}_{32,16,8,4,2}(W_0)$;
   $M_0 = \text{SAR}(W_5, 63)$; $M_1 = \text{SAR}(W_5 \ll 1, 63)$;
   $X \leftarrow X \ggg 11$; $X \leftarrow X \oplus ((X \oplus (X \ggg 17)) \land M_0)$; $X \leftarrow X \oplus ((X \oplus (X \ggg 31)) \land M_1)$.
   Applied sequentially to Half A `(0<-5, 1<-6, 2<-7, 3<-4)` then Half B `(4<-3, 5<-0, 6<-1, 7<-2)`.
   Scalar mask derivations are wrapped in `core::hint::black_box` to prevent branchification.
4. **In-Place Split-Chi ($\chi_{5,3}$):** 5-lane odd cycle `{S0..S4}` and 3-lane odd cycle `{S5..S7}`.
5. **Non-Affine Pair-Breaking Permutation ($\pi_{v3}$):** `(5, 6, 7, 0, 2, 1, 4, 3)`.

## 3. AEAD Key Commitment ($H_K$), 3-Case Padding & Root Finalization
- **$H_K$ (One-Way Davies-Meyer):** $X_0 = ((\text{VER\_PROFILE}_{16} \ll 48) \mid (\text{0x01} \ll 40), N_0, N_1, N_2, K_0..K_3) \xrightarrow{P_{16}} Y$; $H_K = Y_{0..3} \oplus K_{0..3}$ ($Y_{4..7}$ discarded).
- **3-Case Chunk Padding:**
  - `L == 0` (`CASE_EMPTY = 1`): 0 blocks, $S_7 \oplus 1 \to$ Key FF $\to P_{16}$.
  - `L = 32k` (`CASE_FULL = 2`): $k$ full blocks (no extra data permutation after block $k-1$), $S_6 \oplus L, S_7 \oplus 2 \to$ Key FF $\to P_{16}$.
  - `L = 32k + m` (`CASE_PARTIAL = 3`): $k$ full blocks + surgical partial overwrite of bytes `0..m-1` + `0x80` at byte `m` (trailing bytes untouched, no extra data permutation), $S_6 \oplus L, S_7 \oplus 3 \to$ Key FF $\to P_{16}$.
- **RootFinalizer EOF (Disjoint Lanes):** $S_5 \oplus \text{total\_chunks}$, $S_6 \oplus \text{total\_bytes}$, $S_7 \oplus \text{0xFF00000000000001}$, Key FF $\to P_{16} \to$ Master Tag.

## 4. Security Claims & Threat Model
- **Generic Sponge AEAD Security (Confidentiality & Integrity):** **128-bit** ($2^{c/2}$ for $c = 256$).
- **Key-Recovery Resistance:** **256-bit** classical (**128-bit** quantum Grover bound).
- **Key Commitment (CMT-4):** **128-bit** multi-key commitment for `EnterpriseCmt256`/`HighSecCmt256` (32-byte tags); **64-bit** birthday commitment for `Compact128` (16-byte tags).
- **Nonce Requirement (Nonce-Respecting AEAD):** NIBBLX-512 is **not** nonce-misuse resistant. Reusing `(Key, Nonce)` with the same `AD` leaks the XOR of the first 32 plaintext bytes of each chunk. With 192-bit random nonces, birthday collision probability remains below $2^{-32}$ up to $2^{80}$ messages per key.
- **Role of `CT-MR2_v3.1`:** Treated as defense-in-depth (algebraic degree & trail splitting) atop the conservative 14/16-round ARX-Chi core; even under worst-case `CT-MR2` bypass, $P_{14}$ and $P_{16}$ retain conservative analytical bounds ($\ge 51$ and $\ge 59$ active $\chi_{5,3}$ slices, respectively).
- **Hybrid Memory-Hard KDF (`kdf.rs`):** Uses Argon2id-style hybrid addressing — Pass 0 uses strictly public data-independent indexing (cache-timing resistant) with division-free Lemire 128-bit fast-range reduction; Pass $\ge 1$ uses division-free data-dependent indexing.
- **Stream Poisoning:** `StreamDecryptor` permanently locks out further calls upon the first `AuthenticationFailed` event.

## 5. `.mrf3` Framed Streaming File Format (64-Byte Authenticated Header)
- `0..8`: Magic `"MRF3ARX\0"` | `8..10`: `VER_PROFILE_16` (LE) | `10..12`: KDF MiB (LE) | `12..16`: KDF Passes (LE)
- `16..40`: 192-bit Random Salt | `40..64`: 192-bit Random Nonce (entire 64-byte header bound as `AD`)
- Framed Chunks: `[u32 len_le || u8 is_final || ciphertext[len] || chunk_tag[16|32]]`
- Trailer: `[master_tag[16|32]]` (CLI writes verified chunks to a `.nibblx.tmp` file and atomically renames only after Master Tag verification passes)

## 6. Official v3.1.x Known-Answer Test (KAT) Vectors
- Key: `000102...1F` (32B), Nonce: `A0A1A2...B7` (24B), AD: `"MORPHIC-SPONGE-ARX-512-V31"`, PT: `303132...5F` (48B)
- **EnterpriseCmt256 (`0x0312`):**
  - CT[0..16]: `77CEF81522F6B226A89A51D2176F3647`
  - Master Tag (32B): `7ECC05CF488892E5887DB909A7FE4ED9365CC720513972E0BDB2EC38B283A0A5`
- **Compact128 (`0x0311`):**
  - CT[0..16]: `F5A04788F4560BCFF2737638123E7C90`
  - Master Tag (16B): `6403866E4FE58DD55D941BBDEAB47DF4`
