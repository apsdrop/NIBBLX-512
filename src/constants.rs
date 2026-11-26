//! Cryptographic constants for Morphic Sponge ARX-512 v3.1.0 ("Seal-Pack").

/// 16 asymmetric 64-bit round constants: `SplitMix64((r + 1) * 0x9E3779B97F4A7C15 mod 2^64)`.
pub const RC: [u64; 16] = [
    0xE220_A839_7B1D_CDAF,
    0x6E78_9E6A_A1B9_65F4,
    0x06C4_5D18_8009_454F,
    0xF88B_B8A8_724C_81EC,
    0x1B39_896A_51A8_749B,
    0x53CB_9F0C_747E_A2EA,
    0x2C82_9ABE_1F45_32E1,
    0xC584_133A_C916_AB3C,
    0x3EE5_7890_41C9_8AC3,
    0xF3B8_488C_368C_B0A6,
    0x657E_ECDD_3CB1_3D09,
    0xC2D3_26E0_055B_DEF6,
    0x8621_A03F_E0BB_DB7B,
    0x8E1F_7555_983A_A92F,
    0xB54E_0F16_00CC_4D19,
    0x84BB_3F97_971D_80AB,
];

/// Full Double-Rotate ARX left-rotation constants ($\sigma_k$, odd & coprime to 64).
pub const ARX_SIGMA: [u32; 4] = [7, 13, 29, 43];
/// Full Double-Rotate ARX right-rotation constants ($\rho_k$, odd & coprime to 64).
pub const ARX_RHO: [u32; 4] = [19, 27, 41, 53];

/// PF1 non-linear pre-fold rotation constant (`W = Y.wrapping_add(Y.rotate_right(17))`).
pub const PREFOLD_ROT: u32 = 17;

/// Constant-Time Morphic Rotation (`CT-MR2_v3.1`) fixed immediate strides ({11, 28, 42, 59}).
pub const MORPH_BASE_ROT: u32 = 11;
pub const MORPH_STRIDE_0: u32 = 17;
pub const MORPH_STRIDE_1: u32 = 31;

/// 16-bit Protocol Version + Profile Identifiers (v3.1.0 domain-separated from v3.0.0).
pub const VER_COMPACT_128: u64 = 0x0311;
pub const VER_ENTERPRISE_256: u64 = 0x0312;
pub const VER_HIGHSEC_256: u64 = 0x0313;
pub const VER_KDF: u64 = 0x031F;

/// 8-bit Domain Separation Tags.
pub const DOM_CMT: u64 = 0x01;
pub const DOM_AD: u64 = 0x10;
pub const DOM_CHUNK: u64 = 0x20;
pub const DOM_ROOT: u64 = 0x30;
pub const DOM_KDF: u64 = 0x40;

/// Dedicated 64-bit RootFinalizer EOF domain constant (strictly isolated in lane S7).
pub const DOM_ROOT_EOF: u64 = 0xFF00_0000_0000_0001;
/// Dedicated 64-bit KDF Finalization domain constant (strictly isolated in lane S7).
pub const DOM_KDF_FINAL: u64 = 0xFF00_0000_0000_0002;

/// Chunk flag indicating the final chunk of a stream (prevents truncation).
pub const FLAG_IS_FINAL: u64 = 0x01;

/// Mutually exclusive 2-bit Capacity padding domain tags (Fix 1).
pub const CASE_EMPTY: u64 = 0x01;
pub const CASE_FULL: u64 = 0x02;
pub const CASE_PARTIAL: u64 = 0x03;
