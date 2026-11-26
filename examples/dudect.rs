//! Zero-dependency 3-Suite Welch's t-test constant-time audit (true Reparaz et al. `dudect`)
//! for NIBBLX-512 v3.1.1:
//!   Suite 1: Scalar `encrypt_in_place` (64 B) — Fixed all-zero vs. Random
//!   Suite 2: 4-Way SIMD `encrypt_in_place` (256 KB = 4 x 64 KB chunks) — Fixed vs. Random
//!   Suite 3: `decrypt_in_place` tag verification (`ct_eq_slice`) — 0-byte vs. 31-byte prefix match
//!
//! Run:
//!   cargo run --release --example dudect -- 1000000

use std::env;
use std::hint::black_box;
use std::process;

use nibblx512::{
    decrypt_in_place, encrypt_in_place, MorphicTag, Profile, CHUNK_SIZE, KEY_LEN, NONCE_LEN,
};

const DEFAULT_SAMPLES: usize = 1_000_000;
const BATCH_SIZE: usize = 10_000;
const MSG_LEN: usize = 64;
const SIMD_MSG_LEN: usize = CHUNK_SIZE * 4; // 256 KB -> forces 4-way AVX2/NEON kernel
const T_THRESHOLD: f64 = 4.5;

#[derive(Clone, Copy)]
struct SplitMix64(u64);

impl SplitMix64 {
    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline(always)]
    fn fill_bytes(&mut self, dst: &mut [u8]) {
        let mut i = 0usize;
        while i + 8 <= dst.len() {
            dst[i..i + 8].copy_from_slice(&self.next_u64().to_le_bytes());
            i += 8;
        }
        if i < dst.len() {
            let tail = self.next_u64().to_le_bytes();
            let rem = dst.len() - i;
            dst[i..].copy_from_slice(&tail[..rem]);
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Welford {
    n: u64,
    mean: f64,
    m2: f64,
}

impl Welford {
    #[inline(always)]
    fn push(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / (self.n as f64);
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    #[inline(always)]
    fn variance(&self) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            self.m2 / ((self.n - 1) as f64)
        }
    }

    fn welch_t(&self, other: &Self) -> f64 {
        if self.n < 2 || other.n < 2 {
            return 0.0;
        }
        let num = self.mean - other.mean;
        let den = (self.variance() / (self.n as f64) + other.variance() / (other.n as f64)).sqrt();
        if den == 0.0 {
            0.0
        } else {
            num / den
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn read_ticks() -> u64 {
    // SAFETY: LFENCE + RDTSC are always available on x86-64.
    unsafe {
        core::arch::x86_64::_mm_lfence();
        let t = core::arch::x86_64::_rdtsc();
        core::arch::x86_64::_mm_lfence();
        t
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
fn read_ticks() -> u64 {
    use std::time::Instant;
    static mut ORIGIN: Option<Instant> = None;
    let now = Instant::now();
    // SAFETY: Single-threaded benchmark harness.
    let start = unsafe { *ORIGIN.get_or_insert(now) };
    now.duration_since(start).as_nanos() as u64
}

fn summarize_suite(
    name: &str,
    total_samples: usize,
    percentiles: &[f64; 6],
    cutoffs: &[u64; 6],
    stats0: &[Welford; 6],
    stats1: &[Welford; 6],
) -> bool {
    println!("=== {} ({} samples) ===", name, total_samples);
    let mut max_abs_t = 0.0f64;
    for i in 0..6 {
        let t = stats0[i].welch_t(&stats1[i]);
        let abs_t = t.abs();
        if abs_t > max_abs_t {
            max_abs_t = abs_t;
        }
        println!(
            "  Crop p={:4.2} (<= {:8} ticks) | n0={:7}, n1={:7} | mean0={:9.1}, mean1={:9.1} | t = {:+6.3}",
            percentiles[i],
            if cutoffs[i] == u64::MAX { 0 } else { cutoffs[i] },
            stats0[i].n,
            stats1[i].n,
            stats0[i].mean,
            stats1[i].mean,
            t
        );
    }
    let passed = max_abs_t < T_THRESHOLD;
    println!(
        "  Max |t| = {:.4} (threshold < {:.1}) -> {}\n",
        max_abs_t,
        T_THRESHOLD,
        if passed { "PASS" } else { "FAIL" }
    );
    passed
}

#[derive(Clone, Copy)]
struct ScalarSample {
    key: [u8; KEY_LEN],
    nonce: [u8; NONCE_LEN],
    msg: [u8; MSG_LEN],
}

#[inline(never)]
fn measure_scalar_enc(s: &mut ScalarSample, ad: &[u8]) -> u64 {
    let t0 = read_ticks();
    let tag = encrypt_in_place(
        black_box(&s.key),
        black_box(&s.nonce),
        black_box(ad),
        Profile::EnterpriseCmt256,
        black_box(&mut s.msg[..]),
    )
    .unwrap();
    let t1 = read_ticks();
    black_box(tag);
    t1.wrapping_sub(t0)
}

fn run_suite_scalar_encrypt(rng: &mut SplitMix64, total_samples: usize) -> bool {
    let fixed = ScalarSample {
        key: [0x00u8; KEY_LEN],
        nonce: [0x00u8; NONCE_LEN],
        msg: [0x00u8; MSG_LEN],
    };
    let ad = b"DUDECT-NIBBLX512-V311";
    let mut samples = vec![fixed; BATCH_SIZE];
    let mut classes = vec![0u8; BATCH_SIZE];
    let mut ticks = vec![0u64; BATCH_SIZE];

    let prepare = |rng: &mut SplitMix64, samples: &mut [ScalarSample], classes: &mut [u8]| {
        for i in 0..samples.len() {
            let c = (rng.next_u64() & 1) as u8;
            classes[i] = c;
            if c == 0 {
                samples[i] = fixed;
            } else {
                rng.fill_bytes(&mut samples[i].key);
                rng.fill_bytes(&mut samples[i].nonce);
                rng.fill_bytes(&mut samples[i].msg);
            }
        }
    };

    prepare(rng, &mut samples, &mut classes);
    for i in 0..BATCH_SIZE {
        ticks[i] = measure_scalar_enc(&mut samples[i], ad);
    }
    let mut sorted = ticks.clone();
    sorted.sort_unstable();
    let percentiles = [1.00, 0.99, 0.98, 0.95, 0.90, 0.80];
    let mut cutoffs = [u64::MAX; 6];
    for (idx, &p) in percentiles.iter().enumerate() {
        if p < 1.0 {
            let pos = ((BATCH_SIZE as f64) * p) as usize;
            cutoffs[idx] = sorted[pos.min(BATCH_SIZE - 1)];
        }
    }

    let mut stats0 = [Welford::default(); 6];
    let mut stats1 = [Welford::default(); 6];
    let mut done = 0usize;
    while done < total_samples {
        let n = (total_samples - done).min(BATCH_SIZE);
        prepare(rng, &mut samples[..n], &mut classes[..n]);
        for i in 0..n {
            ticks[i] = measure_scalar_enc(&mut samples[i], ad);
        }
        for i in 0..n {
            let tk = ticks[i];
            let x = tk as f64;
            let c = classes[i];
            for k in 0..6 {
                if tk <= cutoffs[k] {
                    if c == 0 {
                        stats0[k].push(x);
                    } else {
                        stats1[k].push(x);
                    }
                }
            }
        }
        done += n;
    }

    summarize_suite(
        "Suite 1: Scalar encrypt_in_place (64 B)",
        total_samples,
        &percentiles,
        &cutoffs,
        &stats0,
        &stats1,
    )
}

#[inline(never)]
fn measure_simd_enc(key: &[u8; KEY_LEN], nonce: &[u8; NONCE_LEN], ad: &[u8], buf: &mut [u8]) -> u64 {
    let t0 = read_ticks();
    let tag = encrypt_in_place(
        black_box(key),
        black_box(nonce),
        black_box(ad),
        Profile::EnterpriseCmt256,
        black_box(buf),
    )
    .unwrap();
    let t1 = read_ticks();
    black_box(tag);
    t1.wrapping_sub(t0)
}

fn run_suite_simd_4way_encrypt(rng: &mut SplitMix64, total_samples: usize) -> bool {
    let simd_samples = (total_samples / 250).max(2_000);
    let simd_batch = 64usize;
    let ad = b"DUDECT-NIBBLX512-SIMD4";

    let mut keys = vec![[0u8; KEY_LEN]; simd_batch];
    let mut nonces = vec![[0u8; NONCE_LEN]; simd_batch];
    let mut classes = vec![0u8; simd_batch];
    let mut ticks = vec![0u64; simd_batch];
    let mut buf_fixed = vec![0x00u8; SIMD_MSG_LEN];
    let mut buf_rand = vec![0x00u8; SIMD_MSG_LEN];
    rng.fill_bytes(&mut buf_rand);

    let mut work = vec![0u8; SIMD_MSG_LEN];
    let prepare = |rng: &mut SplitMix64,
                   keys: &mut [[u8; KEY_LEN]],
                   nonces: &mut [[u8; NONCE_LEN]],
                   classes: &mut [u8]| {
        for i in 0..classes.len() {
            let c = (rng.next_u64() & 1) as u8;
            classes[i] = c;
            if c == 0 {
                keys[i] = [0u8; KEY_LEN];
                nonces[i] = [0u8; NONCE_LEN];
            } else {
                rng.fill_bytes(&mut keys[i]);
                rng.fill_bytes(&mut nonces[i]);
            }
        }
    };

    // Calibration batch
    prepare(rng, &mut keys, &mut nonces, &mut classes);
    for i in 0..simd_batch {
        if classes[i] == 0 {
            work.copy_from_slice(&buf_fixed);
        } else {
            work.copy_from_slice(&buf_rand);
        }
        ticks[i] = measure_simd_enc(&keys[i], &nonces[i], ad, &mut work);
    }
    let mut sorted = ticks.clone();
    sorted.sort_unstable();
    let percentiles = [1.00, 0.99, 0.98, 0.95, 0.90, 0.80];
    let mut cutoffs = [u64::MAX; 6];
    for (idx, &p) in percentiles.iter().enumerate() {
        if p < 1.0 {
            let pos = ((simd_batch as f64) * p) as usize;
            cutoffs[idx] = sorted[pos.min(simd_batch - 1)];
        }
    }

    let mut stats0 = [Welford::default(); 6];
    let mut stats1 = [Welford::default(); 6];
    let mut done = 0usize;
    while done < simd_samples {
        let n = (simd_samples - done).min(simd_batch);
        prepare(rng, &mut keys[..n], &mut nonces[..n], &mut classes[..n]);
        for i in 0..n {
            // Identical copy cost for both classes before timing starts
            let src = if classes[i] == 0 { &buf_fixed } else { &buf_rand };
            work.copy_from_slice(src);
            ticks[i] = measure_simd_enc(&keys[i], &nonces[i], ad, &mut work);
        }
        for i in 0..n {
            let tk = ticks[i];
            let x = tk as f64;
            let c = classes[i];
            for k in 0..6 {
                if tk <= cutoffs[k] {
                    if c == 0 {
                        stats0[k].push(x);
                    } else {
                        stats1[k].push(x);
                    }
                }
            }
        }
        done += n;
    }
    black_box(&mut buf_fixed);

    summarize_suite(
        "Suite 2: 4-Way SIMD encrypt_in_place (256 KB)",
        simd_samples,
        &percentiles,
        &cutoffs,
        &stats0,
        &stats1,
    )
}

#[derive(Clone, Copy)]
struct TagSample {
    ct: [u8; MSG_LEN],
    tag: MorphicTag,
}

#[inline(never)]
fn measure_tag_verify(
    key: &[u8; KEY_LEN],
    nonce: &[u8; NONCE_LEN],
    ad: &[u8],
    s: &mut TagSample,
) -> u64 {
    let t0 = read_ticks();
    let res = decrypt_in_place(
        black_box(key),
        black_box(nonce),
        black_box(ad),
        Profile::EnterpriseCmt256,
        black_box(&mut s.ct[..]),
        black_box(&s.tag),
    );
    let t1 = read_ticks();
    let _ = black_box(res);
    t1.wrapping_sub(t0)
}

fn run_suite_decrypt_tag_ct(rng: &mut SplitMix64, total_samples: usize) -> bool {
    let tag_samples = (total_samples / 2).max(100_000);
    let key = [0x42u8; KEY_LEN];
    let nonce = [0x24u8; NONCE_LEN];
    let ad = b"DUDECT-TAG-VERIFY";

    let mut base_ct = [0x5Au8; MSG_LEN];
    let valid_tag =
        encrypt_in_place(&key, &nonce, ad, Profile::EnterpriseCmt256, &mut base_ct).unwrap();
    let valid_bytes = valid_tag.as_bytes();

    // Class 0: mismatch at byte 0; Class 1: matches first 31 bytes, mismatch only at byte 31
    let mut bad_first = [0u8; 32];
    bad_first.copy_from_slice(valid_bytes);
    bad_first[0] ^= 0xFF;

    let mut bad_last = [0u8; 32];
    bad_last.copy_from_slice(valid_bytes);
    bad_last[31] ^= 0xFF;

    let s0 = TagSample {
        ct: base_ct,
        tag: MorphicTag::Commit256(bad_first),
    };
    let s1 = TagSample {
        ct: base_ct,
        tag: MorphicTag::Commit256(bad_last),
    };

    let mut samples = vec![s0; BATCH_SIZE];
    let mut classes = vec![0u8; BATCH_SIZE];
    let mut ticks = vec![0u64; BATCH_SIZE];

    let prepare = |rng: &mut SplitMix64, samples: &mut [TagSample], classes: &mut [u8]| {
        for i in 0..samples.len() {
            let c = (rng.next_u64() & 1) as u8;
            classes[i] = c;
            samples[i] = if c == 0 { s0 } else { s1 };
        }
    };

    prepare(rng, &mut samples, &mut classes);
    for i in 0..BATCH_SIZE {
        ticks[i] = measure_tag_verify(&key, &nonce, ad, &mut samples[i]);
    }
    let mut sorted = ticks.clone();
    sorted.sort_unstable();
    let percentiles = [1.00, 0.99, 0.98, 0.95, 0.90, 0.80];
    let mut cutoffs = [u64::MAX; 6];
    for (idx, &p) in percentiles.iter().enumerate() {
        if p < 1.0 {
            let pos = ((BATCH_SIZE as f64) * p) as usize;
            cutoffs[idx] = sorted[pos.min(BATCH_SIZE - 1)];
        }
    }

    let mut stats0 = [Welford::default(); 6];
    let mut stats1 = [Welford::default(); 6];
    let mut done = 0usize;
    while done < tag_samples {
        let n = (tag_samples - done).min(BATCH_SIZE);
        prepare(rng, &mut samples[..n], &mut classes[..n]);
        for i in 0..n {
            ticks[i] = measure_tag_verify(&key, &nonce, ad, &mut samples[i]);
        }
        for i in 0..n {
            let tk = ticks[i];
            let x = tk as f64;
            let c = classes[i];
            for k in 0..6 {
                if tk <= cutoffs[k] {
                    if c == 0 {
                        stats0[k].push(x);
                    } else {
                        stats1[k].push(x);
                    }
                }
            }
        }
        done += n;
    }

    summarize_suite(
        "Suite 3: decrypt_in_place Tag Verify (byte-0 vs byte-31 mismatch)",
        tag_samples,
        &percentiles,
        &cutoffs,
        &stats0,
        &stats1,
    )
}

fn main() {
    let total_samples: usize = env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_SAMPLES);

    let mut rng = SplitMix64(0xD0DE_C700_5120_0031);

    let ok1 = run_suite_scalar_encrypt(&mut rng, total_samples);
    let ok2 = run_suite_simd_4way_encrypt(&mut rng, total_samples);
    let ok3 = run_suite_decrypt_tag_ct(&mut rng, total_samples);

    if ok1 && ok2 && ok3 {
        println!("OVERALL DUDECT AUDIT: PASS (all 3 suites < {:.1})", T_THRESHOLD);
    } else {
        println!("OVERALL DUDECT AUDIT: FAIL (timing leak or heavy OS jitter detected)");
        process::exit(1);
    }
}
