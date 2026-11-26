//! Standalone zero-dependency throughput & Cycles-per-Byte benchmark for NIBBLX-512 v2.
//!
//! Run on Windows PowerShell:
//!   $env:RUSTFLAGS="-C target-cpu=native"; cargo run --release --example bench
//!
//! Run on Linux/macOS:
//!   RUSTFLAGS="-C target-cpu=native" cargo run --release --example bench

use std::thread;
use std::time::Instant;

use nibblx512::chunk::{
    compute_ad_digest, derive_key_commit_ctx, encrypt_4_full_chunks, encrypt_chunk,
};
use nibblx512::stream::{encrypt_in_place_scalar, RootFinalizer};
use nibblx512::{
    decrypt_in_place, encrypt_in_place, MorphicTag, Profile, CHUNK_SIZE, KEY_LEN, NONCE_LEN,
};

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn read_cycles() -> Option<u64> {
    // SAFETY: RDTSC is always available on x86-64.
    Some(unsafe { core::arch::x86_64::_rdtsc() })
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
fn read_cycles() -> Option<u64> {
    None
}

fn main() {
    let key = [0x42u8; KEY_LEN];
    let nonce = [0x99u8; NONCE_LEN];
    let ad = b"MORPHIC-SPONGE-ARX-512-BENCH";
    let profile = Profile::EnterpriseCmt256;

    // 64 MB buffer = 1,024 chunks of 64 KB (exact 4-chunk multiple)
    let len_1c = CHUNK_SIZE * 1024;
    let mut buf_scalar = vec![0x5Au8; len_1c];
    let mut buf_simd = vec![0x5Au8; len_1c];

    println!("=== NIBBLX-512 v3.1.1 Seal-Pack Benchmark (EnterpriseCmt256) ===");
    println!("Payload per single-core pass: {} MB ({} x 64 KB chunks)", len_1c / (1024 * 1024), len_1c / CHUNK_SIZE);

    // 1. Scalar 1-Core
    let c0 = read_cycles();
    let t0 = Instant::now();
    let tag_scalar = encrypt_in_place_scalar(&key, &nonce, ad, profile, &mut buf_scalar).unwrap();
    let dt_scalar = t0.elapsed().as_secs_f64();
    let c1 = read_cycles();
    let mbps_scalar = (len_1c as f64) / (1024.0 * 1024.0) / dt_scalar;
    match (c0, c1) {
        (Some(a), Some(b)) => {
            let cpb = (b.wrapping_sub(a) as f64) / (len_1c as f64);
            println!("1-Core Scalar       : {:8.1} MB/s ({:.2} GB/s) | {:5.2} cpb", mbps_scalar, mbps_scalar / 1024.0, cpb);
        }
        _ => println!("1-Core Scalar       : {:8.1} MB/s ({:.2} GB/s)", mbps_scalar, mbps_scalar / 1024.0),
    }

    // 2. Hybrid 4-Way SIMD 1-Core (AVX2 / NEON)
    let c0 = read_cycles();
    let t0 = Instant::now();
    let tag_simd = encrypt_in_place(&key, &nonce, ad, profile, &mut buf_simd).unwrap();
    let dt_simd = t0.elapsed().as_secs_f64();
    let c1 = read_cycles();
    let mbps_simd = (len_1c as f64) / (1024.0 * 1024.0) / dt_simd;
    assert_eq!(tag_scalar, tag_simd, "SIMD tag mismatch vs Scalar!");
    assert_eq!(buf_scalar, buf_simd, "SIMD ciphertext mismatch vs Scalar!");
    match (c0, c1) {
        (Some(a), Some(b)) => {
            let cpb = (b.wrapping_sub(a) as f64) / (len_1c as f64);
            println!(
                "1-Core 4-Way SIMD   : {:8.1} MB/s ({:.2} GB/s) | {:5.2} cpb ({:.2}x speedup)",
                mbps_simd,
                mbps_simd / 1024.0,
                cpb,
                mbps_simd / mbps_scalar
            );
        }
        _ => println!(
            "1-Core 4-Way SIMD   : {:8.1} MB/s ({:.2} GB/s) ({:.2}x speedup)",
            mbps_simd,
            mbps_simd / 1024.0,
            mbps_simd / mbps_scalar
        ),
    }

    // Verify full 64 MB decrypt + tag integrity
    decrypt_in_place(&key, &nonce, ad, profile, &mut buf_simd, &tag_simd).unwrap();
    assert!(buf_simd.iter().all(|&b| b == 0x5A));

    // 3. Multi-Core Parallel 64 KB Chunk Tree (256 MB across all CPU threads, zero external deps)
    let threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let chunks_per_thread = 256usize; // 16 MB per thread (multiple of 4)
    let total_chunks = threads * chunks_per_thread;
    let total_bytes = total_chunks * CHUNK_SIZE;
    let mut big_buf = vec![0x33u8; total_bytes];
    let mut chunk_tags = vec![MorphicTag::Commit256([0u8; 32]); total_chunks];

    let ctx = derive_key_commit_ctx(&key, &nonce, profile);
    let ad_digest = compute_ad_digest(&key, &nonce, &ctx, profile, ad);

    let t0 = Instant::now();
    thread::scope(|s| {
        let mut buf_rest = big_buf.as_mut_slice();
        let mut tag_rest = chunk_tags.as_mut_slice();
        for tid in 0..threads {
            let (my_buf, b_next) = buf_rest.split_at_mut(chunks_per_thread * CHUNK_SIZE);
            let (my_tags, t_next) = tag_rest.split_at_mut(chunks_per_thread);
            buf_rest = b_next;
            tag_rest = t_next;

            let base_idx = (tid * chunks_per_thread) as u64;
            let total_chunks_u64 = total_chunks as u64;
            let key_ref = &key;
            let nonce_ref = &nonce;
            let ctx_ref = &ctx;
            let ad_ref = &ad_digest;

            s.spawn(move || {
                let mut idx = base_idx;
                let mut rest = my_buf;
                let mut t_off = 0usize;
                let batch_bytes = CHUNK_SIZE * 4;
                while rest.len() >= batch_bytes {
                    let (batch, tail) = rest.split_at_mut(batch_bytes);
                    let (c0, r1) = batch.split_at_mut(CHUNK_SIZE);
                    let (c1, r2) = r1.split_at_mut(CHUNK_SIZE);
                    let (c2, c3) = r2.split_at_mut(CHUNK_SIZE);
                    let last_final = (idx + 4) == total_chunks_u64;
                    let tags = encrypt_4_full_chunks(
                        key_ref,
                        nonce_ref,
                        ctx_ref,
                        ad_ref,
                        profile,
                        idx,
                        last_final,
                        [c0, c1, c2, c3],
                    )
                    .unwrap();
                    my_tags[t_off..t_off + 4].copy_from_slice(&tags);
                    idx += 4;
                    t_off += 4;
                    rest = tail;
                }
                for chunk in rest.chunks_mut(CHUNK_SIZE) {
                    let is_final = (idx + 1) == total_chunks_u64;
                    my_tags[t_off] = encrypt_chunk(
                        key_ref, nonce_ref, ctx_ref, ad_ref, profile, idx, is_final, chunk,
                    )
                    .unwrap();
                    idx += 1;
                    t_off += 1;
                }
            });
        }
    });

    let mut root = RootFinalizer::new(&key, &nonce, &ctx, &ad_digest, profile);
    for t in chunk_tags.iter() {
        root.absorb_chunk_tag(t, CHUNK_SIZE).unwrap();
    }
    let _master = root.finalize();
    let dt_mc = t0.elapsed().as_secs_f64();
    let mbps_mc = (total_bytes as f64) / (1024.0 * 1024.0) / dt_mc;

    println!(
        "{}-Core Parallel SIMD : {:8.1} MB/s ({:.2} GB/s) ({} MB total)",
        threads,
        mbps_mc,
        mbps_mc / 1024.0,
        total_bytes / (1024 * 1024)
    );
    println!("Integrity Check     : PASS (64 MB round-trip + SIMD==Scalar bit-exact)");
}
