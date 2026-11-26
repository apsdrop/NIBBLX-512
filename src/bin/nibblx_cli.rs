//! Zero-dependency O(1)-RAM Multi-GB Streaming File Encryption CLI for NIBBLX-512 v3.1.1.
//!
//! Uses OS CSPRNG (`BCryptGenRandom` on Windows, `/dev/urandom` on Unix), 16 MiB pure NIBBLX-512
//! hybrid memory-hard KDF (`kdf::derive_key_memory_hard`), 64-byte authenticated `.mrf3` header
//! bound as AD, 4-chunk (256 KB) AVX2/NEON SIMD streaming batches with immediate per-chunk CMT-4
//! verification, and atomic temp-file rename after Master Tag verification.
//!
//! Safe password usage (recommended — avoids exposing password in process lists / shell history):
//!   NIBBLX_PASS="..." nibblx_cli encrypt <input_file> <output.mrf3>
//!   printf '%s\n' "..." | nibblx_cli encrypt <input_file> <output.mrf3> --pass-stdin
//!
//! Legacy positional password (emits a warning on stderr):
//!   nibblx_cli encrypt <input_file> <output.mrf3> <password>

use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::process;
use std::time::Instant;

use nibblx512::ct_ops;
use nibblx512::kdf;
use nibblx512::{
    MorphicTag, Profile, StreamDecryptor, StreamEncryptor, CHUNK_SIZE, NONCE_LEN,
};

const MAGIC: &[u8; 8] = b"MRF3ARX\0";
const HEADER_LEN: usize = 64;
const DEFAULT_KDF_MIB: u16 = 16;
const DEFAULT_KDF_PASSES: u32 = 3;

#[cfg(windows)]
fn os_random_bytes(out: &mut [u8]) -> io::Result<()> {
    #[link(name = "bcrypt")]
    extern "system" {
        fn BCryptGenRandom(
            h_algorithm: *mut core::ffi::c_void,
            pb_buffer: *mut u8,
            cb_buffer: u32,
            dw_flags: u32,
        ) -> i32;
    }
    const BCRYPT_USE_SYSTEM_PREFERRED_RNG: u32 = 0x0000_0002;
    // SAFETY: `out` is a valid mutable byte slice and `BCRYPT_USE_SYSTEM_PREFERRED_RNG` accepts NULL handle.
    let status = unsafe {
        BCryptGenRandom(
            core::ptr::null_mut(),
            out.as_mut_ptr(),
            out.len() as u32,
            BCRYPT_USE_SYSTEM_PREFERRED_RNG,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(io::Error::new(io::ErrorKind::Other, "BCryptGenRandom failed"))
    }
}

#[cfg(unix)]
fn os_random_bytes(out: &mut [u8]) -> io::Result<()> {
    let mut f = File::open("/dev/urandom")?;
    f.read_exact(out)
}

fn write_frame<W: Write>(
    w: &mut W,
    data: &[u8],
    is_final: bool,
    tag: &MorphicTag,
) -> io::Result<()> {
    w.write_all(&(data.len() as u32).to_le_bytes())?;
    w.write_all(&[if is_final { 1u8 } else { 0u8 }])?;
    w.write_all(data)?;
    w.write_all(tag.as_bytes())
}

fn encrypt_file(in_path: &str, out_path: &str, password: &[u8]) -> Result<(), String> {
    let t0 = Instant::now();
    let profile = Profile::EnterpriseCmt256;
    let file_len = fs::metadata(in_path)
        .map_err(|e| format!("cannot stat input: {}", e))?
        .len();

    let mut salt = [0u8; NONCE_LEN];
    let mut nonce = [0u8; NONCE_LEN];
    os_random_bytes(&mut salt).map_err(|e| format!("CSPRNG salt error: {}", e))?;
    os_random_bytes(&mut nonce).map_err(|e| format!("CSPRNG nonce error: {}", e))?;

    // 16 MiB = 262,144 x 64-byte blocks
    let block_count = (DEFAULT_KDF_MIB as usize) * 16_384;
    let mut mem = vec![[0u64; 8]; block_count];
    let mut key = kdf::derive_key_memory_hard(password, &salt, &mut mem, DEFAULT_KDF_PASSES)
        .map_err(|e| format!("KDF failed: {}", e))?;
    drop(mem);

    // Build 64-byte authenticated `.mrf3` header (bound as AD)
    let mut header = [0u8; HEADER_LEN];
    header[0..8].copy_from_slice(MAGIC);
    header[8..10].copy_from_slice(&(profile.ver_profile_id() as u16).to_le_bytes());
    header[10..12].copy_from_slice(&DEFAULT_KDF_MIB.to_le_bytes());
    header[12..16].copy_from_slice(&DEFAULT_KDF_PASSES.to_le_bytes());
    header[16..40].copy_from_slice(&salt);
    header[40..64].copy_from_slice(&nonce);

    let mut reader = BufReader::new(File::open(in_path).map_err(|e| e.to_string())?);
    let mut writer = BufWriter::new(File::create(out_path).map_err(|e| e.to_string())?);
    writer.write_all(&header).map_err(|e| e.to_string())?;

    let mut enc = StreamEncryptor::new(&key, &nonce, &header, profile);

    if file_len == 0 {
        let mut empty = [];
        let tag = enc
            .encrypt_next_chunk(&mut empty, true)
            .map_err(|e| e.to_string())?;
        write_frame(&mut writer, &empty, true, &tag).map_err(|e| e.to_string())?;
    } else {
        let mut remaining = file_len;
        let batch_bytes = (CHUNK_SIZE as u64) * 4;
        let mut b0 = vec![0u8; CHUNK_SIZE];
        let mut b1 = vec![0u8; CHUNK_SIZE];
        let mut b2 = vec![0u8; CHUNK_SIZE];
        let mut b3 = vec![0u8; CHUNK_SIZE];

        while remaining >= batch_bytes {
            reader.read_exact(&mut b0).map_err(|e| e.to_string())?;
            reader.read_exact(&mut b1).map_err(|e| e.to_string())?;
            reader.read_exact(&mut b2).map_err(|e| e.to_string())?;
            reader.read_exact(&mut b3).map_err(|e| e.to_string())?;

            let last_is_final = remaining == batch_bytes;
            let tags = enc
                .encrypt_4_chunks([&mut b0, &mut b1, &mut b2, &mut b3], last_is_final)
                .map_err(|e| e.to_string())?;

            write_frame(&mut writer, &b0, false, &tags[0]).map_err(|e| e.to_string())?;
            write_frame(&mut writer, &b1, false, &tags[1]).map_err(|e| e.to_string())?;
            write_frame(&mut writer, &b2, false, &tags[2]).map_err(|e| e.to_string())?;
            write_frame(&mut writer, &b3, last_is_final, &tags[3]).map_err(|e| e.to_string())?;

            remaining -= batch_bytes;
        }

        while remaining > 0 {
            let n = (remaining.min(CHUNK_SIZE as u64)) as usize;
            let is_final = remaining == (n as u64);
            reader
                .read_exact(&mut b0[..n])
                .map_err(|e| e.to_string())?;
            let tag = enc
                .encrypt_next_chunk(&mut b0[..n], is_final)
                .map_err(|e| e.to_string())?;
            write_frame(&mut writer, &b0[..n], is_final, &tag).map_err(|e| e.to_string())?;
            remaining -= n as u64;
        }

        ct_ops::secure_zeroize_bytes(&mut b0);
        ct_ops::secure_zeroize_bytes(&mut b1);
        ct_ops::secure_zeroize_bytes(&mut b2);
        ct_ops::secure_zeroize_bytes(&mut b3);
    }

    let master = enc.finish().map_err(|e| e.to_string())?;
    writer
        .write_all(master.as_bytes())
        .map_err(|e| e.to_string())?;
    writer.flush().map_err(|e| e.to_string())?;
    ct_ops::secure_zeroize_bytes(&mut key);

    let secs = t0.elapsed().as_secs_f64();
    println!(
        "[NIBBLX-512 v3.1.1] Encrypted {} bytes -> {} in {:.3}s ({:.1} MB/s incl. KDF+IO)",
        file_len,
        out_path,
        secs,
        (file_len as f64) / (1024.0 * 1024.0) / secs.max(1e-6)
    );
    Ok(())
}

fn read_tag<R: Read>(r: &mut R, profile: Profile) -> io::Result<MorphicTag> {
    match profile {
        Profile::Compact128 => {
            let mut t = [0u8; 16];
            r.read_exact(&mut t)?;
            Ok(MorphicTag::Compact128(t))
        }
        Profile::EnterpriseCmt256 | Profile::HighSecCmt256 => {
            let mut t = [0u8; 32];
            r.read_exact(&mut t)?;
            Ok(MorphicTag::Commit256(t))
        }
    }
}

fn decrypt_file(in_path: &str, out_path: &str, password: &[u8]) -> Result<(), String> {
    let t0 = Instant::now();
    let mut reader = BufReader::new(File::open(in_path).map_err(|e| e.to_string())?);

    let mut header = [0u8; HEADER_LEN];
    reader
        .read_exact(&mut header)
        .map_err(|_| "truncated or invalid .mrf3 header".to_string())?;
    if &header[0..8] != MAGIC {
        return Err("invalid magic bytes (not a NIBBLX-512 .mrf3 file)".to_string());
    }

    let ver_id = u16::from_le_bytes([header[8], header[9]]);
    let profile = Profile::from_ver_id(ver_id)
        .ok_or_else(|| format!("unsupported protocol version 0x{:04X}", ver_id))?;
    let kdf_mib = u16::from_le_bytes([header[10], header[11]]);
    let kdf_passes = u32::from_le_bytes([header[12], header[13], header[14], header[15]]);
    if kdf_mib == 0 || kdf_mib > 1024 || kdf_passes == 0 {
        return Err("invalid KDF parameters in header".to_string());
    }

    let mut salt = [0u8; NONCE_LEN];
    let mut nonce = [0u8; NONCE_LEN];
    salt.copy_from_slice(&header[16..40]);
    nonce.copy_from_slice(&header[40..64]);

    let block_count = (kdf_mib as usize) * 16_384;
    let mut mem = vec![[0u64; 8]; block_count];
    let mut key = kdf::derive_key_memory_hard(password, &salt, &mut mem, kdf_passes)
        .map_err(|e| format!("KDF failed: {}", e))?;
    drop(mem);

    // Write verified chunks to a temporary file and atomically rename ONLY after Master Tag passes.
    let tmp_path = format!("{}.nibblx.tmp", out_path);

    let result = (|| -> Result<u64, String> {
        let mut writer = BufWriter::new(File::create(&tmp_path).map_err(|e| e.to_string())?);
        let mut dec = StreamDecryptor::new(&key, &nonce, &header, profile);

        let mut bufs: [Vec<u8>; 4] = [
            vec![0u8; CHUNK_SIZE],
            vec![0u8; CHUNK_SIZE],
            vec![0u8; CHUNK_SIZE],
            vec![0u8; CHUNK_SIZE],
        ];
        let mut tags = [MorphicTag::Commit256([0u8; 32]); 4];
        let mut queued = 0usize;
        let mut total_out = 0u64;

        loop {
            let mut len_b = [0u8; 4];
            let mut flag_b = [0u8; 1];
            reader
                .read_exact(&mut len_b)
                .map_err(|_| "unexpected EOF reading frame header".to_string())?;
            reader
                .read_exact(&mut flag_b)
                .map_err(|_| "unexpected EOF reading frame flag".to_string())?;

            let chunk_len = u32::from_le_bytes(len_b) as usize;
            let is_final = match flag_b[0] {
                0 => false,
                1 => true,
                _ => return Err("invalid frame flag".to_string()),
            };
            if chunk_len > CHUNK_SIZE || (!is_final && chunk_len != CHUNK_SIZE) {
                return Err("invalid chunk frame length".to_string());
            }

            reader
                .read_exact(&mut bufs[queued][..chunk_len])
                .map_err(|_| "unexpected EOF reading chunk payload".to_string())?;
            tags[queued] = read_tag(&mut reader, profile)
                .map_err(|_| "unexpected EOF reading chunk tag".to_string())?;

            if !is_final && chunk_len == CHUNK_SIZE {
                queued += 1;
                if queued == 4 {
                    let [b0, b1, b2, b3] = &mut bufs;
                    dec.decrypt_4_chunks([b0, b1, b2, b3], &tags, false)
                        .map_err(|e| e.to_string())?;
                    for b in bufs.iter() {
                        writer.write_all(b).map_err(|e| e.to_string())?;
                    }
                    total_out += (CHUNK_SIZE as u64) * 4;
                    queued = 0;
                }
            } else {
                // Flush any queued full chunks + final chunk
                if queued == 3 && chunk_len == CHUNK_SIZE {
                    let [b0, b1, b2, b3] = &mut bufs;
                    dec.decrypt_4_chunks([b0, b1, b2, b3], &tags, true)
                        .map_err(|e| e.to_string())?;
                    for b in bufs.iter() {
                        writer.write_all(b).map_err(|e| e.to_string())?;
                    }
                    total_out += (CHUNK_SIZE as u64) * 4;
                } else {
                    for q in 0..queued {
                        dec.decrypt_next_chunk(&mut bufs[q][..CHUNK_SIZE], &tags[q], false)
                            .map_err(|e| e.to_string())?;
                        writer
                            .write_all(&bufs[q][..CHUNK_SIZE])
                            .map_err(|e| e.to_string())?;
                        total_out += CHUNK_SIZE as u64;
                    }
                    dec.decrypt_next_chunk(&mut bufs[queued][..chunk_len], &tags[queued], true)
                        .map_err(|e| e.to_string())?;
                    writer
                        .write_all(&bufs[queued][..chunk_len])
                        .map_err(|e| e.to_string())?;
                    total_out += chunk_len as u64;
                }
                break;
            }
        }

        for b in bufs.iter_mut() {
            ct_ops::secure_zeroize_bytes(b);
        }

        let master = read_tag(&mut reader, profile)
            .map_err(|_| "unexpected EOF reading Master Tag".to_string())?;
        dec.finish_and_verify(&master).map_err(|e| e.to_string())?;
        writer.flush().map_err(|e| e.to_string())?;
        drop(writer);

        fs::rename(&tmp_path, out_path).map_err(|e| format!("atomic rename failed: {}", e))?;
        Ok(total_out)
    })();

    ct_ops::secure_zeroize_bytes(&mut key);

    match result {
        Ok(total_out) => {
            let secs = t0.elapsed().as_secs_f64();
            println!(
                "[NIBBLX-512 v3.1.1] Decrypted & Verified {} bytes -> {} in {:.3}s ({:.1} MB/s incl. KDF+IO)",
                total_out,
                out_path,
                secs,
                (total_out as f64) / (1024.0 * 1024.0) / secs.max(1e-6)
            );
            Ok(())
        }
        Err(e) => {
            let _ = fs::remove_file(&tmp_path);
            Err(e)
        }
    }
}

fn read_password_from_stdin() -> Result<Vec<u8>, String> {
    let mut line = String::new();
    io::stdin()
        .lock()
        .read_line(&mut line)
        .map_err(|e| format!("failed reading password from stdin: {}", e))?;
    while line.ends_with('\n') || line.ends_with('\r') {
        line.pop();
    }
    Ok(line.into_bytes())
}

fn resolve_password(args: &[String]) -> Result<Vec<u8>, String> {
    if args.len() == 5 && args[4] == "--pass-stdin" {
        return read_password_from_stdin();
    }
    if args.len() == 4 {
        if let Ok(pw) = env::var("NIBBLX_PASS") {
            return Ok(pw.into_bytes());
        }
        eprint!("Enter password (stdin): ");
        let _ = io::stderr().flush();
        return read_password_from_stdin();
    }
    if args.len() == 5 {
        eprintln!(
            "[WARN] Passing password as a CLI argument exposes it to local process lists; \
             prefer NIBBLX_PASS env or --pass-stdin."
        );
        return Ok(args[4].as_bytes().to_vec());
    }
    Err("invalid arguments".to_string())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 && args.len() != 5 {
        eprintln!("NIBBLX-512 v3.1.1 — Zero-Copy Streaming File Encryption CLI");
        eprintln!("Usage (recommended — no password in process list):");
        eprintln!("  NIBBLX_PASS=... nibblx_cli encrypt <input> <output.mrf3>");
        eprintln!("  printf '%s\\n' ... | nibblx_cli encrypt <input> <output.mrf3> --pass-stdin");
        eprintln!("  NIBBLX_PASS=... nibblx_cli decrypt <input.mrf3> <output>");
        process::exit(2);
    }

    let mut password = match resolve_password(&args) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("ERROR: {}", e);
            process::exit(2);
        }
    };

    let res = match args[1].as_str() {
        "encrypt" => encrypt_file(&args[2], &args[3], &password),
        "decrypt" => decrypt_file(&args[2], &args[3], &password),
        _ => Err("unknown command (expected 'encrypt' or 'decrypt')".to_string()),
    };

    ct_ops::secure_zeroize_bytes(&mut password);

    if let Err(e) = res {
        eprintln!("ERROR: {}", e);
        process::exit(1);
    }
}
