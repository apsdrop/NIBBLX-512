//! Raw 4-Way SIMD ciphertext stream generator for PractRand / Dieharder statistical testing.
//!
//! Encrypts all-zero 64 KB chunks using 4-Way SIMD (`EnterpriseCmt256`) and writes raw
//! ciphertext continuously to `stdout`, exiting cleanly on `BrokenPipe`.
//!
//! Place at `examples/stream_out.rs` and run:
//!   cargo run --release --example stream_out | RNG_test stdin64

use std::io::{self, BufWriter, ErrorKind, Write};

use nibblx512::{Profile, StreamEncryptor, CHUNK_SIZE, KEY_LEN, NONCE_LEN};

fn main() {
    let key = [0x42u8; KEY_LEN];
    let nonce = [0x24u8; NONCE_LEN];
    let ad = b"MORPHIC-V3-PRACTRAND-STREAM";

    let mut enc = StreamEncryptor::new(&key, &nonce, ad, Profile::EnterpriseCmt256);

    let mut b0 = vec![0u8; CHUNK_SIZE];
    let mut b1 = vec![0u8; CHUNK_SIZE];
    let mut b2 = vec![0u8; CHUNK_SIZE];
    let mut b3 = vec![0u8; CHUNK_SIZE];

    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(CHUNK_SIZE * 4, stdout.lock());

    loop {
        b0.fill(0);
        b1.fill(0);
        b2.fill(0);
        b3.fill(0);

        if enc
            .encrypt_4_chunks([&mut b0, &mut b1, &mut b2, &mut b3], false)
            .is_err()
        {
            break;
        }

        for chunk in [&b0, &b1, &b2, &b3] {
            if let Err(e) = out.write_all(chunk) {
                if e.kind() == ErrorKind::BrokenPipe {
                    return;
                }
                eprintln!("stdout write error: {}", e);
                std::process::exit(1);
            }
        }
    }
}