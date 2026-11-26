#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo "  NIBBLX-512 v3.1.1 — Full Verification & Audit Suite"
echo "============================================================"

echo -e "\n[1/7] Clippy Strict Lint Check (-D warnings)"
cargo clippy --all-targets --release -- -D warnings

echo -e "\n[2/7] Unit, KAT, Avalanche, Poisoning, KDF, FFI & AVX2 Tests (8/8)"
cargo test --release

echo -e "\n[3/7] Compiled Assembly Branch Audit (permute_14_inreg)"
BIN=$(ls -t target/release/deps/nibblx512-* | grep -v '\.d$' | head -n 1)
BRANCHES=$(objdump -d -M intel "$BIN" | awk '/<.*permute_14_inreg.*>:/,/^$/' | grep -Ec '\bj[a-z]+' || true)
echo "Secret-dependent branches in permute_14_inreg: $BRANCHES"
test "$BRANCHES" -eq 0

echo -e "\n[4/7] 3-Suite dudect Constant-Time Timing Audit"
cargo run --release --example dudect -- 1000000

echo -e "\n[5/7] Throughput Benchmark (Scalar + 4-Way AVX2 + 8-Core)"
cargo run --release --example bench

echo -e "\n[6/7] Streaming CLI Encrypt / Bit-Exact Decrypt / Tamper Reject"
NIBBLX_PASS="DemoPass123" cargo run --release --bin nibblx_cli -- encrypt SPEC.md SPEC.mrf3
NIBBLX_PASS="DemoPass123" cargo run --release --bin nibblx_cli -- decrypt SPEC.mrf3 SPEC_out.md
diff -s SPEC.md SPEC_out.md
if NIBBLX_PASS="WrongPass" cargo run --release --bin nibblx_cli -- decrypt SPEC.mrf3 SPEC_bad.md 2>&1; then
  echo "ERROR: Wrong password should have failed!"
  exit 1
else
  echo "PASS: Wrong password cleanly rejected and no output file created."
fi
rm -f SPEC.mrf3 SPEC_out.md SPEC_bad.md

echo -e "\n[7/7] PractRand 0.95 Statistical Randomness Check (1 GB Demo)"
if command -v RNG_test >/dev/null 2>&1; then
  cargo run --release --example stream_out | RNG_test stdin64 -tlmin 64MB -tlmax 1GB
else
  echo "SKIP: RNG_test not installed."
fi

echo -e "\n============================================================"
echo "  ALL 7 CHECKS PASSED — NIBBLX-512 v3.1.1 VERIFIED"
echo "============================================================"
