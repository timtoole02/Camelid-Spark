#!/usr/bin/env bash
# qa/speed/verify-gates.sh — ENFORCED PROOF LANE for the WIN2METAL byte-exact spec-verify gates.
#
# WHY THIS EXISTS (review M2/G1): the three bit-identity gates
#   metal_verify_gemv_batched_bit_identical   (Phase 3 C0 — the batched GEMV)
#   metal_spec_verify_bit_identical           (Phase 3 — linear verify_batch)
#   metal_tree_verify_bit_identical           (Phase 4 — tree verify_batch_tree)
# guard on process-wide OnceLock env gates (f32y / wire / nsg8 / attn2 / split-K). Under a shared
# `cargo test --all-targets` run, a sibling test can read (and latch) those gates OFF *before* the
# gate test runs, so the gate takes its SKIP branch — and a skipped #[test] counts as PASS. So a
# green `--all-targets` does NOT, by itself, exercise the byte-exact assertions.
#
# This script runs each gate in ITS OWN cargo process (fresh OnceLocks), where the gate sets the
# env and latches the gates ON before any sibling can — so the to_bits assertions actually run.
# THIS is the byte-exactness proof of record; CI should run this, not rely on --all-targets.
#
# Usage:  qa/speed/verify-gates.sh
set -uo pipefail
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/Volumes/Untitled/camelid-target}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

GATES=(
  metal_verify_gemv_batched_bit_identical
  metal_spec_verify_bit_identical
  metal_tree_verify_bit_identical
)

rc=0
for g in "${GATES[@]}"; do
  echo "=== $g (isolated process) ==="
  log="$(mktemp "${TMPDIR:-/tmp}/verify_gate.XXXXXX")"
  if ! cargo test --release --lib "$g" -- --nocapture >"$log" 2>&1; then
    echo "  FAIL: $g returned non-zero"; tail -20 "$log"; rc=1; rm -f "$log"; continue
  fi
  # A SKIP in an isolated process means the gates couldn't latch on even with no sibling — a real
  # problem (Metal unavailable, or a gate dependency changed), NOT the benign --all-targets skip.
  if grep -q "SKIP $g" "$log"; then
    echo "  FAIL: $g SKIPPED in isolation (Metal device unavailable or gates won't latch) — investigate"
    grep "SKIP $g" "$log" | sed 's/^/    /'; rc=1; rm -f "$log"; continue
  fi
  if ! grep -q "BIT-IDENTICAL" "$log"; then
    echo "  FAIL: $g produced no BIT-IDENTICAL line — did the assertions run?"; tail -20 "$log"; rc=1; rm -f "$log"; continue
  fi
  grep -E "BIT-IDENTICAL|PASS|test result:" "$log" | sed 's/^/  /'
  rm -f "$log"
done

echo
if [ "$rc" -eq 0 ]; then
  echo "PASS: all 3 byte-exact verify gates ran ENGAGED (split-K straddle 126 & 510) and are BIT-IDENTICAL."
else
  echo "FAIL: a byte-exact verify gate did not run/pass engaged — see above (this is the real proof, not --all-targets)."
fi
exit "$rc"
