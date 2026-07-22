#!/usr/bin/env bash
# FLINT Spark receipt packet — one session, all the receipts the Windows side needs.
# Run from the repo root on the DGX Spark after `cargo build --release`:
#
#   bash scripts/spark-receipt-packet.sh
#
# Model paths default to ./models/<file>; override via env:
#   B70=/path/Llama-3.3-70B-Instruct-Q8_0.gguf   (pull id: llama33_70b_instruct_q8_0)
#   B14=/path/Qwen3-14B-Q8_0.gguf                (pull id: qwen3_14b_q8_0)
#   B1=/path/Llama-3.2-1B-Instruct-Q8_0.gguf     (pull id: llama32_1b_instruct_q8_0)
#
# Legs skip (loudly) when their model file is absent — partial packets are still
# useful. Every leg writes stdout+stderr into the bundle dir; nothing is trimmed.
# Expected wall time with all models present: ~30-60 min (70B legs dominate).
set -u
BIN=${BIN:-./target/release/camelid}
B70=${B70:-./models/Llama-3.3-70B-Instruct-Q8_0.gguf}
B14=${B14:-./models/Qwen3-14B-Q8_0.gguf}
B1=${B1:-./models/Llama-3.2-1B-Instruct-Q8_0.gguf}
OUT="qa/evidence-bundles/spark-packet-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
LP=$(printf 'The lighthouse keeper walked along the shore each morning collecting driftwood and counting the gulls. %.0s' $(seq 1 45))
SP="Write a numbered list of the first ten squares: 1: 1, 2: 4, 3: 9,"

note() { echo "== $*" | tee -a "$OUT/PACKET-LOG.txt"; }
run()  { local name=$1; shift; note "LEG $name: $*"; "$@" >"$OUT/$name.out" 2>"$OUT/$name.err"; echo "   exit=$? (see $name.out/.err)" | tee -a "$OUT/PACKET-LOG.txt"; }
have() { [ -f "$1" ] || { note "SKIP ($2): model not found at $1"; return 1; }; }

note "packet start $(date -u); git $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
note "env sanity (should be empty):"; env | grep -E '^CAMELID' | tee -a "$OUT/PACKET-LOG.txt" || true

# Leg 0 — hardware banner (expect: UNIFIED memory, and note the driver/CUDA versions).
run leg0-banner "$BIN" plan-offload --arch llama-8b --budget-mb 64

if have "$B70" "70B legs"; then
  # Leg 1 — THE headline: 70B under the default (hostreg register on unified).
  # Wanted lines in leg1's .err: "[cuda] hostreg attrs: ..." and
  # "[cuda] hostreg mode=register: 561 zero-copy / 0 uploaded" (80x7+head; send
  # the line even if M>0 — per-tensor fallback engaging is itself a finding).
  run leg1-70b-register "$BIN" bench-generate "$B70" --prompt "$SP" --max-tokens 64
  # Leg 2 — mapped-vs-device A/B: same run under =upload. Tokens must be
  # IDENTICAL to leg 1; the tok/s ratio IS the mapped-read penalty P.
  CAMELID_CUDA_HOSTREG=upload run leg2-70b-upload "$BIN" bench-generate "$B70" --prompt "$SP" --max-tokens 64
  # Leg 3 — PREFILL_K sweep on a ~1k-token prompt: prefill_ms at K=8 vs K=16
  # (tokens must be identical; expect ~2x TTFT drop if weight reads dominate).
  CAMELID_CUDA_PREFILL_K=8  run leg3-70b-k8  "$BIN" bench-generate "$B70" --prompt "$LP" --max-tokens 16
  CAMELID_CUDA_PREFILL_K=16 run leg3-70b-k16 "$BIN" bench-generate "$B70" --prompt "$LP" --max-tokens 16
  # Leg 4 — lossless n-gram spec on the 70B (structured workload): accept rate,
  # S_sync, and the harness's own LOSSLESS byte-compare verdict.
  run leg4-70b-ngram "$BIN" bench-speculative "$B70" --drafter ngram --prompt "$SP" --max-tokens 96
  # Leg 5 — 1B-draft speculation against the 70B (the headline decode candidate;
  # 1:70 cost ratio). Wanted: accept rate, tok/round, plain-vs-spec tok/s, LOSSLESS.
  if have "$B1" "1B-draft leg"; then
    run leg5-70b-draft1b "$BIN" bench-speculative "$B70" --drafter draft --draft-model "$B1" --prompt "$LP" --max-tokens 96
  fi
  # Leg 6 — historical A/B on the 70B: =0 lands on the CPU lane (expected, slow
  # — the pre-hostreg posture); short leg, tokens must still match leg 1's head.
  CAMELID_CUDA_HOSTREG=0 run leg6-70b-off "$BIN" bench-generate "$B70" --prompt "$SP" --max-tokens 16
fi

# Leg 7 — 14B upload-path A/B (small enough that =0 genuinely takes the
# historical VRAM-upload path): three-way tokens must be identical.
if have "$B14" "14B legs"; then
  CAMELID_CUDA_HOSTREG=0      run leg7-14b-off      "$BIN" bench-generate "$B14" --prompt "$SP" --max-tokens 32
  CAMELID_CUDA_HOSTREG=1      run leg7-14b-register "$BIN" bench-generate "$B14" --prompt "$SP" --max-tokens 32
  CAMELID_CUDA_HOSTREG=upload run leg7-14b-upload   "$BIN" bench-generate "$B14" --prompt "$SP" --max-tokens 32
fi

# Leg 8 — suffix prefill session A/B on the 1B (cargo test; token-identical
# turns + pure-hit/extend/divergent engagement traces in the output).
if have "$B1" "suffix leg"; then
  note "LEG leg8-suffix: cargo test --release --test suffix_prefill"
  CAMELID_SUFFIX_PREFILL_GGUF="$B1" CAMELID_RESIDENT_TRACE=1 \
    cargo test --release --test suffix_prefill -- --nocapture \
    >"$OUT/leg8-suffix.out" 2>"$OUT/leg8-suffix.err"
  echo "   exit=$?" | tee -a "$OUT/PACKET-LOG.txt"
fi

# Leg 9 — the outstanding gemma4-E4B NVFP4 receipt (FLINT_HANDOFF Part B) is
# manual (requires the locally-quantized NVFP4 gguf); run it per Part B and drop
# the output into this bundle as leg9-nvfp4.txt if the file is on the box.

note "packet done $(date -u). Send the whole $OUT directory back."
