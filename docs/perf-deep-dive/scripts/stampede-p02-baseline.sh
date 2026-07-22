#!/usr/bin/env bash
# STAMPEDE P0.2 — same-host CPU baseline matrix vs the b9918 pin.
# Both engines CPU-true: CUDA_VISIBLE_DEVICES=-1 for the WHOLE run (llama b9918 zip is
# CPU-only by construction; camelid additionally gets resident-decode/prefill=0 from the harness).
# Usage: bash stampede-p02-baseline.sh <out-dir-suffix>
set -uo pipefail
cd "$(dirname "$0")/../../.."   # repo root

export CAMELID_BIN="$PWD/target/release/camelid.exe"
export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$HOME/tools/llama-cpp-b9918/llama-server.exe}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
export LLAMA_PIN="b9918-0512ef1e5"
export CAMELID_HEAD="$(git rev-parse --short HEAD)"
export CUDA_VISIBLE_DEVICES=-1
export REPEATS="${REPEATS:-5}"

STAMP="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTDIR="docs/perf-deep-dive/PERF_RECEIPTS/same-host/stampede-p0-baseline-${CAMELID_HEAD}-${STAMP}"
mkdir -p "$OUTDIR"

run_row () { # name model_path
  local name="$1" model="$2"
  echo "=== ROW $name ($model) ===" >&2
  MODEL_GGUF="$model" MODEL_ID="$name" OUT_JSON="$OUTDIR/$name.json" \
    node docs/perf-deep-dive/scripts/cpu-baseline-medN.mjs 2>"$OUTDIR/$name.stderr.log"
  echo "--- $name done (exit $?) ---" >&2
}

# primary
run_row llama3b-q8    "$MODELS_DIR/Llama-3.2-3B-Instruct-Q8_0.gguf"
# secondary
run_row qwen3-4b-q8   "$MODELS_DIR/Qwen3-4B-Q8_0.gguf"
run_row qwen3-06b-q8  "$MODELS_DIR/Qwen3-0.6B-Q8_0.gguf"
# K-quant rows (KQUANT conductor receipts exist)
run_row llama3b-q4km  "$MODELS_DIR/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
run_row qwen3-4b-q4km "$MODELS_DIR/Qwen3-4B-Q4_K_M.gguf"

echo "ALL ROWS DONE -> $OUTDIR" >&2
