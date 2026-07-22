#!/usr/bin/env bash
# Item 4 — measure real VRAM at -c 16384 -ngl 99 for each candidate quant
# (llama.cpp hybrid-arch KV allocates only the 8 full-attention layers, comparable
# to Camelid's sparse-KV design). Records peak nvidia-smi memory.used while a
# short greedy generation runs, plus load success.
set -uo pipefail
BIN=${CAMELID_LLAMACPP_BIN:-$HOME/llama.cpp/build/bin}
MODELS=${CAMELID_MODELS_DIR:-models}
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/residency_16k_measurements.txt"
: > "$OUT"
for q in IQ3_XXS Q3_K_M IQ4_XS; do
  echo "== $q ==" | tee -a "$OUT"
  log="$HERE/res16k_$q.log"
  "$BIN/llama-cli.exe" -m "$MODELS/ornith-1.0-9b-$q.gguf" -c 16384 -ngl 99 \
    -p "What is the capital of France?" -n 8 --temp 0 --seed 0 > "$log" 2>&1 &
  pid=$!
  peak=0
  while kill -0 $pid 2>/dev/null; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -gt "$peak" ] && peak=$used
    sleep 1
  done
  wait $pid; rc=$?
  free_at_peak=$((6144 - peak))
  ok=$(grep -c 'Paris\|capital' "$log" || true)
  echo "peak_vram_mib=$peak headroom_vs_6144=$free_at_peak exit=$rc coherent_hits=$ok" | tee -a "$OUT"
  grep -iE 'out of memory|failed to allocate|error' "$log" | head -2 | tee -a "$OUT"
done
echo done
