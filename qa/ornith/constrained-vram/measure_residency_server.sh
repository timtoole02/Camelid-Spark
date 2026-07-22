#!/usr/bin/env bash
# Item 4 — consistent 16K residency measurement via llama-server (full decode state):
# load at -c 16384 -ngl 99, run one 64-token greedy completion, record peak
# nvidia-smi memory.used across load+decode, then kill. One quant per loop.
set -uo pipefail
BIN=${CAMELID_LLAMACPP_BIN:-$HOME/llama.cpp/build/bin}/llama-server.exe
MODELS=${CAMELID_MODELS_DIR:-models}
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/residency_16k_server.txt"
: > "$OUT"
for q in IQ3_XXS Q3_K_M IQ4_XS; do
  echo "== $q ==" | tee -a "$OUT"
  "$BIN" -m "$MODELS/ornith-1.0-9b-$q.gguf" -c 16384 -ngl 99 --port 8117 > "$HERE/res_srv_$q.log" 2>&1 &
  spid=$!
  peak=0
  ok=""
  for i in $(seq 1 180); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -gt "$peak" ] && peak=$used
    if [ -z "$ok" ] && curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8117/health 2>/dev/null | grep -q 200; then ok=1; break; fi
    sleep 2
  done
  if [ -z "$ok" ]; then echo "LOAD FAILED (never healthy)" | tee -a "$OUT"; kill $spid 2>/dev/null; continue; fi
  # 64-token greedy completion; sample VRAM concurrently.
  curl -s http://127.0.0.1:8117/completion -H 'content-type: application/json' \
    -d '{"prompt":"Explain what a mutex is in one paragraph.","n_predict":64,"temperature":0}' > "$HERE/res_srv_${q}_gen.json" &
  cpid=$!
  while kill -0 $cpid 2>/dev/null; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -gt "$peak" ] && peak=$used
    sleep 1
  done
  gen_len=$(grep -oE '"tokens_predicted": *[0-9]+' "$HERE/res_srv_${q}_gen.json" | grep -oE '[0-9]+' || echo 0)
  taskkill //F //PID $spid >/dev/null 2>&1 || kill -9 $spid 2>/dev/null
  # wait for VRAM release before the next quant
  for i in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 500 ] && break
    sleep 2
  done
  echo "peak_vram_mib=$peak headroom_vs_6144=$((6144 - peak)) tokens_generated=$gen_len" | tee -a "$OUT"
done
echo done | tee -a "$OUT"
