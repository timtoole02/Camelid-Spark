#!/usr/bin/env bash
# SPEC_RECHECK Phase 3: run llama.cpp llama-speculative on the SAME prompt files used for the
# Camelid matrix, greedy/lossless, both models GPU-resident (-ngl 99 -ngld 99), pinned acd79d6.
# 3 reps/workload, median decode t/s. Output: results/phase3-llama-spec.jsonl
set -u
cd "$(dirname "$0")"
EXE="$HOME/llama.cpp/build/bin/llama-speculative.exe"
TARGET="$HOME/models/Qwen3-4B-Q8_0.gguf"
DRAFT="$HOME/camelid-dltest/models/Qwen3-0.6B-Q8_0.gguf"
OUT="results/phase3-llama-spec.jsonl"
: > "$OUT"
median() { printf '%s\n' "$@" | sort -n | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }

for W in code json extraction chat creative adversarial; do
  speeds=(); accs=()
  for R in 1 2 3; do
    LOG=$("$EXE" -m "$TARGET" -md "$DRAFT" -f "prompts/${W}.txt" -n 128 \
        --spec-draft-n-max 8 --spec-draft-n-min 1 -ngl 99 -ngld 99 --top-k 1 --temp 0 -c 2048 2>&1)
    SP=$(printf '%s\n' "$LOG" | grep -oE 'decoded[ ]+[0-9]+ tokens in[ ]+[0-9.]+ seconds, speed:[ ]+[0-9.]+' | grep -oE '[0-9.]+$' | tail -1)
    AC=$(printf '%s\n' "$LOG" | grep -oE 'accept[ ]+=[ ]+[0-9.]+' | grep -oE '[0-9.]+$' | tail -1)
    [ -n "$SP" ] && speeds+=("$SP")
    [ -n "$AC" ] && accs+=("$AC")
    echo "  $W rep$R: ${SP:-FAIL} t/s | accept ${AC:-?}%"
  done
  MS=$(median "${speeds[@]}")
  MA=$(median "${accs[@]}")
  echo "{\"engine\":\"llama.cpp\",\"commit\":\"acd79d6\",\"workload\":\"$W\",\"decode_tps_median\":$MS,\"accept_pct_median\":$MA,\"reps\":${#speeds[@]},\"n_gen\":128,\"draft_n_max\":8,\"ngl\":99,\"ngld\":99}" >> "$OUT"
  echo ">>> $W: median ${MS} t/s | accept ${MA}%"
done
echo "=== done: $(wc -l < "$OUT") workloads -> $OUT ==="
