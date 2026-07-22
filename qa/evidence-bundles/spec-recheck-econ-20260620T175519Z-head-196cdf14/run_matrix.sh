#!/usr/bin/env bash
# SPEC_RECHECK Phase 1 driver. Usage: run_matrix.sh <kind>
#   kind = ngram | draft-gpu | draft-cpu
# Loops 6 workloads x gamma{2,4,6,7}, appends one JSON line per cell to results/<kind>.jsonl.
set -u
cd "$(dirname "$0")"
KIND="$1"
GAMMAS="${2:-2 4 6 7}"
EXE="$HOME/Camelid/target/release/camelid.exe"
TARGET="$HOME/models/Qwen3-4B-Q8_0.gguf"
DRAFT="$HOME/camelid-dltest/models/Qwen3-0.6B-Q8_0.gguf"
export CAMELID_COMMIT="$(git rev-parse --short HEAD)"

case "$KIND" in
  ngram)     FLAGS=(--drafter ngram) ;;
  draft-gpu) FLAGS=(--drafter draft --draft-model "$DRAFT") ;;
  draft-cpu) FLAGS=(--drafter draft --draft-model "$DRAFT" --cpu-draft) ;;
  *) echo "unknown kind $KIND"; exit 2 ;;
esac

OUT="results/${KIND}.jsonl"
: > "$OUT"
for W in code json extraction chat creative adversarial; do
  for G in $GAMMAS; do
    echo ">>> $KIND $W gamma=$G"
    LINE=$("$EXE" bench-speculative "$TARGET" "${FLAGS[@]}" \
        --draft-tokens "$G" --workload "$W" --max-tokens 128 --warmup \
        --prompt-file "prompts/${W}.txt" 2>>"results/${KIND}.stderr.log" | grep '^{')
    if [ -n "$LINE" ]; then
      echo "$LINE" >> "$OUT"
    else
      echo "!!! no JSON for $KIND $W gamma=$G (see results/${KIND}.stderr.log)"
    fi
  done
done
echo "=== done $KIND: $(wc -l < "$OUT") cells -> $OUT ==="
