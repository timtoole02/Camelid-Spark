#!/usr/bin/env bash
# Clean interleaved A/B speed measurement for the coalesced-attention spike.
# Cools the GPU first (throttle invalidates back-to-back runs), then interleaves
# baseline (flag off) and candidate (CAMELID_ATTN_COALESCED=1) so both share the
# same thermal state each round -> the RATIO is valid even if clocks droop.
set -u
cd "$(git rev-parse --show-toplevel)"
MODEL="${MODEL:-../models/Qwen3-4B-Q8_0.gguf}"  # override with MODEL=/path/to.gguf
PROMPT=${PROMPT:-qa/speed/depth_prompt.txt}
ROUNDS=${ROUNDS:-5}
MAXTOK=${MAXTOK:-128}
COOL=${COOL:-55}
mkdir -p qa/speed/ab
rm -f qa/speed/ab/*.json

echo "=== cooldown: wait until GPU temp <= ${COOL}C (cap ~4min) ==="
for i in $(seq 1 24); do
  t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
  c=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits | head -1)
  echo "  t=${t}C clk=${c}MHz"
  if [ "${t:-99}" -le "$COOL" ]; then echo "  cool enough"; break; fi
  sleep 10
done

echo "=== interleaved A/B: ${ROUNDS} rounds, ${MAXTOK} tok, temp=0, depth=$(wc -c < "$PROMPT") chars ==="
for r in $(seq 1 "$ROUNDS"); do
  for phase in b c; do
    if [ "$phase" = c ]; then export CAMELID_ATTN_COALESCED=1; else unset CAMELID_ATTN_COALESCED; fi
    ./target/release/camelid.exe bench-generate "$MODEL" --prompt-file "$PROMPT" \
      --max-tokens "$MAXTOK" --temperature 0 --iterations 1 2>/dev/null > "qa/speed/ab/${phase}_${r}.json"
    ts=$(node -e 'const j=require("fs").readFileSync(process.argv[1],"utf8").trim().split("\n").filter(Boolean).map(JSON.parse);console.log(j[0].tokens_per_second.toFixed(2))' "qa/speed/ab/${phase}_${r}.json")
    clk=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits | head -1)
    tmp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    echo "  round $r $phase: ${ts} t/s @ ${clk}MHz ${tmp}C"
  done
done
unset CAMELID_ATTN_COALESCED

echo "=== SUMMARY ==="
node qa/speed/ab_summary.mjs qa/speed/ab
