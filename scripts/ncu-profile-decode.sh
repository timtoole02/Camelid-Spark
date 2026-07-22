#!/usr/bin/env bash
# Targeted Nsight Compute profile of the resident decode GEMV (q8_gemv).
# Single-pass Speed-of-Light + occupancy metrics, small launch count past warmup.
set -u
NCU="/c/Program Files/NVIDIA Corporation/Nsight Compute 2025.2.1/target/windows-desktop-win7-x64/ncu.exe"
PORT=8186
OUT=/tmp/ncu_decode.csv
METRICS="gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,gpu__time_duration.sum,launch__occupancy_limit_blocks"

LONG=$(printf 'The quick brown fox jumps over the lazy dog near the riverbank. %.0s' {1..40})

rm -f "$OUT"
"$NCU" --target-processes all -k "regex:q8_gemv" -c 14 -s 300 \
  --metrics "$METRICS" --csv --log-file "$OUT" \
  ./target/release/camelid.exe serve --addr 127.0.0.1:$PORT --model models/Qwen3-0.6B-Q8_0.gguf --no-open \
  > /tmp/ncu_decode_run.log 2>&1 &
NCUPID=$!

# Wait for health (server is slow under ncu instrumentation).
for i in $(seq 1 180); do
  if curl -s -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo "ready after ${i}s"; break; fi
  sleep 1
done

# Fire one prefill-heavy request: hundreds of batch-1 q8_gemv launches.
echo "firing request..."
curl -s -m 600 -X POST "http://127.0.0.1:$PORT/v1/completions" \
  -H 'content-type: application/json' \
  -d "{\"prompt\":$(node -e "console.log(JSON.stringify(process.argv[1]))" "$LONG"),\"max_tokens\":2,\"temperature\":0,\"top_k\":1}" \
  >/tmp/ncu_decode_req.json 2>&1
echo "request done (rc=$?)"

# Give ncu a moment to finish capturing the -c launches, then stop the server child.
sleep 2
"/c/WINDOWS/system32/taskkill.exe" //IM camelid.exe //F >/dev/null 2>&1
wait $NCUPID 2>/dev/null
echo "=== ncu csv ==="
cat "$OUT" 2>/dev/null
