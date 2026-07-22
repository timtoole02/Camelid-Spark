#!/usr/bin/env bash
# STAMPEDE Phase 6 — Gate 5 measurement matrix (PHASE6_CUDA_STREAMS_PLAN.md §7).
#
# OFF vs ON × {low ctx, depth ~1881} × {Llama-3.2-3B Q8_0, Qwen3-4B Q8_0}, one
# fresh single-engine server per leg (bench-memory-safety rules: free-RAM gate,
# PID-only kills). OFF baselines are minted fresh in the same session — the
# on-file numbers predate the b9918 re-pin era and thermal state matters on this
# laptop. OFF/ON run back-to-back within each (model, ctx) cell to minimize
# thermal drift. The ON legs are engaged-checked (Phase 3 fake-null precedent):
# a missing "overlap ENGAGED" trace line voids the receipt and fails the run.
#
# Gate (STAMPEDE_CONDUCTOR.md:144): ON >= +8% decode at low ctx on BOTH models
# AND no depth regression -> GO.
#
# Usage: scripts/bench-cuda-streams-matrix.sh [receipts-dir] [date-tag]
#   e.g. scripts/bench-cuda-streams-matrix.sh qa/perf 20260709
set -u
OUTDIR="${1:-qa/perf}"
DATE="${2:-manual}"
PORT=8190
EXE=./target/release/camelid.exe
TK="/c/WINDOWS/system32/taskkill.exe"
# Depth targets are per model: buildLongPrompt overshoots ~10% in server tokens
# (measured 1881 -> 2068), and the depth prompt + 128 decode tokens must stay
# inside the model's VRAM-resident KV cap or the tail decodes on CPU fallback
# and the receipt measures the wrong thing (first 4B depth run: 2068+128 > the
# 2090-pos 4B Q8_0 cap -> 1.5 tok/s). 3B's cap is far higher; 4B gets 1550
# (~1700 server tokens, +128 decode ≈ 1830 < 2090 with margin).
DEPTH_TOKENS_3B=1881
DEPTH_TOKENS_4B=1550
DECODE_TOKENS=128
RUNS=5
mkdir -p "$OUTDIR"
FAILED=0
SERVER_PID=""
SERVER_WINPID=""

find_model() {
  if [ -f "$HOME/models/$1" ]; then echo "$HOME/models/$1"; elif [ -f "./models/$1" ]; then echo "./models/$1"; fi
}

free_ram_kb() {
  powershell -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory" | tr -d '\r'
}

require_ram() {
  local sz_kb need_kb free_kb
  sz_kb=$(( $(stat -c%s "$1") / 1024 ))
  need_kb=$(( sz_kb + 3 * 1024 * 1024 ))
  free_kb=$(free_ram_kb)
  if [ -z "$free_kb" ] || [ "$free_kb" -lt "$need_kb" ]; then
    echo "  !! free RAM ${free_kb:-unknown} KiB < required ${need_kb} KiB — leg blocked"
    return 1
  fi
}

start_server() { # envset model log
  local envset="$1" model="$2" log="$3"
  if curl -s -m2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "  !! port $PORT already serving — refusing"; return 1
  fi
  eval "$envset nohup $EXE serve --addr 127.0.0.1:$PORT --model \"$model\" --no-open > \"$log\" 2>&1 &"
  SERVER_PID=$!
  SERVER_WINPID=$(ps -p "$SERVER_PID" -W 2>/dev/null | awk 'NR==2{print $4}')
  local i
  for i in $(seq 1 240); do
    curl -s -m3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "  !! server exited during startup:"; tail -5 "$log"; SERVER_PID=""; return 1; }
    sleep 1
  done
  echo "  !! server not ready after 240s"; stop_server; return 1
}

stop_server() {
  [ -n "$SERVER_PID" ] || return 0
  kill "$SERVER_PID" 2>/dev/null
  local i
  for i in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    [ -n "$SERVER_WINPID" ] && "$TK" //PID "$SERVER_WINPID" //F >/dev/null 2>&1
    sleep 2
  fi
  for i in $(seq 1 15); do curl -s -m2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || break; sleep 1; done
  SERVER_PID=""; SERVER_WINPID=""
}
trap stop_server EXIT

bench_leg() { # state(off|on) model-basename model-key ctx-key decode-prompt-tokens
  local state="$1" model_file="$2" mkey="$3" ckey="$4" dpt="$5"
  local model envset="CAMELID_RESIDENT_TRACE=1" log receipt
  model=$(find_model "$model_file")
  [ -n "$model" ] || { echo "!! model $model_file not found"; FAILED=1; return; }
  [ "$state" = "on" ] && envset="$envset CAMELID_CUDA_STREAMS=1"
  log="$OUTDIR/stampede-p6-cuda-streams-$state-$mkey-$ckey-$DATE.server.log"
  receipt="$OUTDIR/stampede-p6-cuda-streams-$state-$mkey-$ckey-$DATE.json"
  echo "--- $state $mkey $ckey (decode ctx ~$dpt tok) ---"
  require_ram "$model" || { FAILED=1; return; }
  start_server "$envset" "$model" "$log" || { FAILED=1; return; }
  local dpt_args=""
  [ "$dpt" != "0" ] && dpt_args="--decode-prompt-tokens $dpt"
  node scripts/bench-qwen3-cuda-resident.mjs --base "http://127.0.0.1:$PORT" \
    --label "P6 cuda-streams $state $mkey $ckey" \
    --decode-tokens $DECODE_TOKENS --prefill-prompt-tokens 512 --runs $RUNS \
    $dpt_args --out "$receipt" >/dev/null 2>"$OUTDIR/$state-$mkey-$ckey.bench.err" \
    || { echo "  !! bench failed:"; tail -3 "$OUTDIR/$state-$mkey-$ckey.bench.err"; FAILED=1; stop_server; return; }
  stop_server
  if [ "$state" = "on" ]; then
    if ! grep -q "overlap ENGAGED" "$log"; then
      echo "  !! ENGAGED trace MISSING — ON receipt VOID"; FAILED=1; return
    fi
    echo "  engaged: confirmed"
  else
    grep -q "\[cuda-streams\] off" "$log" && echo "  off-trace: confirmed"
  fi
  node -e 'const r=JSON.parse(require("fs").readFileSync(process.argv[1]));console.log(`  decode ${r.decode.tok_s.median} tok/s (min ${r.decode.tok_s.min} max ${r.decode.tok_s.max} sd ${r.decode.tok_s.stddev}), ctx=${r.decode.context_prompt_tokens??0}`)' "$receipt"
}

# Clean-box preconditions (learned the hard way — a leaked GPU-resident server
# from another session once dragged 3B decode from ~48 to 6.6 tok/s and voided
# a whole matrix): refuse to bench if any other process holds the GPU or a
# cargo/rustc build is running. Known non-compute residents (e.g. a game
# launcher that registers a context but does no work) can be waived via
# CAMELID_BENCH_GPU_ALLOWLIST (grep -E pattern).
ALLOW="${CAMELID_BENCH_GPU_ALLOWLIST:-EpicGamesLauncher}"
GPU_OTHERS=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -Ev "$ALLOW" | grep -c .)
if [ "$GPU_OTHERS" != "0" ]; then
  echo "!! GPU not clean — refusing to bench:"
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
  exit 2
fi
if [ "$(ps -W 2>/dev/null | grep -ci 'cargo\|rustc')" != "0" ]; then
  echo "!! cargo/rustc running — refusing to bench under CPU contention"
  exit 2
fi

# Per (model, ctx) cell: OFF then ON back-to-back (thermal locality).
for cell in \
  "Llama-3.2-3B-Instruct-Q8_0.gguf llama3b-q8 lowctx 0" \
  "Llama-3.2-3B-Instruct-Q8_0.gguf llama3b-q8 depth $DEPTH_TOKENS_3B" \
  "Qwen3-4B-Q8_0.gguf qwen3-4b-q8 lowctx 0" \
  "Qwen3-4B-Q8_0.gguf qwen3-4b-q8 depth $DEPTH_TOKENS_4B"; do
  set -- $cell
  bench_leg off "$1" "$2" "$3" "$4"
  bench_leg on "$1" "$2" "$3" "$4"
done

echo
echo "=== Gate 5 summary (ON vs fresh OFF, median decode tok/s) ==="
node - "$OUTDIR" "$DATE" <<'EOF'
const fs = require('fs')
const [dir, date] = process.argv.slice(2)
let fail = false
for (const mkey of ['llama3b-q8', 'qwen3-4b-q8']) {
  for (const ckey of ['lowctx', 'depth']) {
    try {
      const off = JSON.parse(fs.readFileSync(`${dir}/stampede-p6-cuda-streams-off-${mkey}-${ckey}-${date}.json`))
      const on = JSON.parse(fs.readFileSync(`${dir}/stampede-p6-cuda-streams-on-${mkey}-${ckey}-${date}.json`))
      const o = off.decode.tok_s.median, n = on.decode.tok_s.median
      const delta = ((n / o - 1) * 100).toFixed(1)
      console.log(`${mkey} ${ckey}: ${o} -> ${n} tok/s (${delta >= 0 ? '+' : ''}${delta}%)`)
    } catch (e) { console.log(`${mkey} ${ckey}: MISSING (${e.message.split('\n')[0]})`); fail = true }
  }
}
process.exit(fail ? 1 : 0)
EOF
[ $? -eq 0 ] || FAILED=1
[ "$FAILED" -eq 0 ] && echo "MATRIX COMPLETE" || { echo "MATRIX HAD FAILURES"; exit 1; }
