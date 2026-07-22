#!/usr/bin/env bash
# Validate one Qwen3 Q8_0 row on the batched GPU prefill path:
#   1. confirm GPU-resident decode is active (else abort — a CPU number is meaningless)
#   2. batched-vs-serial A/B: greedy completions must be byte-identical (parity-green via
#      serial-prefill == llama.cpp transitivity from the committed bundles)
#   3. throughput median-of-5 on the batched build
# Usage: scripts/validate-cuda-prefill-row.sh <model.gguf> <label> <perf-out.json>
set -u
MODEL="$1"; LABEL="$2"; PERF_OUT="$3"
PORT=8185
EXE=./target/release/camelid.exe
TK="/c/WINDOWS/system32/taskkill.exe"

start_server() {
  local envset="$1" log="$2"
  $TK //IM camelid.exe //F >/dev/null 2>&1; sleep 1
  eval "$envset nohup $EXE serve --addr 127.0.0.1:$PORT --model \"$MODEL\" --no-open > $log 2>&1 &"
  for i in $(seq 1 180); do curl -s -m3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && return 0; sleep 1; done
  echo "server did not become ready"; return 1
}

echo "### $LABEL ###"
# ---- batched (default) ----
start_server "" /tmp/val_batched.log || exit 1
GPU=$(curl -s -m3 http://127.0.0.1:$PORT/health | node -e 'let d="";process.stdin.on("data",c=>d+=c).on("end",()=>{const h=JSON.parse(d);const ep=h.execution_plan||{};console.log(`backend=${ep.selected_backend} cuda_active=${ep.cuda_resident_active} prefill=${ep.prefill_path}`)})')
echo "  runtime: $GPU"
case "$GPU" in *cuda_active=true*) ;; *) echo "  !! GPU-resident NOT active — aborting row"; $TK //IM camelid.exe //F >/dev/null 2>&1; exit 2;; esac
node scripts/qwen3-cuda-prefill-ab.mjs --base http://127.0.0.1:$PORT > /tmp/val_ab_batched.json 2>/tmp/val_ab_err.txt || { echo "  AB(batched) FAILED"; cat /tmp/val_ab_err.txt; }
node scripts/bench-qwen3-cuda-resident.mjs --base http://127.0.0.1:$PORT --label "$LABEL (batched prefill)" --decode-tokens 128 --prefill-prompt-tokens 512 --runs 5 --out "$PERF_OUT" > /tmp/val_bench.json 2>/tmp/val_bench_err.txt || { echo "  bench FAILED"; cat /tmp/val_bench_err.txt; }

# ---- serial (A/B reference + baseline throughput) ----
start_server "CAMELID_CUDA_RESIDENT_PREFILL_BATCHED=0" /tmp/val_serial.log || exit 1
node scripts/qwen3-cuda-prefill-ab.mjs --base http://127.0.0.1:$PORT > /tmp/val_ab_serial.json 2>/tmp/val_ab_err.txt || { echo "  AB(serial) FAILED"; cat /tmp/val_ab_err.txt; }
node scripts/bench-qwen3-cuda-resident.mjs --base http://127.0.0.1:$PORT --label "$LABEL (serial prefill)" --decode-tokens 128 --prefill-prompt-tokens 512 --runs 5 > /tmp/val_bench_serial.json 2>/dev/null || echo "  serial bench FAILED"
$TK //IM camelid.exe //F >/dev/null 2>&1

echo "  --- parity (batched vs serial, empty = identical) ---"
if diff -q /tmp/val_ab_batched.json /tmp/val_ab_serial.json >/dev/null; then echo "  PARITY: IDENTICAL ✓"; else echo "  PARITY: DIVERGED ✗"; diff /tmp/val_ab_batched.json /tmp/val_ab_serial.json | head; fi
echo "  --- throughput (serial baseline -> batched) ---"
# Pipe the two JSON files via stdin (Windows node resolves /tmp to C:\tmp, so
# fs.readFileSync on the bash /tmp path fails — feed it through the shell instead).
{ cat /tmp/val_bench_serial.json; echo "@@@"; cat /tmp/val_bench.json; } | node -e '
let d="";process.stdin.on("data",c=>d+=c).on("end",()=>{
const p=d.split("\n@@@\n");
const s=JSON.parse(p[0]), b=JSON.parse(p[1]);
const sp=s.prefill.tok_s.median, bp=b.prefill.tok_s.median;
console.log(`  prefill ${sp} -> ${bp} tok/s (${(bp/sp).toFixed(2)}x)  |  decode ${s.decode.tok_s.median} -> ${b.decode.tok_s.median} tok/s`);
});' || echo "  (bench parse failed)"
echo "  --- sample completion (batched) ---"
node -e 'let d="";process.stdin.on("data",c=>d+=c).on("end",()=>{const o=JSON.parse(d);const k=Object.keys(o)[0];console.log(`  "${k.slice(0,30)}" -> ${JSON.stringify(o[k].slice(0,70))}`)})' < /tmp/val_ab_batched.json 2>/dev/null
