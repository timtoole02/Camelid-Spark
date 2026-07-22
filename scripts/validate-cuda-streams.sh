#!/usr/bin/env bash
# STAMPEDE Phase 6 — CAMELID_CUDA_STREAMS OFF-vs-ON byte-identical validation
# (PHASE6_CUDA_STREAMS_PLAN.md §6). Cloned from validate-cuda-prefill-row.sh's
# env-flag A/B pattern, hardened to the bench-memory-safety rules: one engine
# resident at a time, free-RAM check (model + 3 GiB) before each leg, servers
# killed by PID only (never by image name — the desktop app runs a camelid.exe
# sidecar that a blanket taskkill would murder).
#
# Legs (each = fresh OFF server -> greedy corpus -> kill, fresh ON server ->
# same corpus -> kill, byte diff):
#   1  Llama-3.2-3B Q8_0          low ctx + ~1881-token depth prompt (gate model)
#   2  Qwen3-4B Q8_0              QK-norm on side_a; depth 1600 (2090-pos resident cap)
#   3  Qwen3-4B Q4_K_M            K-quant gemv lanes + Q8_K activation scratch
#   4  Llama-3.2-3B Q8_0 unfused  CAMELID_RESIDENT_NO_FUSION=1 chain
#   5  ornith-9b Q4_K_M (qwen35)  device-side decode loop (forward_token_device)
#
# Engaged-check (Phase 3 fake-null precedent): an ON leg without the
# "[cuda-streams] overlap ENGAGED" trace line in the server log is VOID, no
# matter what the diff says. OFF legs must show "[cuda-streams] off".
#
# Usage: scripts/validate-cuda-streams.sh [outdir] [leg ...]
#   e.g. scripts/validate-cuda-streams.sh /tmp/p6-val 1 4
#   default: all five legs into /tmp/cuda-streams-val
set -u
OUTDIR="${1:-/tmp/cuda-streams-val}"
shift 2>/dev/null || true
LEGS="${*:-1 2 3 4 5}"
PORT=8189
EXE=./target/release/camelid.exe
TK="/c/WINDOWS/system32/taskkill.exe"
MODELS_A="$HOME/models"
MODELS_B="./models"
mkdir -p "$OUTDIR"
FAILED=0
SERVER_PID=""
SERVER_WINPID=""

find_model() { # basename -> path (or empty)
  if [ -f "$MODELS_A/$1" ]; then echo "$MODELS_A/$1"; elif [ -f "$MODELS_B/$1" ]; then echo "$MODELS_B/$1"; fi
}

free_ram_kb() {
  powershell -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory" | tr -d '\r'
}

require_ram() { # model-path; needs model size + 3 GiB free
  local sz_kb need_kb free_kb
  sz_kb=$(( $(stat -c%s "$1") / 1024 ))
  need_kb=$(( sz_kb + 3 * 1024 * 1024 ))
  free_kb=$(free_ram_kb)
  if [ -z "$free_kb" ] || [ "$free_kb" -lt "$need_kb" ]; then
    echo "  !! free RAM ${free_kb:-unknown} KiB < required ${need_kb} KiB (model + 3 GiB) — leg blocked"
    return 1
  fi
  echo "  free RAM ${free_kb} KiB >= ${need_kb} KiB required — ok"
}

start_server() { # envset model log ready_s
  local envset="$1" model="$2" log="$3" ready="$4"
  if curl -s -m2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "  !! port $PORT already serving — refusing to start (kill the other server by PID first)"
    return 1
  fi
  eval "$envset nohup $EXE serve --addr 127.0.0.1:$PORT --model \"$model\" --no-open > \"$log\" 2>&1 &"
  SERVER_PID=$!
  SERVER_WINPID=$(ps -p "$SERVER_PID" -W 2>/dev/null | awk 'NR==2{print $4}')
  local i
  for i in $(seq 1 "$ready"); do
    curl -s -m3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "  !! server exited during startup:"; tail -5 "$log"; SERVER_PID=""; return 1; }
    sleep 1
  done
  echo "  !! server not ready after ${ready}s"
  stop_server
  return 1
}

stop_server() { # kill THIS server by PID, wait for exit + port free
  [ -n "$SERVER_PID" ] || return 0
  kill "$SERVER_PID" 2>/dev/null
  local i
  for i in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    if [ -n "$SERVER_WINPID" ]; then
      echo "  (escalating: taskkill PID $SERVER_WINPID)"
      "$TK" //PID "$SERVER_WINPID" //F >/dev/null 2>&1
    else
      # WINPID lookup failed at start; last resort is SIGKILL via the MSYS pid.
      echo "  (escalating: kill -9 $SERVER_PID — no WINPID recorded)"
      kill -9 "$SERVER_PID" 2>/dev/null
    fi
    sleep 2
  fi
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "  !! server PID $SERVER_PID SURVIVED escalation — kill it manually before the next leg"
    FAILED=1
  fi
  for i in $(seq 1 15); do curl -s -m2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || break; sleep 1; done
  SERVER_PID=""
  SERVER_WINPID=""
}
trap stop_server EXIT

# A truncated corpus must never diff clean: if BOTH legs lose the same prompt
# (e.g. the probe script is swapped mid-run by a concurrent checkout — observed
# 2026-07-09, the Rung B "divergence" artifact), two 4-key corpora would pass
# byte-identity while claiming 5-prompt coverage. Assert the key count.
check_corpus() { # json-file expected-keys label
  local got
  got=$(node -p 'Object.keys(JSON.parse(require("fs").readFileSync(process.argv[1]))).length' "$1" 2>/dev/null)
  if [ "$got" != "$2" ]; then
    echo "  !! $3 corpus has ${got:-unreadable} keys, expected $2 — receipt VOID (probe/harness anomaly)"
    return 1
  fi
}

run_leg() { # name model-basename extra_env depth_tokens ready_s
  local name="$1" model extra_env="$3" depth="$4" ready="${5:-240}"
  model=$(find_model "$2")
  echo "=== leg $name ==="
  if [ -z "$model" ]; then echo "  !! model $2 not found — leg FAILED (coverage gap)"; FAILED=1; return; fi
  require_ram "$model" || { FAILED=1; return; }
  local ab_args="" want_keys=4
  [ "$depth" != "0" ] && { ab_args="--depth-tokens $depth"; want_keys=5; }

  # ---- OFF ----
  start_server "CAMELID_RESIDENT_TRACE=1 $extra_env" "$model" "$OUTDIR/$name.off.log" "$ready" || { FAILED=1; return; }
  node scripts/qwen3-cuda-prefill-ab.mjs --base "http://127.0.0.1:$PORT" $ab_args \
    > "$OUTDIR/$name.off.json" 2>"$OUTDIR/$name.off.err" || { echo "  !! AB(off) failed:"; tail -3 "$OUTDIR/$name.off.err"; FAILED=1; stop_server; return; }
  stop_server
  grep -q "\[cuda-streams\] off" "$OUTDIR/$name.off.log" \
    && echo "  off-leg trace: single stream confirmed" \
    || echo "  (note: no [cuda-streams] off line — engine may not have rebuilt in this leg)"

  # ---- ON ----
  start_server "CAMELID_RESIDENT_TRACE=1 CAMELID_CUDA_STREAMS=1 $extra_env" "$model" "$OUTDIR/$name.on.log" "$ready" || { FAILED=1; return; }
  node scripts/qwen3-cuda-prefill-ab.mjs --base "http://127.0.0.1:$PORT" $ab_args \
    > "$OUTDIR/$name.on.json" 2>"$OUTDIR/$name.on.err" || { echo "  !! AB(on) failed:"; tail -3 "$OUTDIR/$name.on.err"; FAILED=1; stop_server; return; }
  stop_server
  if ! grep -q "overlap ENGAGED" "$OUTDIR/$name.on.log"; then
    echo "  !! ENGAGED trace MISSING in ON leg — receipt VOID (lever not engaged)"
    FAILED=1
    return
  fi
  echo "  on-leg trace: overlap ENGAGED confirmed"

  check_corpus "$OUTDIR/$name.off.json" "$want_keys" "OFF" || { FAILED=1; return; }
  check_corpus "$OUTDIR/$name.on.json" "$want_keys" "ON" || { FAILED=1; return; }

  # ---- byte diff ----
  if diff -q "$OUTDIR/$name.off.json" "$OUTDIR/$name.on.json" >/dev/null; then
    echo "  PARITY: byte-identical OFF==ON  [PASS]"
  else
    echo "  PARITY: DIVERGED  [FAIL]"
    diff "$OUTDIR/$name.off.json" "$OUTDIR/$name.on.json" | head -20
    FAILED=1
  fi
}

for leg in $LEGS; do
  case "$leg" in
    1) run_leg "llama3b-q8" "Llama-3.2-3B-Instruct-Q8_0.gguf" "" 1881 240 ;;
    2) run_leg "qwen3-4b-q8" "Qwen3-4B-Q8_0.gguf" "" 1600 240 ;;
    3) run_leg "qwen3-4b-q4km" "Qwen3-4B-Q4_K_M.gguf" "" 1881 240 ;;
    4) run_leg "llama3b-q8-nofusion" "Llama-3.2-3B-Instruct-Q8_0.gguf" "CAMELID_RESIDENT_NO_FUSION=1" 1881 240 ;;
    5) run_leg "ornith9b-q4km-devloop" "ornith-1.0-9b-Q4_K_M.gguf" "CAMELID_RUNNABLE_SERVE=1 CAMELID_QWEN35_CUDA=1" 0 600 ;;
    *) echo "unknown leg: $leg"; FAILED=1 ;;
  esac
done

echo
if [ "$FAILED" -eq 0 ]; then echo "ALL REQUESTED LEGS PASS (byte-identical, engaged)"; else echo "VALIDATION FAILED — see above"; fi
exit "$FAILED"
