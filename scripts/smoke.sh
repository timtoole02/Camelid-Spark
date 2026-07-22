#!/usr/bin/env bash
# One-command end-to-end smoke for Camelid's first-demo path.
#
# It exercises the REAL binary — no mocks, no stubs:
#   1. pull TinyLlama 1.1B Chat Q8_0 (the baseline gate row) into ./models
#   2. serve it over the local OpenAI-style API
#   3. wait for /health to report generation_ready
#   4. do one real /v1/chat/completions round-trip
#   5. assert the reply is a non-empty assistant message
#
# Exit 0 = the whole path works on this machine. Any failure exits non-zero
# with the reason. The server is always torn down on exit.
#
# Usage:   scripts/smoke.sh
# Env:
#   CAMELID_BIN     path to a prebuilt `camelid` binary (skips the cargo build)
#   CAMELID_PORT    port to serve on (default 8231 — a dedicated smoke port, kept
#                   off Camelid's normal 8181 so we never poll a server we didn't
#                   start)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"
GGUF="$MODELS_DIR/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
PORT="${CAMELID_PORT:-8231}"
ADDR="127.0.0.1:$PORT"
BASE="http://$ADDR"

say() { printf '\n\033[1m▶ %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31m✗ smoke failed: %s\033[0m\n' "$*" >&2; exit 1; }

# --- pick a JSON tool (jq preferred, python3 fallback) -----------------------
# json_get reads stdin and prints one field. jq and python3 use different path
# syntax, so callers pass BOTH: $1 = jq filter, $2 = python expr over object `d`.
if command -v jq >/dev/null 2>&1; then
  JSON_TOOL=jq
elif command -v python3 >/dev/null 2>&1; then
  JSON_TOOL=python3
else
  fail "need either jq or python3 to parse JSON responses"
fi
json_get() {
  if [[ "$JSON_TOOL" == jq ]]; then
    jq -er "$1"
  else
    python3 -c 'import sys,json; d=json.load(sys.stdin); v=eval(sys.argv[1]); print(v if v is not None else (_ for _ in ()).throw(SystemExit(1)))' "$2"
  fi
}

# --- resolve the binary ------------------------------------------------------
BIN="${CAMELID_BIN:-}"
if [[ -z "$BIN" ]]; then
  for candidate in \
    "$REPO_ROOT/target/release/camelid" \
    "${CARGO_TARGET_DIR:-}/release/camelid"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then BIN="$candidate"; break; fi
  done
fi
if [[ -z "$BIN" ]]; then
  say "no prebuilt binary found — building release (this is slow the first time)"
  ( cd "$REPO_ROOT" && cargo build --release )
  BIN="$REPO_ROOT/target/release/camelid"
fi
[[ -x "$BIN" ]] || fail "camelid binary not found/executable at: $BIN"
say "using binary: $BIN"

# --- 1. pull TinyLlama (skips if already complete) ---------------------------
if [[ ! -f "$GGUF" ]]; then
  say "pulling TinyLlama into $MODELS_DIR"
  "$BIN" pull tinyllama --models-dir "$MODELS_DIR"
else
  say "TinyLlama already present: $GGUF"
fi
[[ -f "$GGUF" ]] || fail "TinyLlama GGUF missing after pull: $GGUF"

# --- 2. serve it (background) ------------------------------------------------
# Refuse to start if a server is already answering on the port — otherwise we
# might poll a foreign server's /health and "pass" without ever testing our own
# binary. (A non-HTTP listener instead trips the bind failure caught below.)
if curl -fsS "$BASE/health" >/dev/null 2>&1; then
  fail "something is already serving on $ADDR — set CAMELID_PORT to a free port and re-run"
fi

say "starting server on $ADDR"
SERVER_LOG="$(mktemp -t camelid-smoke-server.XXXXXX)"
"$BIN" serve --model "$GGUF" --addr "$ADDR" --no-open >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -f "$SERVER_LOG"
}
trap cleanup EXIT

# --- 3. wait for generation_ready -------------------------------------------
# `serve` loads the model *before* it binds the port, so /health stays
# unreachable until load finishes — then reports generation_ready almost at
# once. A long silent wait therefore means a slow/stuck load (often memory
# pressure), not a half-ready server. Budget is generous and configurable.
TIMEOUT="${CAMELID_SMOKE_TIMEOUT:-300}"
say "waiting for the model to load (up to ${TIMEOUT}s; the port opens only once load finishes)"
ready=""
for _ in $(seq 1 "$TIMEOUT"); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "--- server log ---" >&2; cat "$SERVER_LOG" >&2
    fail "server exited before becoming ready"
  fi
  health="$(curl -fsS "$BASE/health" 2>/dev/null || true)"
  if [[ -n "$health" ]]; then
    gr="$(echo "$health" | json_get '.generation_ready' 'd["generation_ready"]' 2>/dev/null || true)"
    if [[ "$gr" == "true" || "$gr" == "True" ]]; then ready=1; break; fi
  fi
  sleep 1
done
if [[ -z "$ready" ]]; then
  echo "--- server log ---" >&2; cat "$SERVER_LOG" >&2
  fail "model did not finish loading within ${TIMEOUT}s — the port never opened. Likely a slow/stuck load (e.g. low free memory). Raise CAMELID_SMOKE_TIMEOUT or free up RAM and retry."
fi
say "server reports generation_ready"

# --- 4. one real chat round-trip --------------------------------------------
say "sending one chat message"
REQ='{"messages":[{"role":"user","content":"Reply with the single word: pong"}],"max_tokens":16,"temperature":0}'
RESP="$(curl -fsS -X POST "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$REQ")" || fail "/v1/chat/completions request failed"

# --- 5. assert a non-empty assistant reply ----------------------------------
CONTENT="$(echo "$RESP" | json_get '.choices[0].message.content' 'd["choices"][0]["message"]["content"]')" \
  || fail "response missing choices[0].message.content — got: $RESP"
[[ -n "${CONTENT//[$' \t\r\n']/}" ]] || fail "assistant reply was empty — got: $RESP"

printf '\n\033[32m✓ smoke passed\033[0m\n'
printf '  model reply: %q\n' "$CONTENT"
