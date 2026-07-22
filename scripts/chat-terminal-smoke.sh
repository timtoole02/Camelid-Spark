#!/usr/bin/env bash
# Phase 8 end-to-end smokes for `camelid chat` (the interactive terminal REPL).
#
# Like the other parity/evidence scripts, this is GATED on env vars pointing at
# real GGUFs and is a no-op (exit 0, with a notice) when they are unset, so it
# never blocks `cargo test`. It drives the actual `camelid chat` binary with
# piped stdin (rustyline falls back to plain line reading when stdin is not a
# TTY), proving the terminal lane produces the same audited output the
# `/v1/chat/completions` SSE path serves.
#
# Env:
#   CAMELID_BIN                 path to the built camelid binary
#                               (default: $CARGO_TARGET_DIR/debug/camelid or
#                               ./target/debug/camelid)
#   CAMELID_CHAT_SUPPORTED_GGUF supported GGUF to chat (e.g. the TinyLlama
#                               1.1B Chat Q8_0 row). Supported-path check.
#   CAMELID_CHAT_EXPECT         expected assistant-text prefix for the supported
#                               run (default "Certainly" — for TinyLlama the
#                               first generated token is 29907 / "C", and the
#                               reply begins "Certainly!").
#   CAMELID_CHAT_UNSUPPORTED_GGUF  a recognized-but-unsupported-architecture GGUF.
#                               Gate check: must exit non-zero with the engine's
#                               typed unsupported-state message.
#   CAMELID_CHAT_ADDR           host:port to spawn the test server on
#                               (default 127.0.0.1:8231).
set -u

bin="${CAMELID_BIN:-${CARGO_TARGET_DIR:-./target}/debug/camelid}"
addr="${CAMELID_CHAT_ADDR:-127.0.0.1:8231}"
expect="${CAMELID_CHAT_EXPECT:-Certainly}"

if [[ ! -x "$bin" ]]; then
  echo "chat-terminal-smoke: camelid binary not found at '$bin' — build it or set CAMELID_BIN" >&2
  exit 2
fi

fail=0

# ---- Supported-path check ---------------------------------------------------
if [[ -n "${CAMELID_CHAT_SUPPORTED_GGUF:-}" ]]; then
  echo "== supported-path: chatting ${CAMELID_CHAT_SUPPORTED_GGUF}"
  out="$(printf 'hello\n/exit\n' | NO_COLOR=1 "$bin" chat \
    --model "$CAMELID_CHAT_SUPPORTED_GGUF" --addr "$addr" --plain --max-tokens 24 2>/dev/null)"
  code=$?
  echo "$out"
  if [[ $code -ne 0 ]]; then
    echo "FAIL supported-path: exit $code (expected 0)" >&2
    fail=1
  elif ! grep -qF "$expect" <<<"$out"; then
    echo "FAIL supported-path: output did not contain expected prefix '$expect'" >&2
    fail=1
  else
    echo "PASS supported-path: streamed reply contained '$expect'"
  fi
else
  echo "skip supported-path: set CAMELID_CHAT_SUPPORTED_GGUF to run"
fi

# ---- Gate check (unsupported architecture is refused) -----------------------
if [[ -n "${CAMELID_CHAT_UNSUPPORTED_GGUF:-}" ]]; then
  echo "== gate: loading ${CAMELID_CHAT_UNSUPPORTED_GGUF} must be refused"
  err="$(printf '/exit\n' | NO_COLOR=1 "$bin" chat \
    --model "$CAMELID_CHAT_UNSUPPORTED_GGUF" --addr "$addr" --plain 2>&1 1>/dev/null)"
  code=$?
  echo "$err"
  if [[ $code -eq 0 ]]; then
    echo "FAIL gate: exit 0 (expected non-zero refusal)" >&2
    fail=1
  elif [[ -z "$err" ]]; then
    echo "FAIL gate: no typed unsupported-state message printed" >&2
    fail=1
  else
    echo "PASS gate: exit $code with typed unsupported-state message"
  fi
else
  echo "skip gate: set CAMELID_CHAT_UNSUPPORTED_GGUF to run"
fi

exit $fail
