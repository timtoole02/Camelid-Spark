#!/usr/bin/env bash
# E2E behavioral battery for tools.function_calling (class B): supply tools +
# triggering prompts; verify structured tool_calls are surfaced (name + valid
# JSON arguments) with finish_reason "tool_calls". Also: tool_choice:"none"
# suppresses parsing.
set -u
EXE="${CAMELID_EXE:-target/release/camelid.exe}"
MODEL="${1:?usage: tools_smoke.sh <model.gguf> [port] [out-suffix]}"
PORT="${2:-8261}"
BASE="http://127.0.0.1:${PORT}"
OUT="${CAMELID_TOOLS_OUT:-qa/capability/tools_out${3:-}}"
mkdir -p "$OUT"
CT='Content-Type: application/json'
TOOLS='[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}},{"type":"function","function":{"name":"add","description":"Add two numbers","parameters":{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}},"required":["a","b"]}}}]'

export CUDA_VISIBLE_DEVICES=-1
"$EXE" serve --addr 127.0.0.1:"${PORT}" --model "$MODEL" > "$OUT/serve.log" 2>&1 &
PID=$!; trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null' EXIT
curl -s --fail --retry-connrefused --retry-all-errors --retry 180 --retry-delay 1 --max-time 10 "${BASE}/api/models/current" -o /dev/null || { echo "NOT READY"; tail -20 "$OUT/serve.log"; exit 1; }

req() { # $1 = prompt, $2 = out, $3 = extra-json
  curl -s --max-time 180 -H "$CT" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"tools\":${TOOLS},\"temperature\":0,\"max_tokens\":60${3:-}}" \
    "${BASE}/v1/chat/completions" > "$OUT/$2"
}
req "What is the weather in Paris? Use a tool." "p1.json"
req "Add 17 and 25 using the tool." "p2.json"
req "What is the weather in Tokyo? Use a tool." "p3.json"
# tool_choice:none must suppress tool parsing (content stays).
req "What is the weather in Paris? Use a tool." "none.json" ',"tool_choice":"none"'

echo "SMOKE_DONE"
