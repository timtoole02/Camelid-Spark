#!/usr/bin/env bash
# E2E behavioral battery for structured.json_grammar (class B): response_format
# json_object -> JSON-grammar-constrained decoding. Every response's content MUST
# be valid JSON. Also: without json_object the model is free (control).
set -u
EXE="${CAMELID_EXE:-target/release/camelid.exe}"
MODEL="${1:?usage: json_smoke.sh <model.gguf> [port] [out-suffix]}"
PORT="${2:-8271}"
BASE="http://127.0.0.1:${PORT}"
OUT="${CAMELID_JSON_OUT:-qa/capability/json_out${3:-}}"
mkdir -p "$OUT"
CT='Content-Type: application/json'

export CUDA_VISIBLE_DEVICES=-1
"$EXE" serve --addr 127.0.0.1:"${PORT}" --model "$MODEL" > "$OUT/serve.log" 2>&1 &
PID=$!; trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null' EXIT
curl -s --fail --retry-connrefused --retry-all-errors --retry 180 --retry-delay 1 --max-time 10 "${BASE}/api/models/current" -o /dev/null || { echo "NOT READY"; tail -20 "$OUT/serve.log"; exit 1; }

 jreq() { # $1 prompt, $2 out
  curl -s --max-time 240 -H "$CT" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"response_format\":{\"type\":\"json_object\"},\"temperature\":0,\"max_tokens\":300}" \
    "${BASE}/v1/chat/completions" > "$OUT/$2"
}
jreq "Give me a JSON object with a person's name, age, and city." "p1.json"
jreq "Return JSON with keys a, b, c set to the numbers 1, 2, 3." "p2.json"
jreq "Describe a cat as JSON." "p3.json"
jreq "List three colors as a JSON object with a colors array." "p4.json"
# control: no response_format -> free text (not constrained).
curl -s --max-time 120 -H "$CT" \
  -d '{"messages":[{"role":"user","content":"Say hello in one sentence."}],"temperature":0,"max_tokens":20}' \
  "${BASE}/v1/chat/completions" > "$OUT/free.json"

echo "SMOKE_DONE"
