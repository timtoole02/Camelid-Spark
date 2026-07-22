# Phase 3 — end-to-end client validation

Both clients ran against the patched Camelid (`qwen3-4b`, release build
`target/release/camelid.exe` @ 2026-06-23 11:08) through a logging reverse proxy
(`logging-proxy.mjs`, 127.0.0.1:8182 → Camelid 127.0.0.1:8181) that tees the raw
request and response bytes. Full wire log: `proxy.log`.

## OpenCode v1.17.9 (Mike's reporting client) — PASS

Config: `opencode-run/opencode.json` (provider `camelid`, AI SDK
`@ai-sdk/openai-compatible`, baseURL → proxy). OpenCode's openai-compatible
provider sends `stream_options.include_usage: true` by default on streamed turns —
this is exactly what produced Mike's
`"OpenAI stream_options are not supported yet"` error before the fix.

Proof: `opencode-run/opencode-stream-options-proof.txt` (extracted from
`proxy.log`).

- **Request** (OpenCode → Camelid): `"stream":true,"stream_options":{"include_usage":true}`
- **Response**: `RESP 200` (previously this was a `400 unsupported_parameter`
  with the "stream_options are not supported yet" message — that error is gone).
- **Raw SSE** ends with the terminal usage chunk on the wire:
  `data: {"...","choices":[],"usage":{"prompt_tokens":544,"completion_tokens":4,"total_tokens":548}}`
  then `data: [DONE]`.

Note (out of scope): OpenCode's *full agent* turn additionally sends a `tools`
array, which Camelid still rejects (`tool/function calling is not supported by
Camelid generation routes yet`). That is a separate, pre-existing limitation
documented in the prior OpenClaw bring-up (Camelid does not surface structured
`tool_calls`), explicitly out of scope for this mission. The `stream_options`
surface — the entire subject of this change — works.

## OpenClaw 2026.6.9 — PASS

Config: the existing working bring-up config with **one** flag flipped —
`compat.supportsUsageInStreaming` from `false` → **`true`**. That flag had to be
`false` in the prior bring-up *specifically because* Camelid rejected
`stream_options`; with this fix it can be enabled. Before/after configs:
`openclaw-config-before.json` / `openclaw-config-after.json` (the live config was
restored to `before` after the run).

Command: `openclaw infer model run --local --model camelid/qwen3-4b --prompt
"Reply with exactly: PONG" --thinking off` → `EXIT=0`, output `PONG`
(`openclaw-session.txt`).

Proof (from `proxy.log`, after `OPENCLAW RUN START`):
- **Request** (OpenClaw → Camelid): `"stream":true,"stream_options":{"include_usage":true}`
- **Response**: `RESP 200`, terminal usage chunk on the wire:
  `data: {"...","choices":[],"usage":{"prompt_tokens":18,"completion_tokens":3,"total_tokens":21}}`
  then `data: [DONE]`. The agent turn completed and usage was surfaced.

## Verdict

Mike's reported gap is closed. The exact request both clients emit
(`stream_options.include_usage: true`) is now accepted and answered with an
OpenAI-conformant terminal usage chunk, proven on the wire for both clients.
