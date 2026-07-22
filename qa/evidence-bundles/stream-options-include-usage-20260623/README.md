# Receipt bundle — `stream_options.include_usage`

Adds OpenAI-compatible `stream_options.include_usage` to Camelid's
chat-completions **streaming** endpoint, closing the user-reported gap
(`"OpenAI stream_options are not supported yet; Camelid streams plain SSE
chunks"`). Code change: `src/api/mod.rs` only (HTTP serialization). No decode,
sampler, or parity-locked kernel touched. Exactly one new support-ledger row
(`api_features` → `stream_options.include_usage`).

## What "supported" means here (one exact row)

`stream:true` + `stream_options:{include_usage:true}` → after the `finish_reason`
chunk and before `data: [DONE]`, Camelid emits exactly one terminal chunk:
`{id, object:"chat.completion.chunk", created, model, choices:[], usage:{prompt_tokens,
completion_tokens, total_tokens}}`. The usage integers are the **same** values the
non-streaming endpoint returns (single source of truth). Omitting `stream_options`
is byte-identical to the prior baseline. Malformed shapes / other subfields are
tolerated and ignored (no error), matching the oracle (owner-approved — see
`oracle_contract.md` → "Decision").

## Files

### Phase 0 — oracle (llama-server acd79d6, captured on Qwen3-4B-Q8_0)
- `oracle_contract.md` — chosen oracle + provenance, extracted conformance
  targets, the **permissive error finding**, the owner decision, and the
  token-count binding.
- `ref_usage_on.sse` / `ref_usage_off.sse` — raw oracle SSE (usage on / baseline).
- `ref_err_stream_false.json` / `ref_err_bad_type.json` — the two "error" cases:
  oracle returns **HTTP 200** (permissive), recorded verbatim.

### Phase 2 — wire-format parity (Camelid, same prompt/seed/model)
- `camelid_usage_on.sse` / `camelid_usage_off.sse` / `camelid_nonstream.json` — raw.
- `analyze_sse.mjs` — structural SSE analyzer (no token values).
- `oracle_usage_on.analysis.json` / `camelid_usage_on.analysis.json` (+ `_off`) —
  analyzer outputs compared in `structural_diff.md`.
- `structural_diff.md` — the structural parity table (all targets ✅) + the
  documented oracle-only fields Camelid intentionally does not adopt + the
  regression-boundary argument.
- `consistency_check.txt` — streaming usage == non-streaming usage (PASS:
  25/2/27 both, output "4" identical).

### Phase 3 — end-to-end clients (proof the user's problem is gone)
- `phase3/e2e-validation.md` — summary (both clients PASS).
- `phase3/proxy.log` — raw tee of every client↔Camelid request+response.
- `phase3/logging-proxy.mjs` — the capture proxy.
- `phase3/opencode-run/` — OpenCode v1.17.9 config + session notes +
  `opencode-stream-options-proof.txt` (OpenCode sent
  `stream_options.include_usage:true` → 200 + terminal usage chunk on the wire).
- `phase3/openclaw-session.txt` + `openclaw-config-before/after.json` — OpenClaw
  2026.6.9 with `supportsUsageInStreaming` flipped `false`→`true` (the flag that
  previously *had* to be off), turn completed, usage chunk on the wire.

## Reproduce

```
# Oracle (Phase 0): build llama-server at acd79d6, then
llama-server -m Qwen3-4B-Q8_0.gguf --port 8099 -c 4096 -ngl 0 -s 42
# Camelid (Phase 2): cargo build --release --bin camelid; serve on :8181 w/ qwen3
# Structural diff:
node analyze_sse.mjs camelid_usage_on.sse
# Unit gates:
cargo test --lib -- stream_options_include_usage \
  chat_request_accepts_stream_options_without_marking_it_unsupported \
  terminal_usage_chunk_has_empty_choices_array_and_usage \
  streaming_chunks_omit_camelid_diagnostics_by_default
```
