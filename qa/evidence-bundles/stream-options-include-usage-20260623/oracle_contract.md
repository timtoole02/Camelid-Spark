# Oracle contract — `stream_options.include_usage`

**Mission:** OpenAI-compatible `stream_options.include_usage` on Camelid's
chat-completions streaming endpoint, so agent clients (OpenCode, OpenClaw) get the
terminal usage chunk they expect.

## Chosen oracle and provenance

- **Oracle: llama.cpp `llama-server`, pinned reference commit `acd79d6`**
  (`acd79d603` — `jinja : add count/d/e filter aliases (#24606)`), built locally at
  `<home>\llama.cpp\build\bin\llama-server.exe`.
- This commit **implements** `stream_options.include_usage`
  (`tools/server/server-task.cpp:262-263`, `:914-926`), so it is the canonical
  oracle per the spec's precedence rule #1. OpenAI live capture was **not** used
  (no `OPENAI_API_KEY` available, and not needed — the pinned oracle supports the
  feature).
- Captured against `Qwen3-4B-Q8_0.gguf` — the **same** model Camelid serves — so
  even token *values* are directly comparable, not just structure.
- Capture parameters: `temperature: 0`, `seed: 42`, `max_tokens: 16`, prompt
  `"What is 2+2? Reply with just the number."` (usage cases) / `"hi"` (error cases).

## Extracted conformance targets (observed in `ref_usage_on.sse`)

1. **`usage` on content/role chunks: OMITTED** (not `usage: null`). Every
   role/content/finish chunk has no `usage` key at all. → Camelid serializes
   `usage: Option<CompletionUsage>` with `skip_serializing_if = "Option::is_none"`,
   so the key is absent on every non-terminal chunk.
2. **Terminal usage chunk shape:** `"choices": []` (empty array, present — not
   omitted); `id`, `created`, `model`, `object: "chat.completion.chunk"` all still
   present; populated `usage` object. → matched.
3. **Ordering:** terminal usage chunk comes **after** the chunk bearing
   `finish_reason` (the `delta: {}` chunk) and **before** `data: [DONE]`. → matched.
4. **Usage integers:** `{prompt_tokens, completion_tokens, total_tokens}`.
   - Oracle also adds `prompt_tokens_details.cached_tokens`. **Not adopted** by
     Camelid (see "Intentional non-adoptions").

### Oracle-specific fields intentionally NOT adopted

The oracle decorates every chunk with fields Camelid does not emit. Adopting them
would change the byte-for-byte `usage_off` baseline (regression boundary, invariant
in Phase 2), so they are deliberately excluded:

- `system_fingerprint` (e.g. `"b1-acd79d6"`) — on every oracle chunk; Camelid has
  never emitted it.
- `timings` (prompt/predicted ms) — oracle appends it to the last chunk.
- `usage.prompt_tokens_details.cached_tokens` — oracle extension.

Camelid's terminal usage chunk therefore equals the spec's required minimal shape:
`{id, object, created, model, choices: [], usage: {prompt_tokens, completion_tokens,
total_tokens}}`. Structural parity is asserted on the spec-required fields; the
oracle's extra decoration is out of scope.

## Error cases — the oracle is PERMISSIVE (recorded finding)

The spec hypothesized the oracle would define `400 invalid_request_error`
envelopes. **It does not.** Captured behavior of `llama-server acd79d6`:

| Case | Request | Observed |
|---|---|---|
| `ref_err_stream_false` | `stream: false` + `stream_options: {include_usage: true}` | **HTTP 200**, normal non-streaming completion. `stream_options` silently ignored. |
| `ref_err_bad_type` | `stream: true` + `stream_options: {include_usage: "yes"}` | **HTTP 200**, normal stream, **no** terminal usage chunk. `include_usage` coerced to `false`. |

Root cause in oracle source: `json_value(stream_opt, "include_usage", false)`
returns the default on type mismatch and there is no `stream:false` cross-check
(`server-task.cpp:262-263`).

### Decision (recorded)

The spec's invariant #5 ("fail-explicit errors") / Phase 1 ("reject with 400")
**conflict** with the captured oracle (permissive, HTTP 200). This was surfaced to
the project owner, who directed: **"make this work, no false-stream [rejection], it
just works."**

→ **Camelid matches the oracle: permissive, no 4xx for either case.**
- `stream: false` + `stream_options`: not rejected. The non-streaming response
  already returns `usage`, so it "just works" untouched.
- malformed / wrong-typed / unknown subfields: tolerated and ignored
  (`stream_options_include_usage` resolves to `false`); the request never errors.
- Only an explicit `include_usage: true` turns the terminal usage chunk on.

This honors invariant #1 (oracle before code — confirmed, not assumed) and #2
(exact-row support — `include_usage` only; unknown subfields tolerated, never
promoted). It is a deliberate, documented departure from Phase 1's 400s, owner-
approved.

## Token-count binding (invariant #3 — single source of truth)

Streaming usage integers are computed from the **same** values the non-streaming
endpoint uses, in `src/api/mod.rs`:

- `prompt_tokens`  = `prepared.token_ids.len()` (post chat-template tokenization),
  identical expression to the non-streaming path (`prompt_token_count`).
- `completion_tokens` = `generated.len()` (the sampled-token vector the decode loop
  already accumulates; same vector the telemetry `completion_tokens` is read from).
- `total_tokens` = `prompt_tokens + completion_tokens`.

No second counting path; no estimates. Internal consistency (stream == non-stream
for identical prompt+output) is asserted empirically in `consistency_check.txt`.
