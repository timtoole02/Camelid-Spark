# Structural parity — Camelid vs oracle (llama-server acd79d6)

Both captured on `Qwen3-4B-Q8_0.gguf`, prompt `"What is 2+2? Reply with just the
number."`, `temperature: 0`, `seed: 42`, `max_tokens: 16`. Raw bytes:
`ref_usage_on.sse` / `camelid_usage_on.sse` (and `_off` baselines). Structural
summaries: `*_usage_on.analysis.json` (produced by `analyze_sse.mjs`).

Token **values** legitimately differ (Camelid's Qwen3 chat-template rendering
tokenizes to 25 prompt tokens vs the oracle's 21; the oracle was also given
`max_tokens:16` and ran to `length` while Camelid stopped at `stop` after 2
tokens). The mission is **structural** equivalence + correct field presence/typing.

## `stream_options.include_usage: true` — structural conformance

| Conformance target (from `oracle_contract.md`) | Oracle | Camelid | Match |
|---|---|---|---|
| Content/role chunks **omit** `usage` (no `usage: null`) | yes | yes | ✅ |
| Exactly **one** terminal usage chunk | 1 | 1 | ✅ |
| Terminal chunk `choices` is an **empty array** (present, not omitted) | `[]` | `[]` | ✅ |
| Terminal chunk keeps `id` / `created` / `model` | yes | yes | ✅ |
| Terminal chunk `object` == `chat.completion.chunk` | yes | yes | ✅ |
| `usage` has `prompt_tokens` / `completion_tokens` / `total_tokens` | yes | yes | ✅ |
| Usage chunk **after** the `finish_reason` chunk | yes | yes | ✅ |
| Usage chunk **before** `data: [DONE]` | yes | yes | ✅ |
| `[DONE]` is the last frame | yes | yes | ✅ |

### Documented, intentional structural differences (not defects)

| Field | Oracle | Camelid | Why |
|---|---|---|---|
| `system_fingerprint` on every chunk | present | **absent** | Camelid never emitted it; adding it would break the byte-identical usage-off baseline. |
| `timings` on last chunk | present | **absent** | Oracle debug extra; out of scope. |
| `usage.prompt_tokens_details.cached_tokens` | present | **absent** | Oracle extension; the spec's required usage shape is the 3 integers only. |

These are oracle-specific decorations Camelid has never produced. The spec's
structural gate is on the required fields above; the extras are explicitly out of
scope (see `oracle_contract.md` → "fields intentionally NOT adopted").

## `stream_options` omitted — regression boundary

`camelid_usage_off.sse`: `usage_chunk_count: 0`, `empty_choice_chunk_count: 0`,
ends with the `finish_reason` chunk then `data: [DONE]` — i.e. **no `usage` key
appears anywhere**. The only serialization change is the new
`usage: Option<CompletionUsage>` field with
`#[serde(skip_serializing_if = "Option::is_none")]`, which emits **zero bytes**
when `None`. Therefore the usage-off stream is byte-identical to the pre-change
baseline by construction. Asserted in the unit test
`streaming_chunks_omit_camelid_diagnostics_by_default`
(`assert!(value.get("usage").is_none())`).

## Internal consistency (invariant #3) — `consistency_check.txt`

Same prompt, `stream:false` vs `stream:true,include_usage:true`:

```
streaming_usage    = {prompt_tokens:25, completion_tokens:2, total_tokens:27}
nonstreaming_usage = {prompt_tokens:25, completion_tokens:2, total_tokens:27}
outputs_identical  = true   ("4" == "4")
VERDICT: PASS — streaming usage == non-streaming usage
```

The streaming counts are computed from the same expressions the non-streaming
path uses (`prepared.token_ids.len()`; sampled-token vector length), so they are
equal by construction and verified empirically. No pre-existing counting-bug
finding to report.
