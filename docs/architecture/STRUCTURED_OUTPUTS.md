# Structured Outputs (`response_format` constrained decoding)

Status: shipped for `/v1/chat/completions`, non-streaming, single- and
multi-choice. This document is the authoritative contract for the JSON Schema
subset Camelid enforces byte-for-byte, the invariants the implementation pins
with tests, and the relaxations it deliberately carries.

Design principle: **fail closed**. A constraint Camelid cannot enforce
byte-for-byte is a typed request-time error naming the offending keyword —
never a silently dropped or partially honored constraint.

## 1. Supported subset (the contract)

`response_format: {"type":"json_object"}` — any valid JSON object (the
pre-existing JSON-object grammar; unchanged).

`response_format: {"type":"json_schema","json_schema":{"schema":{...}}}` —
the output is a value matching the schema, within this subset:

| Feature | Support |
|---|---|
| Root | `object` or `array` only (top-level scalars have no terminator) |
| Object | `properties` + `required`; `additionalProperties:false` is **mandatory** |
| Array | `items` is **mandatory** |
| Scalars | `string`, `integer`, `number`, `boolean`, `null` |
| String literals | `enum` / `const` — string members only, **ASCII-only** (see §3); a non-string sibling `type` is a contradiction and is rejected |
| Type unions | primitive `type` arrays whose members start with distinct bytes — i.e. the OpenAI nullable pattern `["T","null"]`; overlapping shapes (`["integer","number"]`) are rejected |
| Annotations | `description`, `title`, `default`, `examples`, `$schema`, `$id`, `$comment`, `readOnly`, `writeOnly`, `deprecated`, `$defs`, `definitions` are ignored |
| Everything else | typed 400 naming the keyword (`$ref`, `anyOf`/`oneOf`/`allOf`, `minLength`, `pattern`, open objects, untyped nodes, non-string enums, ...) |

Property keys are emitted in canonical unescaped form; property names that
would require JSON escaping (`"`, `\`, control chars) are rejected at compile
time.

### Schema-dimension caps (compile-time, typed 400s)

Every declared property and enum member multiplies the per-step full-vocab
`accepts()` scan, and the automaton state is cloned once per candidate token
per decode step on the single engine worker. Bounds (`src/grammar.rs`):

| Cap | Value |
|---|---|
| `MAX_OBJECT_PROPERTIES` (per object) | 256 |
| `MAX_PROPERTY_NAME_BYTES` | 64 |
| `MAX_ENUM_MEMBERS` | 256 |
| `MAX_ENUM_MEMBER_BYTES` (JSON-encoded literal) | 256 |
| `MAX_SCHEMA_DEPTH` (root node is depth 0) | 32 |

Test: `grammar::tests::schema_dimension_caps_reject_oversized_and_accept_at_cap`.

## 2. Invariants

Each invariant names the test that pins it (or the inspected location where a
unit-level pin is not drivable).

1. **EOG iff done.** The per-step mask allows an EOG token exactly when the
   constrained value is complete (`state.is_done()`), and the loop stops the
   moment the value completes. Pinned by inspection: the mask construction and
   the `done` gate in `generate_token_ids` (`src/api/mod.rs`, grammar-mask
   block), plus `grammar::tests::done_only_after_top_level_close` /
   `schema_masks_candidates_without_mutation` at the automaton level.
2. **Speculative and GPU fast lanes are disabled under any constraint.** Both
   the speculative-round filter and the resident-GPU fast-step gate require
   `grammar.is_none()` (`src/api/mod.rs`, decode loop). Every constrained step
   takes the general step with the mask applied.
3. **`stream:true` + constraint → 400.** Tests:
   `stream_true_with_json_object_is_rejected`,
   `stream_true_with_json_schema_is_rejected` (tests/api_vertical_slice.rs).
4. **The prompt-prefix cache never interacts with constrained decoding.**
   Constrained requests neither read nor write it; the gate lives inside
   `lookup_prompt_prefix_cache` / `store_prompt_prefix_cache` so every decode
   path inherits it. Tests:
   `constrained_generation_skips_prompt_prefix_cache_lookup`,
   `constrained_generation_does_not_store_prompt_prefix_cache`.
5. **Constrained output is never reclassified as a tool call.** OpenAI allows
   tools + `response_format` together; tools are still rendered into the
   prompt, but `parse_tool_calls` never runs under an active constraint (a
   schema legitimately declaring a `name` property must not have its output
   converted into a fabricated tool call). Test:
   `constrained_output_is_never_reclassified_as_tool_call`.
6. **Receipts stamp and replay the constraint.** `ReceiptRequest` records the
   raw `response_format` (digest-stable: absent for unconstrained receipts,
   digest-bound when present) and `replay_receipt_request` re-compiles it, so
   an honest constrained receipt verifies. Tests:
   `receipt_request_without_constraint_serializes_unchanged`,
   `receipt_stamp_records_response_format`.
7. **The automaton fails sticky-closed.** A rejected byte poisons
   `SchemaState` (`SchemaMode::Failed`): it accepts nothing afterwards and
   never reports done, so a mask/advance divergence can only surface as an
   error, never as a truncated value with a clean `finish_reason: "stop"`.
   Test: `advance_error_is_sticky_failed`.
8. **No reachable dead ends from exhausted objects.** Once every declared
   property is used, `,` and a key-opening `"` are masked (only `}` continues);
   with a property still unused they remain legal. This can never lock an
   object out: no-unused-properties implies every required property is used,
   so `}` is legal at that point. Tests:
   `exhausted_object_rejects_comma_and_key_quote`,
   `empty_object_schema_rejects_key_quote`,
   `partial_object_still_accepts_comma`.
9. **`json_object` parity.** The `json_object` path and `JsonState` are
   byte-identical in behavior to the pre-json_schema implementation; the
   Schema work added code but changed none of its logic. Pinned by the
   untouched pre-existing `JsonState` test suite.

## 3. Known relaxations / deferred

- **Lone surrogate escapes** (e.g. `"\uD800"` unpaired): accepted inside
  free-form strings. The `\u` handling (`str_advance`) is genuinely shared
  between `JsonState` and `SchemaState`, and the shipped `json_object` path has
  the same relaxation, so a Schema-only rejection is not cleanly separable
  without touching `JsonState` behavior — out of bounds for this campaign.
  Follow-up: track surrogate pairing in the string sub-state for both paths in
  one deliberate change.
- **Non-ASCII free-form strings**: the per-token byte table decodes each token
  in isolation, so byte-level-BPE fragments of a multibyte character are masked
  out; free-form string values degrade to the ASCII-reachable subset on such
  tokenizers. Non-ASCII `enum`/`const` literals — where this would guarantee a
  mid-generation dead end — are rejected at compile time instead
  (`non_ascii_enum_or_const_is_rejected`); if a free-form value still dead-ends
  on a pathological tokenizer, the decode fails typed (see §4), never silently.
- **Env-gated serve lanes** (gemma4, runnable/Ornith, DiffusionGemma): return a
  typed 400 for any `response_format` constraint rather than enforcing it. The
  constraint parse runs before lane dispatch, so malformed schemas 400
  uniformly on every lane.
- **Tool-call constraining** (forcing the model's tool-call JSON to match the
  declared tool schema) is deferred: it requires a per-model grammar for the
  tool-call wire format (each chat template renders tool calls differently),
  which is a separate lane from response_format. Tools + response_format
  compose today per §2.5.
- **Streaming** constrained decoding is deferred (400 today, see §2.3).

## 4. Error taxonomy

| Class | HTTP | `error.type` | When |
|---|---|---|---|
| Malformed request | 400 | `invalid_request_error` | `response_format` shape is wrong (e.g. `json_schema` without a `json_schema.schema` payload), or constraint + `stream:true` |
| Out-of-subset schema | 400 | `unsupported_parameter` | The schema parses but uses a feature outside §1 (message names the keyword or cap), or a constrained request hits an env-gated serve lane |
| Decode-time dead end | 422 | `constraint_unsatisfiable` | No token in this model's tokenizer can produce the next required byte and stopping is not yet legal (pathological tokenizer; see §3) |

The 400s are request-shape facts, knowable before generation; the 422 is a
model×tokenizer fact, only knowable at decode time — hence the distinct status
so callers can distinguish "fix the schema" from "this model cannot honor it".
