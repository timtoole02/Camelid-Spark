# Gemma 4 speculative decode (CPU) — lossless, 2026-06-11

> [!NOTE]
> N-gram (prompt-lookup) speculative decode for the single-node gemma4 CPU runtime.
> Measured on an Apple M4 (16 GB). Lossless: the emitted token stream is identical
> to plain greedy decode — speculation only changes how many tokens fall out of one
> weight read.

## What it is

Decode is memory-bandwidth bound: each sequential token costs a full pass over the
weights. Speculative decode drafts several likely next tokens cheaply, then **verifies
them in one batched forward** (`step_chunk`) so a single weight read can commit
several tokens.

- **Drafter:** the existing `NGramDrafter` (prompt-lookup) — proposes the
  continuation of the most recent earlier occurrence of the current suffix. Zero
  extra weights; wins on repetitive/structured text, proposes nothing on novel text.
- **Verify:** `step_chunk` runs K tokens at K consecutive positions through the full
  gemma4 forward, reading each weight matrix **once** via a new batched `matmul_q`
  (and `matmul_q8k` for the Q6_K tied head). `matmul_q` dots each weight row against
  all K activations, so `out[k]` is bit-identical to K separate `matvec_q` calls.
- **Acceptance:** the longest draft prefix equal to the target's own argmax is
  committed; the divergence position's logits carry into the next round; the KV cache
  (an append-only `Vec` per layer) is truncated back to the accepted length to drop
  rejected drafts. Every committed token is the target's greedy argmax, so output ==
  greedy.

Opt-in: `CAMELID_GEMMA4_SPEC_DECODE=1` on `gemma4-generate`, or
`Gemma4Runtime::generate_greedy_speculative`. Single-node non-MoE rows only
(`supports_chunk_forward`); other rows fall back to the greedy loop. Draft window
`CAMELID_GEMMA4_SPEC_DRAFT_TOKENS` (default 8).

## Losslessness (the gate)

`tests/gemma4_spec_decode_parity.rs` asserts spec == greedy token-for-token. On
`gemma-4-E4B-it-Q8_0` and `gemma-4-E4B_q4_0-it` (QAT, Q6_K head), 4 prompts × 64
tokens — novel prose and highly repetitive text — **all identical**. The drafter and
acceptance logic also have unit tests (`src/inference/speculative.rs`).

## Speed (M4, E4B-It Q8_0)

| Prompt kind | greedy | spec | tokens/verify pass |
| --- | --- | --- | --- |
| Repetitive ("quick brown fox…") | 2.55 tok/s | **5.84 tok/s** (~2.3×) | 4.11 |
| Novel ("explain relativity") | 3.80 tok/s | ~comparable | 1.02 (no drafts hit) |

The win scales with draft acceptance: repetitive/structured output (lists, code,
boilerplate, repeated phrasing) accepts long draft runs; novel prose finds no n-gram
match and runs at greedy speed (`step_chunk` with K=1). No regression either way, and
output is identical.

## Scope / not done

- CPU single-node non-MoE rows only. The GPU-resident path and distributed/MoE rows
  keep the plain per-token loop (no `step_chunk` there yet).
- A model drafter (small second model) is wired for Llama but not gemma4; n-gram is
  the zero-weight win and the natural first lane.

## Reproduce

```
M=/path/to/gemma-4-E4B-it-Q8_0.gguf
cat "$M" >/dev/null                                  # warm the page cache
camelid gemma4-generate "$M" --prompt P --max-tokens 80
CAMELID_GEMMA4_SPEC_DECODE=1 CAMELID_GEMMA4_SPEC_TIMING=1 \
  camelid gemma4-generate "$M" --prompt P --max-tokens 80
# losslessness:
CAMELID_GEMMA4_GGUF="$M" cargo test --release --test gemma4_spec_decode_parity -- --nocapture
```
