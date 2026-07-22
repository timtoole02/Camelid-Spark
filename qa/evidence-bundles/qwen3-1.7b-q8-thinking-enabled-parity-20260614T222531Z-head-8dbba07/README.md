# Qwen3-1.7B Q8_0 — thinking-enabled mode (opt-in)

This bundle documents Qwen3 **thinking mode**, enabled via the request field
`camelid_enable_thinking: true` (mirroring the existing gemma4 field). When set,
the ChatML renderer emits the template's thinking generation prompt — a bare
`<|im_start|>assistant\n` turn with no pre-filled `<think></think>` block — so the
model generates its own `<think>…</think>` reasoning before answering. Default /
`false` keeps the deterministic, parity-locked **thinking-disabled** mode.

## What is proven

- **Thinking engages correctly.** For every probe, both camelid and the pinned
  llama.cpp reference begin their generation with `<think>` — the renderer drives
  the model into its reasoning mode exactly as the reference template does.
- **Leading-trace parity.** Greedy generation is **token-identical to the pinned
  llama.cpp f32 reference for the leading reasoning trace** — an identical-prefix
  envelope of **26–205 tokens** across four probes (per-probe counts in the
  artifact). All four probes match through at least the first 5 tokens; two match
  through ≥50, one through ≥100.

## What is NOT claimed

Full token-parity over an entire thinking trace. Qwen3-1.7B reasoning traces run
hundreds of tokens, so they reliably reach the documented **f32-accumulation
frontier** — the same class of near-tie that bounds every camelid row (e.g.
TinyLlama's ≤5-token envelope). The divergence is benign: for
"What is the capital of France?" the first difference is at token 73, after
`" largest city and has a lot of historical"`, where camelid picks `" sites"`
(6594) and the reference picks `" landmarks"` (59924) — a synonym near-tie in the
logits, not a defect. The forward pass is correct (73 identical tokens of coherent
reasoning precede it).

**The parity-locked exact-row support mode remains thinking-disabled.** Thinking
mode is offered as an opt-in feature with the leading-trace envelope above.

## Comparator & path

llama.cpp `1 (5d56eff)`, `-ngl 0 -ctk f32 -ctv f32 -fa off --no-repack -c 16384`,
fed the identical thinking-enabled ChatML prompt via `/completion`. camelid runs
the **cpu_reference** path (`CAMELID_METAL_RESIDENT_*=0`) — the f32 path that
matches the reference's numeric math — via `/v1/chat/completions` with
`camelid_enable_thinking:true`.

## Reproduce

```
# reference: feed the thinking-enabled prompt (bare assistant turn) to /completion
#   <|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n
# camelid:
camelid serve --addr 127.0.0.1:8186 --model Qwen3-1.7B-Q8_0.gguf
curl /v1/chat/completions -d '{"model":"Qwen3 1.7B Instruct","messages":[{"role":"user","content":"…"}],
  "temperature":0,"top_k":1,"seed":0,"camelid_enable_thinking":true,"max_tokens":256}'
# compare generated token ids (re-encode camelid text with parse_special:true so
# <think>/</think> map to their single ids) against the reference's tokens.
```
