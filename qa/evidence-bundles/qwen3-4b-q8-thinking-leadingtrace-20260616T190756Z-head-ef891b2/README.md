# Qwen3-4B Q8_0 — thinking mode (opt-in, leading-trace lane)

This bundle documents Qwen3 **thinking mode** for the 4B Q8_0 row, enabled
via the request field `camelid_enable_thinking: true` (also surfaced as
`camelid serve --enable-thinking` / `camelid chat --enable-thinking` and the WebUI
**Thinking** toggle). When set, the ChatML renderer emits a bare
`<|im_start|>assistant\n` generation turn (no pre-filled `<think></think>` block),
so the model generates its own `<think>…</think>` reasoning before answering.
Default / `false` keeps the deterministic, parity-locked **thinking-DISABLED**
mode.

**Support scope: `thinking_opt_in_leading_trace_only`.** This is NOT a parity-locked
exact row. The most this bundle claims is *leading-trace parity*.

## What is proven

- **Thinking engages correctly.** For every probe, both camelid and the llama.cpp
  reference begin their generation with `<think>` — the renderer drives the model
  into its reasoning mode exactly as the reference template does
  (`all_probes_engage_thinking = true`).
- **Leading-trace parity.** Greedy generation is **token-identical to the llama.cpp
  reference for the leading reasoning trace** — an identical-prefix envelope of
  **35–235 tokens** across the four probes:
  - `What is the capital of France?` — 235 tokens
  - `Name a primary color.` — 95 tokens
  - `What is 2+2?` — 35 tokens
  - `Say hello.` — 116 tokens
  All four probes match through at least the first 5 tokens
  (`all_probes_match_at_5 = true`).
- **Template-shape byte parity** (separate, stronger guarantee): the
  `enable_thinking=true` rendering is byte-locked by
  `qa/prompt-packs/qwen3-chatml-thinking-template-pack-v1.json` and was
  cross-checked against the same llama.cpp build's `--jinja /apply-template`.

## What is NOT claimed

Full token-parity over an entire thinking trace. Qwen3 reasoning traces run
hundreds of tokens, so they reliably reach the documented **f32-accumulation
frontier** — the same class of near-tie that bounds every camelid row. The
divergence is benign (a synonym near-tie in the logits, coherent reasoning on
both sides up to the divergence point), not a defect.

**The parity-locked exact-row support mode remains thinking-DISABLED.** Thinking
mode is offered as an opt-in feature with the leading-trace envelope above.

## Comparator & path

- Reference: `llama.cpp b9430 (d48a56eff)`,
  `-ngl 0 -ctk f32 -ctv f32 -fa off --no-repack -c 16384`, fed the identical bare-assistant thinking-enabled
  ChatML prompt via `/completion`.
- camelid: `cpu_reference (CAMELID_METAL_RESIDENT_DECODE=0 CAMELID_METAL_RESIDENT_PREFILL=0)` — the f32 path matching the reference's numeric math —
  via `/v1/chat/completions` with `camelid_enable_thinking:true`.

## Reproduce

```
# reference (one server at a time; an 8B Q8 reference + an 8B camelid forward do
# not co-reside in 16 GB):
llama-server -m <gguf> -ngl 0 -ctk f32 -ctv f32 -fa off --no-repack -c 16384 --port 8090
node scripts/thinking-leadingtrace-qwen3.mjs --mode reference --llama http://127.0.0.1:8090 --out ref.json
# camelid:
CAMELID_METAL_RESIDENT_DECODE=0 CAMELID_METAL_RESIDENT_PREFILL=0 camelid serve --addr 127.0.0.1:8185 --model <gguf> --no-open
node scripts/thinking-leadingtrace-qwen3.mjs --mode camelid --camelid http://127.0.0.1:8185 --model-id "Qwen3 4B Instruct" --out cam.json
# compare:
node scripts/thinking-leadingtrace-qwen3.mjs --mode compare --ref ref.json --cam cam.json --row-id qwen3_4b_instruct_q8_0 --display-name "Qwen3 4B Instruct Q8_0" --bundle-out qwen3-4b-thinking-leadingtrace.json
```
