# Qwen3-1.7B Q8_0 — ChatML chat parity evidence (exact row)

This bundle proves **token-identical greedy parity** between camelid's CPU
reference path and a pinned llama.cpp reference for the **Qwen3-1.7B Instruct
Q8_0** exact row, in **ChatML chat mode with thinking disabled**.

## What is proven (and nothing more)

- One GGUF: `Qwen/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf`
  (sha256 `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a`).
- One quant: Q8_0. One arch: `qwen3` (dense). One mode: ChatML, **thinking
  disabled** (`<think>\n\n</think>\n\n` generation prompt → direct answer).
- Three fixed single-turn user prompts.
- For each prompt, at **1, 5, and 50 generated tokens**:
  - prompt-token parity (camelid-rendered ChatML tokens == reference tokens),
  - generated-token parity (content tokens, trailing EOS/`<|im_end|>` excluded),
  - generated-text parity.
- Result: **all_pass = true** (see `qwen3-chatml-chat-parity.json`).

## Comparator

llama.cpp `1 (5d56eff)` (built 2025-04-28), run as the reference with
`-ngl 0 -ctk f32 -ctv f32 -fa off --no-repack` — the pinned f32 / no-flash-attn /
no-repack configuration that matches camelid's pure-f32 CPU numeric path. The
reference is fed the identical rendered ChatML string via `/completion` (which
parses the `<|im_*|>` and `<think>`/`</think>` specials).

## What is NOT claimed

- Other Qwen3 sizes (0.6B / 4B / 8B / 14B / 32B), base variants, other quants,
  Qwen3-MoE (A3B), longer contexts, or **thinking-mode** generation. These are
  separate, unproven rows and fail closed.
- Raw-completion greedy parity (non-chat) is token-identical at the first token
  for all probe prompts and at 5 tokens for prompts without an f32 near-tie; one
  raw probe hits a documented 0.015-logit near-tie at token 3 (the same
  f32-accumulation frontier as other camelid rows). The supported, parity-locked
  surface is the ChatML thinking-disabled chat path captured here.

## Reproduce

```
# reference (from the pinned llama.cpp build dir, DYLD set):
llama-server -m Qwen3-1.7B-Q8_0.gguf -c 4096 --port 8090 -ngl 0 \
  -ctk f32 -ctv f32 -fa off --no-repack
# camelid:
camelid serve --addr 127.0.0.1:8185 --model Qwen3-1.7B-Q8_0.gguf
# harness:
node scripts/chat-parity-qwen3.mjs \
  --camelid http://127.0.0.1:8185 --llama http://127.0.0.1:8090 \
  --model-id "Qwen3 1.7B Instruct" --out qwen3-chatml-chat-parity.json
```
