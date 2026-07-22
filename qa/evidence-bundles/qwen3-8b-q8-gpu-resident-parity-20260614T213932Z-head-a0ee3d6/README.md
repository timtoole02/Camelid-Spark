# Qwen3-8B Q8_0 — GPU-resident path parity evidence (exact row)

This bundle proves the **GPU-resident decode + prefill path** runs the Qwen3-8B
Instruct Q8_0 exact row correctly — including its **untied output embeddings**
(the first Qwen3 row with a separate `output.weight`) and the per-head QK-norm —
meeting the row's support bar on the resident path.

## What is proven (and nothing more)

**Resident == llama.cpp on the support bar** (3 fixed ChatML thinking-disabled
prompts at 1, 5, and 50 tokens): the same camelid binary with the Metal resident
decode+prefill enabled reproduces, token-and-text identical, the recorded
llama.cpp `reference_content_tokens` from the merged `qwen3-8b-q8-chatml-parity-*`
bundle (identical GGUF, sha256 `408b9555…`). This is the exact bar the 8B row was
promoted on — now met on the GPU-resident path. Result: **all_pass = true** (see
`qwen3-8b-gpu-resident-parity.json`).

## Large context (demonstrative, not a token-parity claim)

A single user turn whose rendered ChatML prompt tokenizes to **7,982 tokens** runs
end-to-end on the resident path and produces coherent output. The resident decode
trace confirms **single-shot prefill**: positions 0..7,980 are filled in one pass
and decode begins at pos 7,981 (no per-token seeding below the prompt length), so
the 16,384-token single-shot resident prefill path is exercised for 8B.

A token-parity *reference* at 8B large-context is intentionally omitted: the f32
CPU reference prefill for an 8B model runs at roughly **8 minutes per 1k tokens**
on this hardware, which makes a same-box comparison at multi-thousand-token
contexts impractical. The correctness basis for large-context resident prefill is:
(1) the resident prefill **kernel** is proven **bit-exact vs llama.cpp at 15,373
tokens** on Qwen3-1.7B (sibling bundle `qwen3-1.7b-q8-gpu-resident-bigctx-parity-*`),
the same code path; and (2) the 8B-specific dims and untied output projection are
covered by the support-bar parity above.

## Context ceilings

- Single-shot GPU-resident prefill: **16,384 tokens**.
- KV context: **40,960 tokens** (the GGUF's native `context_length`).

## Environment

Captured on a clean Mac mini M4 (16 GB, `mini2`) with the model on the fast
internal SSD. 8B Q8_0 resident weights load as evictable page-aligned wire pages
(NOCOPY default); the GPU reads them in place.

## What is NOT claimed

- Other Qwen3 sizes/variants/quants, Qwen3-MoE, thinking-mode generation, context
  above the ceilings, token-parity at 8B large-context, or a resident-prefill
  throughput/perf number.

## Reproduce

```
# resident:
CAMELID_METAL_RESIDENT_DECODE=1 CAMELID_METAL_RESIDENT_PREFILL=1 \
  camelid serve --addr 127.0.0.1:8188 --model Qwen3-8B-Q8_0.gguf
# support bar: compare resident chat generations (3 prompts, 1/5/50 tokens) to the
# recorded llama.cpp reference_content_tokens in
# qa/evidence-bundles/qwen3-8b-q8-chatml-parity-*/qwen3-8b-chatml-chat-parity.json
# (same GGUF sha256 408b9555…).
```
