# Qwen3-1.7B Q8_0 — GPU-resident path parity evidence (exact row)

This bundle proves the **GPU-resident decode + prefill path** runs the Qwen3-1.7B
Instruct Q8_0 exact row correctly: it applies the per-head QK-norm Qwen3 requires
and produces **token-and-text-identical** greedy output to the already-proven
references, including at a **large (15,373-token) single-shot prefill context**.

## Why this work exists

The GPU-resident decode/prefill paths previously bailed to the CPU plan whenever a
model carried `attn_q_norm` / `attn_k_norm` (Qwen3's per-head RMSNorm on Q and K,
applied after projection and before RoPE), because the resident kernels did not
apply it. That kept Qwen3 off the GPU-resident path — and the CPU prefill is too
slow to make large contexts practical. This change wires the per-head norm into
both resident paths (using the existing `rms_norm_per_head_f32` kernel) so Qwen3
runs resident, which is what makes large-context prefill usable.

## What is proven (and nothing more)

1. **Resident == cpu_reference** (small context, 5 chat prompts, 64 tokens each):
   the same camelid binary with the Metal resident decode+prefill enabled is
   token-and-text identical to the same binary with them disabled. The
   cpu_reference path is itself proven token-and-text identical to llama.cpp
   (see the sibling `qwen3-1.7b-q8-chatml-parity-*` bundle), so this is a
   transitive proof of resident correctness.
2. **Resident == llama.cpp at 15,373-token context** (single-shot prefill, 24
   tokens): a single user turn whose rendered ChatML prompt tokenizes to 15,373
   tokens — just under the 16,384 single-shot resident-prefill ceiling — produces
   output token-and-text identical to the pinned llama.cpp reference. The resident
   decode trace confirms prefill filled positions 0..15,371 in one shot (decode
   begins at pos 15,372; no per-token seeding below the prompt length).

Result: **all_pass = true** (see `qwen3-gpu-resident-parity.json`).

## Context ceilings

- **Single-shot GPU-resident prefill: 16,384 tokens** (a prompt above this falls
  back to the CPU plan).
- **KV context: 40,960 tokens** (the GGUF's native `context_length`); generation
  continues resident from the prefilled cache up to this ceiling.
- On a 16 GB host the f32 KV cache at 40,960 positions is ~9.4 GB, so the full
  ceiling is reachable with the 1.7B model alone but leaves little headroom; the
  parity above was captured at 15,373 tokens, the largest single-shot prefill
  that fits alongside the llama.cpp oracle for a same-box comparison.

## Numeric note

When QK-norm is present, the resident prefill **forces the f32 activation path**
(the tiled-mm / half-precision / attention-as-matmul lanes are disabled) so Q and
K stay materialized as f32 `[token][head][head_dim]` and the per-head norm is
bit-for-bit the cpu_reference order. This is slower than the mm prefill path but
is token-exact; correctness over throughput for this arch.

## What is NOT claimed

- Other Qwen3 sizes, base variants, other quants, Qwen3-MoE, or **thinking-mode**
  generation. Those are separate rows and fail closed.
- A throughput/perf claim for the resident Qwen3 prefill (it runs the f32 path).

## Reproduce

```
# reference (pinned llama.cpp build dir, DYLD_LIBRARY_PATH set to that bin dir):
llama-server -m Qwen3-1.7B-Q8_0.gguf -c 16384 --port 8090 -ngl 0 \
  -ctk f32 -ctv f32 -fa off --no-repack
# camelid, resident on:
CAMELID_METAL_RESIDENT_DECODE=1 CAMELID_METAL_RESIDENT_PREFILL=1 \
  camelid serve --addr 127.0.0.1:8186 --model Qwen3-1.7B-Q8_0.gguf
# camelid, resident off (cpu_reference baseline):
CAMELID_METAL_RESIDENT_DECODE=0 CAMELID_METAL_RESIDENT_PREFILL=0 \
  camelid serve --addr 127.0.0.1:8187 --model Qwen3-1.7B-Q8_0.gguf
# then compare greedy chat completions across the two camelid endpoints, and the
# resident endpoint against llama.cpp /completion fed the identical ChatML string.
```
