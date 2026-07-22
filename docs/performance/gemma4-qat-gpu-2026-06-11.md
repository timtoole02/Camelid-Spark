# Gemma 4 QAT on the GPU — parity + speed (2026-06-11)

> [!NOTE]
> Evidence for lifting the gemma4 GPU-resident gate to QAT models. Measured on an
> Apple M4 (10-core GPU, 16 GB). Model on the external T7 (USB); all numbers are
> **warm** (page cache primed) — the cold first run faults 5 GB off USB and is not
> representative (see the cold-cache caveat below).

## What shipped

The GPU-resident gemma4 decode path previously failed closed on anything but
Q8_0 weights. It now runs **QAT models** via a hybrid lane:

- **Layer projections (Q4_0)** run on the GPU — the parity-gated Q4_0 wire GEMV
  dispatched through `encode_gemma4_matmul` (kernels: PRs #243/#244/#245; resident
  routing: #246).
- **Tied head (Q6_K)** runs on the **CPU** — `forward_token_hidden` returns the
  final hidden state, then the CPU does `rms_norm → Q6_K logits matvec →
  final_logit_softcap`, identical to the CPU runtime's head. Q6_K has no GPU
  kernel and the head is a single cheap op (~7.7 ms/step here), so this costs
  little and avoids a Q6_K GPU kernel.
- **Trimmed shared-KV exports** (E4B QAT omits `attn_k`/`attn_k_norm`/`attn_v` on
  non-owning layers) are tolerated: those layers project no K/V and read the
  source layer's cache, so never-read placeholders keep the layer shape uniform.
  A KV-owning layer that omits them is still a hard error (fail closed).

The all-Q8 GPU path is unchanged (head still encoded on the GPU).

## Token parity — GPU hybrid == CPU (the gate)

Model: `gemma-4-E4B_q4_0-it.gguf` (layers Q4_0, `token_embd`/`per_layer_token_embd`
Q6_K). Greedy, 32 new tokens, three prompts. `gemma4-generate` (CPU) vs
`CAMELID_GEMMA4_GPU=1 gemma4-generate-gpu` (hybrid):

| Prompt | Result |
| --- | --- |
| "The capital of France is" | identical token IDs |
| "Once upon a time" (32 tokens) | identical token IDs |
| "Q: What is 2+2? A:" | identical token IDs |

Regression check — `gemma-4-E4B-it-Q8_0.gguf` (all-Q8), "Once upon a time", 24
tokens: GPU == CPU, identical token IDs. The Q8 lane is unaffected.

## Speed (warm)

`gemma-4-E4B_q4_0-it.gguf`, prompt "Write a short story about a robot learning to
paint.", 64 new tokens, warm:

| Runtime | tok/s |
| --- | --- |
| CPU (`gemma4-generate`) | 12.22 |
| GPU hybrid (`gemma4-generate-gpu`) | **15.23** |

≈ **+25 %** decode throughput, token-for-token identical to CPU. Per-step (32-token
run, `CAMELID_GEMMA4_*_TIMING=1`): CPU step ≈ 67.8 ms (attention 12.9 + ffn/ple
45.1 + head 7.7 + embed 2.2); GPU forward ≈ 49 ms (GPU layers + readback + CPU
head) + 2.3 ms CPU prep.

## Cold-cache caveat (do not profile cold)

The **first** GPU run measured 4.26 tok/s — slower than CPU. That was cold page
cache: the 5 GB model faulting off USB on the first GPU forward, not compute. The
immediate warm re-run was 15.23 tok/s. Always warm-run before profiling gemma4 on
the T7.

## Not yet done

- A GPU Q6_K head kernel would remove the per-token hidden readback + CPU head;
  marginal here (head ≈ 7.7 ms) but cleaner.
- The hybrid still uploads the (unused) Q6_K `token_embd` into a GPU buffer in
  `Gemma4ResidentModel::new`; making that buffer optional would save ~0.5 GB on
  the QAT lane.

## Reproduce

```
M=/path/to/gemma-4-E4B_q4_0-it.gguf
cat "$M" >/dev/null                          # warm the page cache
camelid gemma4-generate "$M" --prompt P --max-tokens 64
CAMELID_GEMMA4_GPU=1 camelid gemma4-generate-gpu "$M" --prompt P --max-tokens 64
# add CAMELID_GEMMA4_GPU_TIMING=1 / CAMELID_GEMMA4_CPU_TIMING=1 for per-step splits
```
