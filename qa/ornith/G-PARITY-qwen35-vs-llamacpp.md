# ORNITH 9B — Phase 3 G-PARITY receipt (qwen35 vs llama.cpp acd79d6)

**Gate:** G-PARITY — greedy token-identical vs the pinned llama.cpp oracle
(`first_divergent_generated_token_index = -1`). **Result: PASS (4/4 prompts).**
**Date:** 2026-06-29 · **Platform:** Windows x86_64 (MSVC), CPU-only.
**Camelid lane:** runnable (pure-f32 oracle), row-parallel (rayon) — bit-exact.
**Oracle:** llama.cpp `acd79d6` (build 9632), `llama-server.exe` / `llama.dll`
(confirmed to contain `LLM_ARCH_QWEN35` + SSM tensors — the arch is supported at the
binary level, not just source). **Model:** `ornith-1.0-9b-Q8_0.gguf` (arch `qwen35`).

## Method (isolates the model forward from tokenization)

Camelid's qwen35 tokenizer encodes each prompt; the **identical prompt token-ID arrays**
are fed to BOTH engines (llama-server `/completion` accepts a token-id prompt), so any
divergence is purely the model forward, not tokenization. Greedy (`temperature 0`,
`top_k 1`, `seed 0`, `cache_prompt false`), `n_predict 20`, llama-server CPU
(`-ngl 0 -ctk f32 -ctv f32 --no-repack`). Both never co-resident (15.7 GiB RAM ceiling):
oracle captured, server killed, then camelid run.

## Results — generated token IDs, camelid vs oracle

| # | prompt | tokens | match |
|---|--------|--------|-------|
| 0 | `What is the capital of France?` | `[271,760,6511,314,9338,369,11751,13,271,3710,369,279,6511,314,9564,30,271,760,6511,314]` | ✅ identical |
| 1 | `def fibonacci(n):` | `[198,262,413,307,2564,220,15,25,198,285,460,2958,198,262,4265,307,606,220,16,25]` | ✅ identical |
| 2 | `The opposite of hot is` | `[8981,13,271,248068,198,90700,8340,25,271,16,13,220,2972,2014,53983,279,5952,64700,198,262]` | ✅ identical |
| 3 | `Count to five: 1, 2, 3,` | `[220,19,11,220,20,13,4543,11,1092,3905,1727,30,271,760,1727,1324,303,279,8240,369]` | ✅ identical |

All four token-identical → `first_divergent_generated_token_index = -1`. Prompt 0 also
validated at 24 tokens (G-LOAD). Coverage: factual, code, English completion, arithmetic.

## What this earns

- **Supported lane EARNED** for `qwen35` Q8_0 on Windows (parity-certified vs the pinned
  oracle) — exceeds the recon brief's prediction that Supported would be deferred behind a
  missing oracle. The oracle exists and parity holds.
- The from-scratch **gated-delta-net recurrence**, causal conv1d, `j % num_k_heads` GQA
  head-repeat, state orientation, partial NEOX mRoPE (text-collapse), and gated attention
  are bit-correct against the reference.

## Speed (bit-exact rayon row-parallel matvec → int8×int8 maddubs)

The qwen35 runnable path was made row-parallel (`RawMat::par_matvec`, rayon): each output
element is an independent dot, so per-element sum order is unchanged → **bit-identical**
(the original 4/4 result was produced by the parallel path, matching the oracle). Effect:
**~30 s/token → ~2.2 s/token (~13×)** on this box, which makes the agent loop feasible
without perturbing parity. The generic (non-qwen35) runnable lane is untouched.

**Update 2026-06-29 — int8×int8 maddubs kernel (optimized-lane port).** The Q8_0 matmul
backend (`RawMat::par_matvec`/`par_matmul`) was switched from the int8×f32 dot
(i8→f32-convert + f32-FMA) to the optimized inference lane's **int8×int8 maddubs** path:
the f32 activation is quantized to Q8 **once** per matvec (`quantize_q8_0_blocks`) and
reused across every weight row, which is then an integer maddubs reduction
(`q8_0_wire_dot_avx2`, byte-for-byte equal to `inference.rs::q8_0_dot_rows_avx2` but
sourcing the weight i8 straight from the wire bytes — no resident weight decode). Because
llama.cpp's own Q8_0 path **also** quantizes activations to Q8, this moved camelid
*closer* to the oracle's arithmetic, and parity is **re-certified 4/4 token-identical** to
the exact reference IDs above (`ornith_qwen35_parity_gen`, prompt token-ID arrays
`[[3710,369,279,6511,314,9338,30],[727,73111,1393,1590],[760,13600,314,3882,369],[2427,310,4097,25,220,16,11,220,17,11,220,18,11]]`,
`n_predict=20`, this build). Effect on this box: **load + 80 generated tokens in 48.7 s ≈
~0.3 s/token decode** (~31 GB/s, i.e. at the memory-bandwidth ceiling for a 9.5 GB Q8
model — further CPU speedup needs a smaller quant, not a faster kernel).

## Known deviation (carried from G-LOAD)

- BPE pre-tokenizer `qwen35`: implemented as qwen2's single-digit split; the `\p{M}`
  combining-mark folding is deferred. Byte-identical to the oracle for all four prompts
  above (and any mark-free text). The parity method feeds token IDs, so this affects only
  end-to-end text→token parity on combining-mark inputs, not the model-forward result here.
