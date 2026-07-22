# Determinism Baseline — Pillar One, Phase 0

**Scope:** TinyLlama 1.1B Chat Q8_0, CPU forward pass only. Measure-before-build.
**Status:** Phase 0 complete. STOP-and-report gate. No engine code changed.

## Environment
- Commit: `617a34bb7dce3a872e4a39ef33869bc0b324d381` (branch `feat/deterministic-cpu`, off `main`)
- Host: Apple M4, 10 cores, `FEAT_I8MM=1` (so the CPU Q8_0 path uses the i8mm packed-rows4 kernel)
- Model: `tinyllama-1.1b-chat-v1.0.Q8_0.gguf` (sha256 `a4c9bb1dbaa372f6…`)
- Binary: release, `bench-generate`, prompt `"hello"` (2 prompt tokens, BOS), 50 generated tokens, greedy (temperature 0)
- Date: 2026-06-14T06:34:55Z
- Credit: Camelid's CPU Q8_0 kernels and the parity contract they are gated against follow llama.cpp's reference reduction layout (block-wise Q8_0 dot, per-output accumulation).

## Q1 — Is the TinyLlama Q8_0 hot path single- or multi-threaded? Identify each reduction site.

The CPU forward pass is **multi-threaded via rayon**. There are three reduction-bearing sites, all in `src/inference.rs`:

| Site | What it reduces | Reduction kernel (fn @ line) | Parallelism |
|---|---|---|---|
| **1. Matmul / linear projections** (Q/K/V, attn-out, FFN gate/up/down) | Q8_0 weight row · f32 activation, over the K (contraction) dim | `q8_0_packed_rows4_dot_i8_matmul` @16416 → `q8_0_packed_rows4_dot` @16611 (i8mm/NEON/AVX/scalar); f32 fallback `dot_product_row` @17019 | `par_chunks_mut` over **output** columns/rows (e.g. @6303, @10545/10557, @11190, @11311, @16081) |
| **2. Attention** | QKᵀ scores (per cached position), softmax sum, Σ score·V | `attention_context_for_head_into` @17650 (QKᵀ via `dot_product_row`; softmax serial; value accum `*out += prob * v`) | batched prefill: `par_chunks_mut` over **output** rows @17625; single-token decode is serial per head |
| **3. Final logit projection** (lm_head) | Q8_0 weight row · f32 hidden, over K | same kernels as Site 1 (`output_projection_runtime_with_plan` @6066 → packed-rows4 / fallback) | `par_chunks_mut` over **output** (vocab) columns |

Entry path for one decode step: `generate_next_token_with_history_diagnostics` @2897 → `forward_single_token_timed_internal` @2652 → `forward_layer_timed` @4207 (×layers) → the kernels above → `forward_final_norm_and_logits` @2248.

## Q2 — Is accumulation order currently fixed, or does it depend on thread/SIMD scheduling?

**It is already fixed.** At every site the rayon parallelism partitions the **output** space: each thread owns a disjoint set of output elements and performs that output's *entire* K-dimension reduction serially, in a fixed block order, within one thread. There is:

- **No** cross-thread float combine — no `par_iter().sum()/reduce()/fold()` over floats (audited: zero occurrences in `src/inference.rs` + `src/inference/*.rs`).
- **No** atomic-float accumulation, no `Mutex<f32>` accumulator, no tree/pairwise merge of per-thread partials.
- A **compile-time-fixed** SIMD horizontal-sum order (i8mm `q8_0_packed_4x8_*` lanes accumulate into a fixed `sums[0..4]` layout); SIMD does not "schedule," so lane order is invariant on a given binary.

Therefore thread scheduling and rayon work-stealing can only change *which* thread computes *which* output — never the order of additions inside any one output's reduction. Chunk boundaries always fall on whole-output boundaries, so chunk size (which can vary with thread count) does not split a reduction.

### Empirical confirmation (default CPU path, no determinism flag, nothing changed)
- 5 in-process iterations → **byte-identical** 50-token streams.
- Fresh process re-run (run A vs run B) → **byte-identical**.
- `--threads 1` vs default (10 threads) → **byte-identical**. This is the decisive test: changing the rayon thread count does not change a single token.

First 8 token ids (all CPU runs): `[29892, 322, 769, 306, 29915, 645, 367, 1250]`.

## Q3 — Baseline tokens/sec (hello, 50 tokens, greedy, median of 5)

| Path | Median tok/s | Per-iteration | Peak RSS |
|---|---|---|---|
| **CPU forward pass** (Metal stack off, `CAMELID_NO_GPU_SAMPLE=1`) | **11.06** | 10.78 / 10.89 / 11.06 / 12.00 / 12.85 | 1.20 GB |
| GPU resident decode (CLI default fast path, for reference) | 84.87 | 84.87 / 83.31 / 84.91 / 83.16 / 85.16 | 1.97 GB |

Note the CLI default (`apply_default_fast_stack`, main.rs:438) turns the Metal resident-decode stack on for every subcommand on macOS, so the *default* `bench-generate` rides the GPU. The 11.06 tok/s figure is the pure-CPU path the deterministic mode will use.

## Q4 — For each reduction site: does order-stability require disabling threading, disabling auto-vectorization, or neither?

**Neither, at all three sites.** Determinism on the CPU path is **already real and free**:
- Site 1 (matmul): output-partitioned, serial K-reduction in fixed block order. Neither.
- Site 2 (attention): output-partitioned (prefill) / serial (decode), fixed-order softmax and value accumulation. Neither.
- Site 3 (logits): same kernel family as Site 1. Neither.

The `--threads 1 == --threads 10` byte-identity proves we do not need to disable threading; the fixed SIMD lane layout proves we do not need to disable auto-vectorization.

## The one real divergence: GPU default ≠ CPU

GPU default and CPU agree for the first **25** generated tokens, then diverge (CPU →`304 1074 366`, GPU →`366 29915 276` at index 25). The Metal path is a *different* numerically-ordered reduction (threadgroup sums), so the GPU default is **not** a bit-reproducible reference. This is the substantive reason a deterministic mode must **force the CPU forward pass** rather than pin the GPU path.

## Cost finding → recommendation for Phase 1

- **Determinism of the CPU computation is already proven and near-free** (run-to-run, cross-process, and thread-count-invariant with zero changes).
- The measured "overhead" of deterministic mode is therefore **not** a determinism penalty inside the CPU path — it is the **CPU-vs-GPU gap**: ~11 tok/s (deterministic CPU) vs ~85 tok/s (default GPU), i.e. deterministic mode is ~7.7× slower because it forgoes the GPU fast path, not because it serializes any reduction.
- **Recommended `--deterministic` semantics:** (1) force the order-stable CPU forward pass (disable the Metal/resident-decode stack for that invocation); (2) keep rayon threading on — it is order-invariant; (3) record the pinned per-site reduction order in DECISIONS.md as the contract a future cross-machine trace depends on. No default path change, no kernel change, no perf change to the default.
- **Caveat for Phase 2 pinned floats:** the exact logit values are machine/ISA-specific (i8mm here on M4; an M1 runner without i8mm takes a different-but-internally-deterministic kernel). The *portable* invariant is run-to-run / thread-count byte-identity; the *committed exact floats* are an M4-i8mm reference and the pin test must be env-gated to the real model (self-skip when absent), matching the repo's existing real-model test convention.

## Decode-time overhead figure (Phase 2 — measured with the flag landed)

`--deterministic` on the same binary (hello → 50 tok, greedy, median of 5, M4):

| Path | Median tok/s | Per-iteration | Peak RSS | Determinism |
|---|---|---|---|---|
| Default (no flag, GPU resident decode) | **88.67** | 88.73 / 88.26 / 88.03 / 88.73 / 88.67 | 1.85 GB | unchanged — token stream byte-identical to the pre-change baseline |
| `--deterministic` (CPU) | **12.51** | 12.65 / 12.51 / 12.30 / 12.59 / 12.29 | 1.15 GB | bit-exact across runs, processes, and `--threads 1` vs `--threads 10`; identical to the bare-CPU stream |

**On-record overhead:** deterministic mode runs at ~14% of default throughput (12.51 vs 88.67 tok/s, ~7.1×) — entirely the cost of forgoing the GPU fast path, with **zero** determinism penalty inside the CPU computation. The default path was re-measured on the post-change binary and is byte-for-byte identical to the pre-change baseline (same 50-token stream, same ~88 tok/s), confirming the flag is inert when off.

Reproduce: `camelid bench-generate <tinyllama-q8> --prompt hello --max-tokens 50 --temperature 0 --iterations 5 --warmup [--deterministic]`. Pinned first-position logits regression: `tests/deterministic_forward.rs` (env-gated on `CAMELID_TINYLLAMA_Q8_GGUF`).
