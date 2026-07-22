# Performance findings (2026-06-11) — where the speed actually is

> [!NOTE]
> Honest snapshot from an overnight perf pass on an Apple M4 (10-core GPU,
> 16 GB). Conclusions are evidence-backed; benchmark numbers trace to committed
> bundles. No claim is made beyond what was measured.

## TL;DR

- **The CPU quant dot kernels are already at speed.** A direct micro-benchmark
  (`perf_q4_q8_q6_dot`) shows Q4_0, Q8_0, and Q6_K row-dots within ~1.3× of each
  other (Q4_0 ≈ Q8_0; Q6_K ~0.7–1.3×). There is **no slow-kernel bug**.
- **The supported Q8_0 rows are at the hardware wall.** Llama 3.2 3B Q8_0,
  same-host vs llama.cpp (Metal) and MLX-LM: prefill **587.3 vs 543.7 vs 577.9
  tok/s** (Camelid fastest), decode **29.7 vs 29.1 vs 29.1 tok/s** (Camelid
  fastest by a hair). Margins are narrow; this is the ~120 GB/s unified-memory
  wall, where everyone lands. See [`BENCHMARKS.md`](../benchmarks/BENCHMARKS.md).
- **"Slow" QAT decode was cold cache, not compute.** Gemma 4 E4B QAT Q4_0
  (5 GB on the external T7/USB) measured **0.02 tok/s cold** but **1.34 tok/s on
  the immediate warm re-run** — a 67× swing purely from OS page cache. Warm
  per-step is ffn+ple ~164 ms, attention ~11 ms, Q6_K head ~9 ms. The cold cost
  is reading the model from USB the first time, not the engine.

## What that means for "make it faster"

CPU decode is **memory-bandwidth-bound**, and a few CPU cores cannot saturate
the full unified bandwidth the way the GPU can. The genuine levers, in order of
honest payoff, are therefore not kernel micro-opts:

1. **GPU-resident decode for every row.** The Q8_0 gemma4 path already runs
   fully resident on Metal (~120 GB/s wall). The QAT (Q4_0 experts / Q6_K head)
   and distributed rows have **no GPU kernels yet** — building Q4_0/Q6_K Metal
   GEMV (mirroring the proven Q8 wire-nocopy path, parity-tested vs CPU) is the
   largest real win for those rows. Substantial, parity-gated.
2. **Speculative decode — "read less per token."** At the bandwidth wall the only
   way past it is to read fewer weight bytes per accepted token. Camelid's
   speculative path is byte-exact-proven and already shows ~+11–12% in draft
   mode; widening its coverage is the honest decode win for the supported rows.
3. **Cold-start residency.** First-run latency on USB-resident models is a real
   UX cost (the 67× above). A background page-cache warmer at load would make the
   first decode match the warm decode for models that fit RAM. Parity-neutral
   (pure prefetch), but it does not move the warm/steady-state wall.

## What is NOT a win (measured, ruled out)

- Rewriting the Q4_0/Q6_K CPU dots — already Q8-class.
- "Fixing" QAT decode speed in code — the cold number was a USB page-cache
  artifact, reproducible-away with a warm run.

## Methodology

`perf_q4_q8_q6_dot` (in `src/inference` tests, `#[ignore]`d) times each wire
row-dot in isolation. Gemma 4 phase timing via `CAMELID_GEMMA4_CPU_TIMING=1`.
Same-host 3B comparison is the committed bundle behind `BENCHMARKS.md`.
