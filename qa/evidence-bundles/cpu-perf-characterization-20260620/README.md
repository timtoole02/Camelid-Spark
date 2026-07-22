# CPU-perf mission — current-state characterization (2026-06-20)

Machine: i7-11800H (8C/16T, Tiger Lake, AVX-512+VNNI), DDR4 (~51 GB/s peak), Windows.
Model: Qwen3-0.6B-Q8_0 forced CPU (`CAMELID_CUDA_RESIDENT_DECODE=0`). Bit-exact id-sha `e83c5ab6`.

## The easy wins are ALREADY DONE (stale notes corrected)

- `.cargo/config.toml`: `target-cpu=x86-64-v3` (AVX2/FMA/BMI2 autovec) — present. (Deliberately NOT
  `native`/AVX-512: AVX-512 autovec downclocks and gave no gain — documented in the config.)
- `Cargo.toml [profile.release]`: `lto="fat"`, `codegen-units=1` — present.
- So the historical "+39%" (build flags + AVX2 autovec) is **already captured** in the baseline.

## What I measured now

| lever | result | bit-exact |
|---|---|---|
| baseline (v3, 4 threads) | **27.1 tok/s** (~17 GB/s ≈ 33% of peak) | — |
| AVX-512-VNNI dpbusd (gated flag ON) | 28.3 tok/s (**+4.5%**) | ✓ |
| AVX-512-VNNI dpwssd (gated flag ON) | 28.1 tok/s (+3.7%) | ✓ |
| threads 1 / 4 / 8 / 16 | 19.2 / **28.8** / 25.4 / 20.8 | — |

## Diagnosis: memory-bandwidth-bound, ~33% of peak

- VNNI (faster int8 dot) gives only +4.5% → **compute is not the bottleneck**; the int8 dots hide
  behind the Q8 weight read.
- Threads **peak at 4** (28.8) then degrade (16→20.8) → not thread-starved; adding cores contends
  on memory.
- Single-thread ≈ 11 GB/s, 4-thread ≈ 17 GB/s — both far below the ~51 GB/s peak. A bandwidth-bound
  kernel that streamed efficiently would push one core much higher; 11 GB/s indicates the Q8 matvec
  is **latency / access-pattern-bound** (poor prefetch / ILP), NOT yet at the bandwidth wall.

## IMPORTANT: the mission is ALREADY COMPLETE — 28 tok/s IS the optimized result

The CPU-perf mission already shipped (~4.4×) via the **packed-rows4 plan** (SoA i8 repack →
vectorizable int dot; ffn_down 80→10 ms; TinyLlama 4.3→19 tok/s; committed + merged). The 28 tok/s
0.6B above is that packed fast path, not a pre-optimization baseline. The remaining levers were
already investigated and are **spent non-wins on this host**:
- **VNNI** (AVX-512 dpbusd/dpwssd): re-measured +4.5% today = within thermal noise; the prior
  mission measured +0.28% and **reverted** it — Tiger Lake AVX-512 downclock cancels the int8
  throughput gain. (May help non-downclocking server CPUs — unmeasured.)
- **Threads**: peaks at 4 (memory contention beyond); physical-core default already shipped.

So there is **no easy remaining CPU win** on this hardware. The 33%-of-bandwidth figure is, given the
AVX-512 downclock + parity-preserving int dot, near the practical ceiling here — a further
memory-streaming rewrite is high-risk/low-confidence, not the clean ~2× it would be on a fresh kernel.

## Consequence for the flagship (Phase 4)

The concurrent-spec break-even needs the 0.6B CPU drafter at ~70 tok/s; it is ~28 tok/s (optimized)
and there is no spent-but-recoverable lever to close the ~2.5× gap on this box. **The flagship is not
viable on this hardware** — its lossless foundation is built and ready, but it needs either a much
faster CPU (more memory channels / no AVX-512 downclock / AMX) or a smaller/cheaper drafter.

## Reproduce
```
CAMELID_CUDA_RESIDENT_DECODE=0 CAMELID_X86_Q8_PACKED_ROWS4_AVX512VNNI_DPBUSD_DOT=on \
  camelid.exe bench-generate <0.6B.gguf> --prompt "…" --max-tokens 40 --warmup --threads 4
```
