# PERF_GAP_REPORT.md — Camelid vs llama.cpp

Blunt, evidence-first. Every number traces to `PERF_RECEIPTS/`. Same model, same GGUF, same Q8_0 quant, same host, same OS, same build mode (both Release), greedy/temp=0.

- **Host:** i7-11800H (8C/16T, Tiger Lake — AVX2+AVX-512+VNNI), ~16 GiB DDR4 (~51 GB/s peak), RTX 3060 Laptop 6 GiB (sm_86), Win11.
- **Camelid** `ce7dceb6` (release: `target-cpu=x86-64-v3`, fat-LTO). **llama.cpp** `acd79d6` (Release; CPU = AVX-512+FMA+tinyBLAS+REPACK+OpenMP; CUDA = FA+graphs, arch 86).
- **Models:** Llama-3.2-3B-Instruct-Q8_0 (primary, shared on disk), Qwen3-4B-Q8_0, Qwen3-0.6B-Q8_0. SHA-256 in `PERF_RECEIPTS/env/model-sha256.txt`.

## 2026-07-08 re-pin — STAMPEDE Phase 0 (CURRENT)

**Pins:** Camelid `2b8b97c4` (main; release build) vs llama.cpp **b9918 `0512ef1e5`** (official
prebuilt win-cpu-x64, Clang 20.1.8 — same provenance as the old pin). Env receipt:
`PERF_RECEIPTS/env/environment-b9918-20260708.md`. Full receipts:
`PERF_RECEIPTS/same-host/stampede-p0-baseline-2b8b97c4-20260708T0715Z/` (cpu-baseline-medN,
REPEATS=5, CUDA_VISIBLE_DEVICES=-1, cache-defeated, greedy; llama-bench cross-check in same dir).

| Model | Quant | Stage | Camelid | llama.cpp | Ratio | was (June pin) | movement |
|---|---|---|---:|---:|---:|---:|---|
| Llama-3.2-3B | Q8_0 | prefill | 22.85 | 51.46 | **0.44×** | 0.80× (23.73/30.6) | llama +68% (repack GEMM); gap WIDENED |
| Llama-3.2-3B | Q8_0 | decode | 7.79 | 8.71 | **0.89×** | 0.66× (5.97/9.08) | Camelid +30% (#362 defaults); llama −4% |
| Qwen3-4B | Q8_0 | prefill | 15.85 | 34.48 | **0.46×** | ≈0.62× | gap widened |
| Qwen3-4B | Q8_0 | decode | 6.43 | 7.67 | **0.84×** | — | close |
| Qwen3-0.6B | Q8_0 | prefill | 110.41 | 263.44 | **0.42×** | — | gap widened |
| Qwen3-0.6B | Q8_0 | decode | 49.5 | 45.89 | **1.08×** | ~0.61× | **Camelid AHEAD** |
| Llama-3.2-3B | Q4_K_M | prefill | 13.72 | 90.76 | **0.15×** | (no prior row) | widest gap in matrix |
| Llama-3.2-3B | Q4_K_M | decode | 12.9 | 15.2 | **0.85×** | — | close |
| Qwen3-4B | Q4_K_M | prefill | 10.42 | 66.41 | **0.16×** | — | K-quant prefill has no owner |
| Qwen3-4B | Q4_K_M | decode | 11.51 | 11.73 | **0.98×** | — | ~parity |

llama-bench cross-check (true CPU, `-t 8 -p 512 -n 128 -r 3`): 3B Q8 pp512 55.38 / tg128 9.34;
3B Q4_K_M pp512 96.14 / tg128 15.16 — server-level numbers track bench within the usual ~5–8%.
Greedy decode text: Camelid ≡ llama on both Llama-3B rows; Qwen3 rows diverge (own chat
templates/near-ties — informational only, not a Camelid parity gate).

**The blunt read at the new pin:**
1. **Decode is nearly closed** (0.84–1.08×). The June "0.66×, no cheap win" framing is obsolete:
   the #362 win-default promotion landed +30% after the report was written. Remaining decode gap
   ≈ one lever (see P0.4 census: GQA QKV decode is single-threaded — `src/inference.rs:13942`,
   ~13.7% of the weight stream on one thread; parallelizing it is parity-safe and modeled ≈ +20%).
2. **Prefill is the campaign.** Uniform 0.42–0.46× on Q8_0 and a brutal 0.15–0.16× on Q4_K_M
   (llama's K-quant repack prefill outruns even its Q8; Camelid's K-quant lane ships a decode
   owner but NO prefill owner). Phase 3 (tiled GEMM owner) is top priority and must cover
   K-quants, not just Q8.
3. **Fork-join is NOT the decode story:** 169 real regions/token × 2.76 µs (hot) ≈ 0.4% of token
   time; even the all-cold bound is 5.5%. Phase 4 (spinning pool) is **KILLED** per its own gate
   (`stampede-p04-region-census-2b8b97c4-20260708.md`).

**Update 2026-07-08 (STAMPEDE Phase 2.0 shipped):** parallelizing the GQA QKV decode branch
(win-x86_64 defaults; flag `CAMELID_X86_Q8_QKV_GQA_PARALLEL_DECODE`, default-on) moved 3B Q8 decode
8.15 → **11.17** (+37%, same-run A/B) and Qwen3-4B Q8 6.43 → **8.56** (+33% vs P0 receipt) —
**Camelid CPU decode is now AHEAD of llama.cpp b9918 on both rows (1.21× / 1.17×)**, greedy text
byte-identical. Decode row above is superseded; receipts
`same-host/stampede-p20-qkv-gqa-{OFF,ON}-*-20260708.json`.

**Update 2026-07-08 (STAMPEDE Phase 3):** (a) the Q8 prefill GEMM owner re-validated at b9918 with
an engaged-checked paired sweep (+12.3% 3B / +11.9% 4B, every CI excluding 1.0) and its default
FLIPPED on win-x86_64 (D15) — 3B Q8 prefill default path is now ~30.3 tok/s (0.55–0.59× of llama);
(b) NEW batched Q4_K prefill owner (`CAMELID_X86_KQUANT_MATMUL_OWNER`, opt-in): 3B Q4_K_M prefill
14.94 → **22.39 (+50%)**, 0.15× → 0.23×, bitwise twin-tested, e2e byte-identical; measurement
harness fix: `bench-owner-sweep` had been silently measuring the first-resolved cached plan for
every config (fake-null trap) — uncached-plan bypass + per-config engaged-check landed, and the
per-record `owner_prefill_taken` counts are now part of every sweep receipt. Q4_K_M residual gap =
Q6_K layers (no owner yet) + the dot itself (repack/AVX-512 escalation staged).

## Headline gap table (2026-06-21 pin `acd79d6` — HISTORICAL, superseded above)

| Model | Quant | HW | Stage | Camelid | llama.cpp | Gap | Suspected cause | Evidence | Proposed fix | Parity risk | Difficulty |
|---|---|---|---|---:|---:|---:|---|---|---|---|---|
| Llama-3.2-3B | Q8_0 | CPU | prefill tok/s | 18.9 | 30.6 | **0.62×** | No tiled AVX-512 GEMM; AVX2 fragmented per-role kernels | `same-host/…cpuonly-nocache…json` | Unified tiled Q8 GEMM owner (§ARCH) | Low (int acc bit-exact) | High |
| Llama-3.2-3B | Q8_0 | CPU | decode tok/s | 5.97 | 9.08 | **0.66×** | Memory-bandwidth bound (~33% peak); llama AVX-512+repack streams better | same + `cpu-perf-characterization-20260620` | None cheap (see below) | — | High |
| Qwen3-4B | Q8_0 | CPU | pp512 / tg128 (llama true CPU) | — | 23.6 / 7.44 | — | (Camelid CPU ratio ≈ 3B's 0.62×) | `llama-bench/cpu-TRUE-cudaHidden.txt` | — | — | — |
| Qwen3-0.6B | Q8_0 | CPU | tg128 | ~28* | 45.8 | ~0.61× | same memory wall | llama-bench (true CPU) + char-20260620 | — | — | — |
| Llama-3.2-3B | Q8_0 | CUDA | decode tok/s | ~53** | 69.3 | ~0.77× | parity-locked attention (no non-bit-exact flash-attn) | `llama-bench/cuda-ngl99.txt` + campaign | (already optimized) | locked | n/a |
| Qwen3-4B | Q8_0 | CUDA | decode low-ctx | 41.6 | 54.4 | **0.77×** | `q8_gemv` ~76% BW; rest parity-locked | `qa/perf/decode-attention-campaign.md` (cross-validated: llama 54.41 measured this session) | none low-risk | locked | n/a |
| Qwen3-4B | Q8_0 | CUDA | decode ctx~1881 | 26.2 | 51.1 | **0.51×** | uncoalesced K/V (coalescing flips near-tie greedy tokens) | campaign | non-bit-exact flash-attn (rejected) | **breaks parity** | n/a |
| Llama-3.2-3B | Q8_0 | CUDA(default) | prefill tok/s | ~905 | (CPU 30.6) | **>2× faster** | Camelid GPU batched prefill (shipped) | `same-host/…162052Z` (GPU-confound run) | — | — | done |

\* Qwen3-0.6B Camelid CPU decode from `cpu-perf-characterization-20260620` (~28 tok/s optimized). \** Llama-3B Camelid CUDA decode extrapolated from the campaign's ratio; a direct measure is a follow-up.

## Commands (exact)

```
# llama.cpp ground truth
llama-bench.exe -m <gguf> -ngl 0  -t 8 -p 512 -n 128 -r 3     # CPU
llama-bench.exe -m <gguf> -ngl 99       -p 512 -n 128 -r 3     # CUDA
# Same-host CPU (CUDA hidden, cache defeated), greedy temp=0:
CUDA_VISIBLE_DEVICES=-1 node docs/perf-deep-dive/scripts/cpu-prefill-matrix.mjs   # llama vs Camelid A vs Camelid B
```
Camelid forced CPU with `CAMELID_CUDA_RESIDENT_DECODE=0 CAMELID_CUDA_RESIDENT_PREFILL=0` and `CUDA_VISIBLE_DEVICES=-1`.

## Methodology traps found (both would have produced false claims)

- **`-ngl 0` is not CPU-only on a CUDA build.** ggml offloads compute-bound prefill matmuls to the GPU even at `ngl0`; llama-bench pp512 (740/545/2585) was GPU-assisted. True CPU prefill (CUDA hidden) is 30.9/23.6/167. Without catching this, the "prefill gap" would have been mis-stated as ~40×.
- **Camelid prompt-prefix-caches by default.** A warmup request caches the prefix, so a naive "prefill" re-request is a cache hit (measured 998 tok/s — impossible for real CPU compute). The harness defeats it with a unique system-message nonce. (llama gets `cache_prompt:false`.)
- Both engines were therefore measured CPU-only (`CUDA_VISIBLE_DEVICES=-1`), cache-defeated, greedy, same prompt.

## The blunt read

1. **CPU is where Camelid is behind, and the gap is uniform ~0.62–0.68× across BOTH prefill and decode** (prefill 18.9 vs 30.6; decode 5.97 vs 9.08). Because it's uniform, it's a **per-kernel-throughput** gap, NOT a prefill-batching problem — Camelid's prefill amortizes ~3.3× over decode, the same as llama's ~3.4×. Cause is architectural: llama.cpp runs ONE tiled tinyBLAS GEMM (AVX-512+FMA+REPACK) with an in-kernel chunk scheduler; Camelid runs AVX2 bespoke per-role kernels. See `LLAMA_CPP_ARCHAEOLOGY.md §1–2`.

2. **The "easy" CPU fixes are dead — proven, not assumed.**
   - Enabling the gated x86 SIMD packed-rows4/GEMM4 kernels (`CAMELID_X86_Q8_*`) **REGRESSES** prefill −11% / decode −8% on this box, byte-identical output. *This vindicates the team's default-off discipline.* (`same-host/…cpuonly-nocache…json`, config B.)
   - Prefill **routing** (layer-major on/off, chunk 64/256/512/all) moves <3% — noise. (`same-host/cpu-prefill-routing-*.json`.)
   - The "+39% AVX2" from the stale note is **already shipped** (`target-cpu=x86-64-v3`+fat-LTO). AVX-512 is deliberately excluded (downclocks Tiger Lake; VNNI gave +0.28–4.5% on memory-bound decode, reverted).

3. **CUDA is near-optimal and the residual is a deliberate correctness choice, not lost performance.** `q8_gemv` is ~76% of DRAM bandwidth; the depth gap needs coalesced K/V reads that re-associate the attention dot and flip near-tie greedy tokens — which Camelid refuses (losslessness). On a GPU box (the default here) Camelid's prefill is excellent (~905 tok/s).

4. **Sampling, SSE/server, tokenizer, mmap/load are NOT bottlenecks** in either engine.

## What's fixable now vs not

- **SHIPPED (P0.6) — phase-adaptive CPU threading, +24% prefill, bit-identical, zero decode regression.** Prefill is compute-bound and scales to logical cores; decode is bandwidth-bound and is *hurt* by SMT siblings (opposite optima — a single global pool can't serve both). The fix runs only the prefill forward pass on a dedicated wider Rayon pool (logical cores), leaving the global/decode pool at physical cores. Measured prefill **19.19 → 23.73 tok/s (1.24×)**, decode unchanged, output byte-identical. Closes the prefill gap 0.62× → **0.80×** of llama.cpp. (`same-host/p1-prefill-pool-validate-*.json`, `same-host/p1-cpu-thread-sweep-*.json`; `src/inference.rs::prefill_thread_pool`.) This refines headline #1: the uniform CPU gap was *partly* thread width on the prefill side, not only kernel quality.
- **P1 (real, fixable, parity-safe, but medium-high effort):** unified tiled Q8 GEMM owner with register-blocked AVX2 (and an AVX-512 *prefill-only* variant) — now scoped to **prefill only** (decode is bandwidth-bound; faster dots don't move it). It chases the *residual* prefill gap (0.80× → ~1.0×) once the wider prefill pool is saturated. Plan in `SPEED_FIX_PLAN.md`.
- **Decode (bandwidth-bound, no cheap bit-exact win):** the packed VNNI/AVX2/scalar dots are byte-identical AND identical-throughput (`same-host/p1-vnni-kernel-matrix-*.json`) — confirming decode waits on DRAM, not the ALU. llama's decode edge (8.65 vs 5.73) is memory-access/prefetch + ~29% non-matmul per-token overhead (qkv/rope/norm/KV/fork-join, from the streaming role profile), not dot speed. Bigger prize, architectural effort, separate from P1.
- **Not worth chasing now:** CUDA decode (parity-locked), AVX-512 *decode* (downclock + bandwidth-bound), the existing gated packed *decode* kernels on this host (measured no-op/regression), sampler/server.
