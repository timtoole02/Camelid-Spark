# SPEED_CAMPAIGN.md Phase 3 — raw-decode profile (Gate 3)

Date (UTC): 2026-06-20
Machine: RTX 3060 Laptop GPU (GA106, CC 8.6, 30 SM), 6 GB GDDR6, driver 576.83, CUDA 12.9
Model: Qwen3-4B-Q8_0 (sha256 8c2f07f2…473300), fully VRAM-resident
Camelid: feat/bactrian-experimental-fast, worktree dirty (resident-CUDA fix + isolation uncommitted)
Tools: `CAMELID_DECODE_TIME` per-stage timing; Nsight Compute 2025.2.1 `ncu` (hardware counters)

## Question

Phase 2 measured Camelid raw GPU decode at **0.73–0.76×** of llama.cpp (`-fa1`) on the same
4B Q8 model. Is that gap recoverable kernel headroom, or are both engines already at the
memory wall? (SPEED_CAMPAIGN.md §1 lane 4 / §4 Phase 3 / risk register "raw lane likely loses".)

## Dominant cost (named, with numbers)

**Decode is ~100% GPU-forward, and the forward is bottlenecked on the `q8_gemv` kernel,
which saturates neither memory nor compute.**

### 1. Where the wall-time goes — `CAMELID_DECODE_TIME` (4B, per token)

```
[decode-time] per token: step wall 23.40ms | forward 23.40ms (embed 0.000 layers 0.00)
                         | sample 0.00ms | in-step other 0.00ms | loop other 0.00ms
```

- forward ≈ step wall (23.4 ms); sampling, embedding, CPU loop overhead, and sync stalls are
  **~0 ms**. There is no CPU-side or transfer bottleneck to blame — the entire decode token is
  the resident GPU forward pass (36 layers of GEMV + attention).

### 2. Roofline (engine level)

- Bytes read per token = resident weights streamed once by the GEMVs = **4315 MiB = 4.525 GB**
  (per-layer q,k,v,o,gate,up,down ×36 + output projection; embedding is a row lookup, not streamed).
- Peak DRAM bandwidth = 6001 MHz mem-clock × 2 (GDDR6 DDR) × 192-bit / 8 = **288 GB/s**
  (verified convention: RTX 3090 9751 MHz ×2 = 19.5 Gbps ✓).

| engine | decode t/s (4B Q8) | achieved DRAM (t/s × 4.525 GB) | % of 288 GB/s peak |
|---|---|---|---|
| **Camelid** resident | ~41–42.5 | ~186–192 GB/s | **~65–67%** |
| **llama.cpp** `-fa1` | 54.5 | ~247 GB/s | **~86%** |

Both are memory-bound (GEMV reads every weight once/token). llama.cpp runs **near the wall (86%)**;
Camelid leaves **~20 points (~55–60 GB/s) unextracted**. The 0.75× gap is **not algorithmic** —
it is the GEMV kernel not saturating bandwidth.

### 3. Kernel level — Nsight Compute on `q8_gemv` (4B, hardware counters)

```
ncu --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
              sm__throughput.avg.pct_of_peak_sustained_elapsed,\
              sm__warps_active.avg.pct_of_peak_sustained_active \
    --kernel-name regex:q8_gemv --launch-count 4 --target-processes all \
    camelid.exe bench-generate Qwen3-4B-Q8_0.gguf --prompt "Hi there" --max-tokens 2
```

| q8_gemv launch (grid) | role | peak DRAM % | peak SM % | warps active % |
|---|---|---|---|---|
| 512 blocks | large FFN gate/up/down | 59.5 | 52.5 | 75.0 |
| 320 blocks | output projection | 65.7 | 55.0 | 73.2 |
| 128 blocks | attention Q/K/V/O | 42.2 | 37.7 | 62.9 |
| 128 blocks | attention Q/K/V/O | 42.5 | 37.8 | 62.9 |

(0.6B cross-check: 33.7–53.3% DRAM, same pattern.)

**Reading:** the kernel hits **42% (small attention projections) → 66% (large FFN) of peak DRAM**,
and **never exceeds 66% DRAM or 55% SM** — it saturates neither. DRAM throughput tracks grid size:
the small attention GEMVs launch only **128 blocks across 30 SMs**, too few memory transactions
in flight to hide DRAM latency. This is a classic under-occupied / latency-bound GEMV, worst on
the narrow attention-projection matrices.

## Conclusion → where the recoverable headroom is

1. **Raw decode is memory-latency-bound inside `q8_gemv`, not at the bandwidth wall** — confirmed
   by both the roofline (Camelid 65% vs llama.cpp 86% of peak) and ncu (kernel 42–66% DRAM, neither
   resource saturated).
2. **The headroom to match llama.cpp's ~86% (≈ +30% decode t/s) exists**, concentrated in the
   **small attention-projection GEMVs (128 blocks → 42% DRAM)**. The large FFN GEMVs (66%) are
   closer to acceptable.
3. **Optimization levers** (in priority order): raise in-flight parallelism for narrow GEMVs
   (more blocks per output row / split-K so all 30 SMs stay busy), 128-bit vectorized + fully
   coalesced Q8 block loads, and fusing the three small attention projections into one wider GEMV.
4. **Campaign implication (unchanged):** this is why the flagship bet is the *concurrent spec*
   lane (config 3), not raw decode — but closing even half this kernel gap would lift every
   Camelid lane (raw, n-gram spec, and the future draft-model spec) since they all decode through
   `q8_gemv`.

## Optimization attempt — warps-per-row (RESULT: bit-exact but flat; bound is the reduction)

Hypothesis: the kernel is memory-latency-bound, so giving each output row **2 warps** (WPR=2)
instead of 1 doubles in-flight memory requests and the grid, and halves the `terms` shared
footprint (higher occupancy). The per-block float terms are order-independent, so the final
lane-0 sum stays in strict block order — **bit-exact**.

Implemented + measured (then reverted):

| metric | 1 warp/row (baseline) | 2 warps/row | verdict |
|---|---|---|---|
| output-id sha (0.6B / 4B) | 0f9ac8fd / a3974dc7 | **identical** | ✅ bit-exact (parity held) |
| 4B decode t/s | ~42.5 | ~43.1 | +1.4% — **within noise** |
| q8_gemv DRAM% (large FFN) | 60–66 | 62–68 | +2 pts |
| q8_gemv DRAM% (small attn, grid 2×) | 42 | 41 | **unchanged** |

The grid doubled and per-row warps doubled, yet DRAM% barely moved and the small GEMVs did
**not** improve at all. This **rules out occupancy and grid-starvation** as the limiter and pins
it on the **serial block-order reduction** (lane 0 summing `myterms[0..bpr]` in order): every
warp ends in the same sequential dependency chain, so adding warps/blocks cannot help. That sum
order is the **bit-exact parity anchor** (GPU == cpu_reference == llama.cpp) and cannot be
reordered without abandoning that guarantee. **Conclusion: the bit-exact `q8_gemv` is at its
performance ceiling on this hardware (~65% of peak); the ~25% gap to llama.cpp is the cost of
exact CPU-reference parity, not a fixable kernel inefficiency.** Reverted (kept the simpler
kernel); the finding is the deliverable. This extends the prior block-size-sweep finding to a
complete, evidence-backed boundary.

**Implication:** do not invest further in bit-exact raw-decode kernel tuning. The recoverable
speed is in the **concurrent spec lane (Phase 4)** — and, separately, a *deliberate* decision to
trade GPU==CPU bit-exactness for a faster parallel-reduction GEMV (a support-claim change, not a
unilateral one).

## Reproduce

```
# per-stage timing
CAMELID_DECODE_TIME=1 camelid.exe bench-generate Qwen3-4B-Q8_0.gguf --prompt "…" --max-tokens 64

# kernel counters (Nsight Compute 2025.2.1; non-admin profiling worked on this box)
ncu --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --kernel-name regex:q8_gemv --launch-count 4 --target-processes all camelid.exe bench-generate …
```

**Caveats.** GPU clocks not pinned (laptop; thermal drift — Camelid t/s spans 38–42.5 across runs;
the % figures use a representative 41–42.5). `q8_gemv` DRAM% is per-launch and grid-size-dependent;
the forward-weighted average sits ~55–60%. Peak 288 GB/s derived from the reported 6001 MHz mem
clock; the 3060 Laptop SKU range is 288–336 GB/s, so % of peak is an upper-ish bound at 288.
