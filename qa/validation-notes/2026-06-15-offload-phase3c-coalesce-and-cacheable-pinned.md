# Layer offload Phase 3c — coalesced transfer + cacheable pinned (the big PCIe win)

Host: Windows 11, RTX 3060 Laptop (6 GB), Optimus dGPU, Gen4 x8 during generation.
Build: `cargo build --release --features cuda`. Model: Llama-3.2-3B-Instruct-Q8_0 (28 layers).
Gate: `CAMELID_OFFLOAD_FORCE_LAYERS=0/14/27 camelid bench-generate <model> --prompt "What is a
transformer?" --max-tokens 50` (6-tok prompt, greedy).

## The investigation
Phase 3b's double-buffer overlap measured only ~7.2 GB/s effective PCIe vs an idle
`bandwidthTest` ceiling of ~12 GB/s. Two changes were made and measured separately.

### Change 1 — coalesce the 7 per-layer transfers into 1 (small win)
Previously each offloaded layer streamed its seven projections (q,k,v,o,gate,up,down) as
seven separate `memcpy_htod` calls into seven scratch buffers. Restructured so a layer's
seven projections are packed contiguously in ONE host buffer and stream as ONE transfer into
ONE scratch buffer; the gemv calls read sub-views (`scratch.slice(off[i]..off[i+1])`).
Result: effective PCIe 7.16 → 7.74 GB/s (marginal). So per-sub-transfer ramp-up was NOT the
main loss — ruled out a hypothesis.

### Direct probe: copy-stream peak vs in-generation average
Added `probe_offload_pcie` (50 back-to-back transfers, no compute). Peak = 8.77 GB/s while the
in-generation average was 7.74 → the overlap pipeline was already ~88% efficient. The wall was
the copy stream's OWN peak being far below the idle 12 GB/s.

### Change 2 — cacheable pinned instead of write-combined (BIG win)
`CudaContext::alloc_pinned` hardcodes `CU_MEMHOSTALLOC_WRITECOMBINED`. A same-run A/B probe
(WC buffer vs a `malloc_host(.., flags=0)` cacheable buffer, identical transfers) showed:
**WC 7.90 GB/s vs CACHEABLE 9.37 GB/s (+18.6%)**. WC host memory reads slower for H2D DMA on
this link. Switched offloaded weights to a `CacheablePinned` wrapper (raw `malloc_host` flags=0,
freed on drop, `unsafe impl Send` under the resident mutex; the driver auto-detects the pinned
pointer so a plain `&[u8]` view still drives the fast async DMA).

Copy-stream peak after switch: **7.36 → 11.08 GiB/s (11.9 GB/s)** — now at the idle hardware
ceiling.

## Parity gate (NON-NEGOTIABLE) — PASS
output_token_ids token-identical across 0/14/27 offload AND identical to the original Phase-3b
baseline (so unchanged vs the CPU/llama.cpp-validated reference):

| config     | force-offload | tok/s |
|------------|---------------|-------|
| resident   | 0  layers     | 54.41 |
| partial    | 14 layers     | 6.75  |
| maxoffload | 27 layers     | 3.19  |

## Throughput across the streaming-optimization phases (tok/s)
| config     | 3a sync | 3b overlap | 3c coalesce+cacheable | total Δ vs 3a |
|------------|---------|------------|-----------------------|---------------|
| resident   | 54.5    | 54.4       | 54.4                  | — (no offload)|
| partial    | 5.0     | 5.49       | **6.75**              | **+35%**      |
| maxoffload | 2.7     | 2.68       | **3.19**              | **+18%**      |

Effective in-generation PCIe: 7.16 → **8.32 GB/s**. Copy-stream peak: 8.77 → **11.9 GB/s**.

### Change 3 — multi-buffered prefetch (NEGATIVE result, kept as a tunable)
Hypothesis: the in-gen 8.32 vs peak 11.9 gap was pipeline bubbles, fixable by letting the copy
stream run further ahead. Generalized 2 scratch buffers → N (`CAMELID_OFFLOAD_BUFFERS`, the copy
stream prefetches N-1 layers ahead; look-ahead prefetch issued AFTER `compute_done` is recorded
so write-after-read is correct — issuing it before clobbered the in-use buffer and produced
garbage, caught by the parity gate). Swept N on maxoffload(27): N=2 3.31, N=3 3.28, N=4 3.29,
N=6 3.24, N=8 3.24 tok/s — **flat; N=2 is optimal**. So offload is NOT buffer-bound. Combined
with the concurrent-bandwidthTest evidence (an external process gets 6-7 GB/s *while we
generate*, and the no-compute probe gets 11.9), the conclusion is the H2D link runs slower
under compute load (the gemv kernels contend for the memory controller / the laptop throttles),
so ~8.3 GB/s is the real loaded-link ceiling here. Default reverted to N=2 (minimal VRAM); the
knob stays for hardware whose loaded link has more headroom.

## Honest reading / what's left
The cacheable-pinned switch closed almost the entire gap to the hardware link ceiling — this was
the dominant lever, and it is platform-specific (WC vs cacheable can go either way on other
links; worth a one-time probe per platform). After it, in-generation PCIe (8.32) is ~70% of the
new copy-stream peak (11.9): a fresh ~30% pipeline gap opened up because transfers are now fast
enough (~9 ms/layer) that per-layer compute + sync bubbles are a larger relative share. The next
lever is more scratch buffers (triple/quad-buffer) to keep the copy stream saturated near peak —
but each extra 108 MiB buffer is VRAM that would otherwise hold a resident layer, so it trades
capacity for speed and only pays off where VRAM isn't the binding constraint. Estimated ceiling
if fully closed: maxoffload ~3.9 tok/s. This remains a CAPACITY feature; offloaded tok/s is
bounded by the link.

Artifacts: qa/dd_{resident,partial,maxoffload}.json (+ db_*/dc_* for the 3b/coalesce steps).
