# Layer offload Phase 3b — double-buffered prefetch overlap

Host: Windows 11, RTX 3060 Laptop (6 GB), Gen4 x8 during generation.
Build: `cargo build --release --features cuda` @ commit ff21a65b (pre-commit; offload change local).
Model: Llama-3.2-3B-Instruct-Q8_0 (28 layers). Prompt "What is a transformer?" (6 tok), 50 tokens greedy.
Harness: `CAMELID_OFFLOAD_FORCE_LAYERS=N camelid bench-generate <model> --prompt ... --max-tokens 50`.

## What changed (3a → 3b)
3a streamed each offloaded layer's weights synchronously on the compute stream (transfer
then compute, serialized). 3b adds a second copy stream + two scratch buffers + CUDA events:
while the GPU computes layer N from scratch[cur], the next offloaded layer's weights stream
into scratch[1-cur]. `copy_done`/`compute_done` events enforce the read/write-after-read
ordering. The first offloaded layer is primed before the loop so resident-layer compute
overlaps its transfer.

## Parity gate (NON-NEGOTIABLE) — PASS
output_token_ids token-identical across all three split ratios (50/50 each):

| config     | force-offload | tok/s | ids == resident |
|------------|---------------|-------|-----------------|
| resident   | 0  layers     | 54.44 | (baseline)      |
| partial    | 14 layers     | 5.49  | YES             |
| maxoffload | 27 layers     | 2.68  | YES             |

Where the bytes live never changes the math. Parity is independent of split ratio. ✓

## Throughput before → after
| config     | 3a (sync) | 3b (overlap) | Δ      |
|------------|-----------|--------------|--------|
| resident   | 54.5      | 54.44        | ~0 (no offload) |
| partial    | 5.0       | 5.49         | +9.8%  |
| maxoffload | 2.7       | 2.68         | -0.7% (noise)  |

## Measured PCIe streaming throughput
Per-offloaded-layer weights = 108.0 MiB (100.66M Q8_0 params, 36/32 packing).
Per-decode-token: resident 18.4 ms | partial 182.1 ms | maxoffload 373.7 ms.
Marginal cost of each extra back-to-back offloaded layer (27 vs 14 regime, no resident
compute left to hide behind): **14.74 ms/layer → 7.16 GB/s effective**, ~45% of the
Gen4 x8 hardware cap (~16 GB/s).

## Honest reading of the result
Double-buffering helps **partial** (+9.8%): priming overlaps the first offloaded transfer
with the 14 resident layers' compute, and one-ahead prefetch hides some of the tail. It does
**not** help **maxoffload**: per-layer compute (~0.66 ms) is ~22x smaller than per-layer
transfer (~14.7 ms), so once every layer streams there is almost nothing left to hide — the
wall is raw transfer time, exactly as a capacity (not speed) feature predicts.

The new ceiling is not overlap, it is **transfer efficiency**: 7.16 GB/s is only 45% of the
x8 link. The cause is the 7-separate-memcpy-per-layer pattern (q,k,v,o,gate,up,down each its
own `memcpy_htod`), which leaves the link idle between sub-transfers. Coalescing the seven
projections into one contiguous host buffer + one memcpy is the next lever and should move
effective throughput toward the ~16 GB/s cap — a transfer change, parity-neutral, to be done
as its own one-change/one-gate step.

Artifacts: qa/db_{resident,partial,maxoffload}.json (+ .err).
