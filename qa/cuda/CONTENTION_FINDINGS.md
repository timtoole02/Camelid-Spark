# CUDA VRAM contention findings (Task 4)

Replaces the prior **"no contention testing performed"** gap in the GPU eval
story. This documents the headroom policy and the contention harness, and is the
home for the captured numbers.

## Headroom policy (enforced at load)

`src/cuda_vram.rs` `evaluate(free_bytes, alloc_bytes, min_headroom_mib)` runs at
resident-engine load time (`src/inference.rs`, the resident-CUDA sizing path):
the projected device allocation (resident weights + streaming scratch + the sized
KV cache) is checked against **free VRAM** and a configurable minimum post-load
headroom **before** any allocation. If it would OOM, or would leave less than the
floor, the resident load is **refused with a named shortfall** and the model runs
on the CPU path — never a mid-load OOM.

- Floor: `CAMELID_MIN_VRAM_HEADROOM_MIB` (default **512 MiB**, sized so a 3B+
  model does not claim the last slice and starve the KV cache / a co-resident
  engine).
- The policy arithmetic is a pure function with unit tests
  (`cuda_vram::tests`) that run on any host (incl. the Windows dev box) — 6 tests,
  green.

## Contention harness

`cuda_vram::measure_contention(primary_bytes, second_bytes, runs)` (feature
`cuda`) occupies a model-sized allocation, then attempts a second allocation of
the same size on the same device, `runs` times. Per run it records whether the
second allocation **failed cleanly** (driver `CUDA_ERROR_OUT_OF_MEMORY`) or
succeeded, and free VRAM before/after the primary. `summarize_contention`
(pure, unit-tested) reports the **median and variance** of free-VRAM-after-primary
and a verdict (`clean-fail` / `fits` / `mixed`).

Run on the CUDA host:

```text
cargo test --features cuda --test cuda_contention -- --ignored --nocapture
```

Output artifact: `qa/cuda/contention-latest.json` (schema
`camelid.cuda_contention/v1`).

## Measurement status

| Item | Status |
| --- | --- |
| Headroom policy arithmetic | ✅ implemented + unit-tested (6/6) on the Windows dev box |
| Headroom refusal wired into resident load path | ✅ implemented; type-checks under `--features cuda` |
| Contention harness code | ✅ implemented; type-checks under `--features cuda` |
| **Contention numbers (median of 5, variance)** | ⏳ **pending** — must be captured on the CUDA host (dev box is RTX 3060 Laptop 6 GB). Not run on the Windows-without-GPU-build path used for this change. |

### Expected result (hypothesis, to be confirmed)

CUDA driver allocations return `CUDA_ERROR_OUT_OF_MEMORY` as a recoverable error
rather than aborting, so the second allocation is expected to **fail cleanly**
(`verdict: clean-fail`) on the 6 GB card when primary+second exceed free VRAM.
The harness asserts the process survives and the outcome is consistent across
runs; replace this section with the captured `contention-latest.json` numbers
(median free-after-primary MiB + variance) once run on the CUDA host.

> Discipline: report median of 5 with variance; do not assume slice behavior —
> measure it. Until the table row above flips to ✅ with numbers, the GPU story
> must state contention is *characterized by a ready harness but not yet measured
> on hardware*, not that it is proven.
