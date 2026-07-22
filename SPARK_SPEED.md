# SPARK_SPEED — what's between this engine and the GB10's ceiling

A four-lens audit (lane reachability, memory budgeting, load path, decode/prefill
hot loop) of this engine on the DGX Spark target: GB10 (cc 12.1), 20-core Grace,
**128 GB unified LPDDR5X @ ~273 GB/s**. Decode is memory-bandwidth-bound, so the
ceilings are ~**3.9 tok/s** for 70B Q8_0 (~70 GB), ~**6.8 tok/s** for 70B Q4_K_M
(~40 GB), ~**34 tok/s** for 8B Q8_0.

The engine's CUDA lanes were developed against a **6 GB discrete RTX 3060** —
separate VRAM, PCIe. Both assumptions are false on the Spark, and that mismatch
(not kernels) is where the speed goes. Kernels are deliberately untouched
(FLINT charter).

## Stage S1 — shipped in this build (config + honesty)

| Change | Why |
|---|---|
| `HardwareProfile.cuda_unified_memory` (CU_DEVICE_ATTRIBUTE_INTEGRATED + size heuristic); `[hw]` banner says **UNIFIED memory** | Every Spark policy keys off this |
| aarch64-Linux execution-plan arm | Previously **no plan arm existed**: every CPU path (parity-probe legs, fallbacks, >16k-prompt prefill) ran the fail-closed plan **single-threaded scalar**. Now: CUDA-resident plan when the GPU drives decode; `CAMELID_PARALLEL_LINEAR=on` for the 20-core CPU reference otherwise (~20-40× on probe legs) |
| gemma4 serve + CUDA lanes default **on** when a CUDA device is present (`CAMELID_GEMMA4_SERVE` / `CAMELID_GEMMA4_CUDA` still override) | The NVFP4 pilot — the point of this build — was unreachable from the GUI without shell env |
| gemma4 CUDA KV cap: 4096 → **32768** on unified memory (`CAMELID_GEMMA4_KV_CAP` overrides) | The 4096 literal was sized for the 6 GB dev card |
| CUDA graphs default **on** for Linux (`CAMELID_CUDA_GRAPHS=0` disables) | Off-default was a Windows/WDDM measurement; GB10 is the "much faster GPU" the code comment anticipates. ~10–20% on ≤8B decode |
| Fit advisor unified branch | The discrete arms **double-counted** (free "VRAM" + 80% host RAM = the same bytes ≈ 200 GB claimed on a 128 GB box). Now: one pool, 2× footprint while the loader double-holds weights (see S2), honest `Unknown` between 1× and 2× |
| Gemma 3 27B pulled from the catalog | gemma3 has **no GPU lane** in this engine (serve 503s by default; its only runtime is a CPU f32 reference) — it was a 27 GB dead download |
| Legacy `CAMELID_CUDA_Q8` ignored on unified memory | That lane re-uploads weights per matmul — same DRAM copied to itself |
| Linux host-RAM probe wired into the gait budget | CPU-lane KV growth was unguarded (`None` → `u64::MAX`) in the same pool the GPU uses |

## Stage S2 — shipped: unified-memory runtime policies

| Change | Why |
|---|---|
| Auto layer-offload **disabled on unified memory** (`CAMELID_OFFLOAD_FORCE_LAYERS` stays as an explicit test hook) | Offloading on one pool adds a duplicate pinned copy + a DDR→DDR memcpy per forward (~3× traffic) while freeing nothing; a model that doesn't fit is now a clean CPU fallback with a log line, not a tri-copy OOM |
| Resident context defaults to **32k** on unified (`CAMELID_CUDA_RESIDENT_MAX_CONTEXT` overrides) | The KV is allocated eagerly (zeroed at build) and, uncapped, sized to ALL remaining free memory at the trained context (131k) — tens of GB gone before the first token |
| VRAM headroom scales to **5% of the pool** on unified (`CAMELID_CUDA_RESIDENT_HEADROOM_MB` overrides) | 512 MiB was a 6 GB-card floor; the OS lives in the same pool |
| GPU prefill's 16384-token cap **lifted on unified** (`CAMELID_CUDA_PREFILL_MAX_TOKENS` overrides) | The engine's own VRAM-sized `max_pos()` bound is authoritative; the literal forced long prompts onto the CPU for no reason |
| GPU speculative verify defaults **on** with a CUDA device (`CAMELID_SPEC_GPU=0` opts out) | CPU chunk-verify beside a resident target silently demoted the whole decode to the CPU lane |
| K-quant guard on the batched GPU verify | The batched verify shares the Q8_0-only GEMM stack; a K-quant engine now falls back to the (lossless) CPU chunk verify instead of reading kernels that don't exist for its formats |

## Stage S2-deep — pending: zero-copy weights (the 70B Q8_0 unlock)

**Until this lands, test 70B via the Q4_K_M row** (fully resident, ~6.8 tok/s
ceiling); the 70B Q8_0 row downloads and merges fine but will fall back to the
CPU lane at load.

**Today 70B Q8_0 is effectively dead on arrival**: the loader materializes all
weights in host RAM (~74 GB), the resident engine uploads a **second** full copy
to "device" memory — the *same physical pool* — and the auto-offload split
(triggered by the phantom "low VRAM" this creates) adds a **third** pinned copy
that is memcpy'd DDR→DDR every token. ≈140 GB demanded of 128 GB → OOM → falls
to the CPU lane.

Fix (the CUDA twin of the existing Metal `CAMELID_METAL_NOCOPY` lane, whose
`WirePages`/mmap infrastructure is already in-tree): `cuMemHostRegister` the
mmap'd wire pages and hand kernels the mapped device pointer — **zero copy,
zero upload**, ~1× footprint, disk-bound loads. Plus: never take the offload
split on unified; bound the eager KV (fills *all* free memory at the 131k
trained context today); make the CPU-materialization estimator count retained
bytes (it counts the big lanes as 0, so model switches leak a full model).

Expected: 70B Q8_0 from broken → **~3.9 tok/s**; every model frees ~1× its
size; loads minutes → seconds.

## Stage S3 — pending: prefill chunking + multi-turn KV reuse

Held until the S1/S2 baseline is validated on real GB10 silicon (these change
numerics-adjacent launch configs or the chat KV lifecycle):

- **K-quant prefill is serial** (one full weight pass per prompt token): 70B
  Q4_K_M 2k-token prompt ≈ 5–10 min TTFT. Q8_0's batched prefill re-reads all
  weights every **8** tokens (`MAX_VERIFY_K`, sized for 6 GB shared-memory
  scratch — raising it needs a launch-config audit against 48 KB smem).
- **Every chat turn re-prefills the entire conversation** (prefix cache is
  bypassed on the resident lane; GPU prefill requires position 0). Fix shape:
  suffix prefill at `base=filled()` like the gemma4 lane's `cached_tokens`.
- The CPU-materialization estimator counts the big lanes as 0 bytes, so model
  switching leaks one whole model per switch until eviction math is honest.
- GPU sampling lane (Gumbel) is default-off after a Windows-driver corruption;
  re-validate on the Spark (`CAMELID_GPU_SAMPLING=1` if the env exists in this
  tree) — costs ~20-25% of sampled-decode throughput while off.

## Verified-good already

The per-token decode loop is clean: ~32 KB H2D + 4 B D2H per token, GPU argmax,
KV scatter/attention device-side, one sync per token. Once weights are resident
(S2), decode should sit near the bandwidth ceiling without kernel changes.
