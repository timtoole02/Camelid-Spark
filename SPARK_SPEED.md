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

## Stage S2-deep — SHIPPED: zero-copy weights (`CAMELID_CUDA_HOSTREG`) — perf pending Spark

The resident engine now serves Q8_0 weights from **host-registered page-aligned
buffers** (`cuMemHostRegister` DEVICEMAP + `cuMemHostGetDevicePointer`, raw
`cudarc::driver::sys` — no VRAM copies). Default **on only for unambiguously
one-pool hardware** — the driver reports INTEGRATED, or the pool is ≥ 96 GiB
(covers a GB10 driver misreporting the attribute; the VRAM≈RAM size heuristic
alone must never flip weight serving off VRAM on a discrete 8/8 or 16/16 box).
Off everywhere else; `CAMELID_CUDA_HOSTREG=1/0` overrides either way, and
**`=upload`** selects a third mode: the same streamed 1× load, but the engine
device-copies each streamed repack (`clone_htod`) instead of registering it —
on one physical pool that is ALSO ~1× steady, with zero mapped-read risk. The
`=1`-vs-`=upload` A/B on the same box isolates the mapped-read penalty (the
receipt line carries the mode: `[cuda] hostreg mode=…: N zero-copy / M
uploaded`). Unknown env spellings warn once and resolve OFF. MoE models keep
the historical loaders (they never pass resident admission), and a
hostreg-loaded session that falls to the CPU path warns loudly once.

**`CAMELID_CUDA_PREFILL_K`** (default 8) widens the batched-prefill chunk —
TTFT scales ~1/K where weight reads dominate (3060 hostreg receipt: 1082-token
prompt 33.6 s → 15.7 s at K=16, tokens identical; VRAM-resident legs barely
move because their weight re-reads are cheap). K=16 needs > 48 KiB of
ordered-sum shared memory past ~736 blocks/row, which the engine now raises
via `cuFuncSetAttribute` (device opt-in permitting; clamped otherwise, capped
at 8 while flash prefill's k<=8 oracle is active). Verify width stays 8.

**Fixed in the same change — 32B/70B Q8_0 decode was broken outright**: the
serial `q8_gemv` stages `blocks_per_row*68` B of shared memory, past the
48 KiB launch default once ffn_dim ≥ ~23K (Qwen3-32B bpr 800 = 54,400 B,
Llama-3.3-70B bpr 896 = 60,928 B) — every resident forward threw
CUDA_ERROR_INVALID_VALUE before this; nothing larger than 8B had ever
exercised the kernel. The engine now raises the cap at build (host-side launch
config, kernel untouched; errors loudly into the CPU path only if even the
device opt-in limit cannot fit). Without this fix the 70B would have loaded
via hostreg and then failed its first token.

Lossless **n-gram speculation** also defaults ON for unified-pool-class serve
(`CAMELID_SPEC_DECODE=off` disables): every accepted draft skips a full weight
pass — the only decode lever that multiplies past the bandwidth ceiling — and
verification rides `q8_gemm_batched` on the resident engine (each block read
once per round, bit-identical reductions, W3-proven over registered memory).
Drafting auto-disables when GPU verify is explicitly off under hostreg (the
CPU chunk verify would disk-stream). See
docs/architecture/SPECULATIVE_DECODE.md. Registration lifetime = engine lifetime (synchronize → unregister →
dealloc, guards drop last); any per-tensor registration failure falls back to
upload silently, and the build prints the engagement receipt:
`[cuda] hostreg: N zero-copy / M uploaded`.

One correction to the original sketch above-the-fold in this file's history:
the resident `q8_gemv`/`q8_gemm_batched` kernels read a widened f32-scale
**SoA** layout, not the raw 34-byte GGUF wire, so registering wire bytes was
never layout-legal under the kernels-unchanged parity charter. Instead the
loader keeps Q8_0 projections **file-backed only** and the builder streams each
tensor disk → transient → its registered SoA buffer: **one steady host copy per
weight** (on one physical pool the registered bytes ARE the GPU copy). Only the
token embedding keeps wire pages (the CPU decodes a row per token; the tied
lm_head shares them).

3060 reference-card receipts (`qa/evidence-bundles/flint-w3-win3060-20260722-head-c50aa93/`):
token-identical hostreg 0-vs-1 on TinyLlama Q8_0 and Qwen3 0.6B/1.7B Q8_0
(64 greedy tokens each), 4-way identical with batched prefill over a 1082-token
prompt, identical under CUDA graphs, 42 GPU kernel/verify/tree unit tests green
over registered memory, spec-drafter rollback green with zero CPU steps; peak
host RAM 1.247 GB vs 1.256 GB flag-off control (≤1×). Discrete throughput is
PCIe-bound as designed (TinyLlama 130.6 → 5.3 tok/s): those runs are
correctness receipts — the perf question belongs to this box.

70B Q8_0 projection: ~78.2 GB registered SoA (lm_head included) + ~1.1 GB
embedding wire pages + a per-tensor build transient ≈ **~80.5 GB peak of the
128 GB pool** (the old triple-copy path demanded ≈140 GB). The fit advisor's
unified arm deliberately keeps the conservative 2× — it is quant/arch-blind
and only Q8_0 llama-family models get the one-copy layout — so the 70B Q8_0
row reads an honest *unknown* fit badge; the load path is the authority, and
it admits the model. Wanted from this box: the two hostreg stderr lines +
greedy tok/s — see FLINT_HANDOFF.md.

Still open from the original list: the CPU-materialization estimator counts the
big lanes as 0 bytes (model switches can leak a full model) — see Stage S3.

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
