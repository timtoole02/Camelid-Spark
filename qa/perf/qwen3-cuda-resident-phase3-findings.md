# Qwen3 CUDA-resident — Phase 3 throughput: baseline + Nsight findings

Hardware: RTX 3060 Laptop GPU (Ampere, sm_86, 30 SMs, 6 GiB, ~336 GB/s), CUDA 12.9,
driver 576.83, Windows + MSVC. Same-session, greedy/deterministic (temperature 0).
Host RAM pressure noted: ~2.4 GiB free during capture (0.6B fits comfortably).

## Baseline (Qwen3-0.6B-Q8_0, median of 5)

| metric | median | min | max | stddev |
|---|---|---|---|---|
| Decode tok/s (empty context, 128 gen) | 129.9 | 122.4 | 132.2 | 3.5 |
| Prefill tok/s (571-token prompt) | 125.9 | 124.7 | 127.1 | 0.8 |

Artifact: `qa/perf/qwen3-0.6b-cuda-resident-baseline.json`.

## Key finding: prefill is NOT batched

`CudaResidentDecode::prefill()` (src/cuda_resident.rs) is a **serial loop** that calls
`forward_pass` once per prompt token — identical work to decoding one token at a
position. The server's GPU prefill path (`inference.rs::try_resident_prefill_cuda`)
calls this serial loop. The `resident_single_shot_prefill` label in the execution
plan is aspirational; the implementation streams every weight from VRAM once **per
prompt token**.

Empirical confirmation: prefill tok/s (125.9) ≈ decode tok/s (129.9), and
571 tokens ÷ 130 tok/s ≈ 4.4 s ≈ the observed 4.54 s prefill wall time. Prefill is
running one token at a time.

The batched GEMM (`q8_gemm_batched`) that reads each weight once and reuses it across
K tokens already exists and is used by `verify_batch` (speculative decode), which
notes it is "much cheaper than k separate forward_token calls." Prefill does not use
it.

## Nsight Compute — decode/serial-prefill `q8_gemv` (batch-1 GEMV)

`--set basic`, `--clock-control none` (counter access on this box requires
`--clock-control none`; default clock locking hit a driver-resource/permissions
error). Three warmup launches:

| launch | grid | duration | DRAM % | Compute % | Achieved occ | Waves/SM |
|---|---|---|---|---|---|---|
| 0 | 128 | 9.54 µs | 45.2 | 20.8 | 53.8 | 0.85 |
| 1 | 128 | 9.31 µs | 46.2 | 21.5 | 53.5 | 0.85 |
| 2 | 128 | 14.85 µs | 57.5 | 25.6 | 59.9 | 0.85 |

Interpretation:
- **Memory-bound** (DRAM > 2× compute), as expected for a batch-1 GEMV that must
  stream every weight per token.
- **Under-filled device**: only 0.85 waves/SM ("grid too small to fill the available
  resources"), 54–60% achieved occupancy, register-limited (45 regs/thread → block
  limit 5). DRAM is at ~45–58%, not saturated — there is headroom even for decode via
  occupancy/grid, but decode is fundamentally weight-bandwidth-limited.

## Phase 3 plan (re-gate Phase 2 parity after each change)

1. **[biggest payoff — DONE, verified] Batched prefill.** Routed
   `try_resident_prefill_cuda` through a batched forward instead of the per-token loop.
   See result below.
2. Decode occupancy: reduce `q8_gemv` register pressure / raise waves/SM to push DRAM
   toward saturation (secondary; decode is bandwidth-bound). NOT yet done.
3. Larger prefill batch: `MAX_VERIFY_K=4` bounds the chunk; a prefill-specific larger-K
   GEMM (dynamic shared memory) would push past the current 3.2× toward the 4× weight-
   read-reduction ceiling and beyond. NOT yet done — re-gate parity.
4. Tensor cores (INT8 IMMA) for the batched prefill GEMM — only if Nsight shows the
   batched GEMM compute-bound on the dp4a integer pipe. Re-profile first.

## Result — change #1 (batched prefill), Qwen3-0.6B-Q8_0, median of 5

`CudaResidentDecode::prefill_batched` ingests the prompt in `MAX_VERIFY_K`-token chunks
through the shared `run_batched_layer_stack` (extracted from `verify_batch`, single
source of truth), skipping the output-projection (prefill needs no logits). The server
defaults to it; `CAMELID_CUDA_RESIDENT_PREFILL_BATCHED=0` forces the serial loop (A/B).

| metric | baseline (serial) | batched | speedup |
|---|---|---|---|
| Prefill tok/s (571-token prompt) | 125.9 | **397.5** | **3.16×** |
| Prefill wall (571 tokens) | 4536 ms | **1437 ms** | 3.16× |
| Decode tok/s (empty context) | 129.9 | 132.0 | ~flat (untouched) |

Artifact: `qa/perf/qwen3-0.6b-cuda-resident-batched-prefill.json`.

### Parity (re-gated — GREEN)
- `prefill_then_decode_matches_sequential` (extended): batched prefill + decode is
  token-identical to sequential forwards, and logits match serial prefill to 1e-4,
  across full chunks + a short final chunk + cross-chunk causal attention.
- `verify_batch_matches_sequential` still green (the shared-helper refactor did not
  disturb the speculative path).
- End-to-end A/B (`scripts/qwen3-cuda-prefill-ab.mjs`): batched vs serial greedy
  /v1/chat/completions over the 3 fixed bundle prompts + a long multi-chunk prompt →
  **byte-identical completions**. Serial-prefill GPU is the path the committed CUDA
  bundles validated token-identical to llama.cpp 9632, so batched == llama.cpp by
  transitivity.
- Full `cuda_resident::tests` suite green (13/13); also fixed a pre-existing
  `rope_matches_cpu` launch bug (missing `pairing` arg, unrelated to this change).

The 3.16× (vs the 4× weight-read-reduction ceiling at K=4) reflects fixed per-token
costs that batching doesn't amortize: causal attention grows with context, and the
norm/RoPE/quantize kernels run per token regardless.

### Across rows (all GPU-resident on the 6 GiB 3060, median of 5, parity-green)

| row | prefill serial → batched | speedup | decode (flat) | A/B parity |
|---|---|---|---|---|
| Qwen3-0.6B-Q8_0 | 125.9 → 397.5 tok/s | 3.16× | ~132 tok/s | identical |
| Qwen3-1.7B-Q8_0 | 79.95 → 217.3 tok/s | 2.72× | ~79 tok/s | identical |
| Qwen3-4B-Q8_0 | 31.81 → 81.35 tok/s | 2.56× | ~40 tok/s | identical |

Each row: GPU-resident decode confirmed active; batched vs serial greedy completions
byte-identical over the 3 fixed bundle prompts + a long multi-chunk prompt (serial ==
llama.cpp 9632 from the committed bundles ⇒ batched == llama.cpp). Decode is untouched
(any small delta is same-session variance). 8B (offload split) not yet validated on the
batched path. Harness: `scripts/validate-cuda-prefill-row.sh`. Artifacts:
`qa/perf/qwen3-{0.6b,1.7b,4b}-cuda-resident-batched-prefill.json`.

## 8B / offloaded models — batched prefill falls back to serial

The batched layer stack (`run_batched_layer_stack`, shared with `verify_batch`) reads
each layer's VRAM weight slice directly; it has **no** offload-streaming path, unlike
the serial `forward_pass`. For an offloaded model (8B on a 6 GiB card streams trailing
layers from host RAM) it would read 1-byte placeholders. `prefill_batched` therefore
checks `is_offloaded()` and defers to the serial `prefill` (which streams correctly).

Verified without loading 8B (8.7 GiB model, only ~6 GiB host RAM free — a host limit,
not a code limit): forcing offload on 0.6B (`CAMELID_OFFLOAD_FORCE_LAYERS=14`, 14/28
layers to host RAM) kept generation **coherent and byte-identical** to the fully
resident run — proving the guard fires (no placeholder garbage) and offload math equals
resident math. So 8B uses serial prefill (offload), no batched speedup (correct — the
streamed layout can't be batched), no regression. Full 8B benchmarking is deferred as a
host-RAM limit.

## Decode — no cheap win (memory-latency-bound)

Decode is untouched by the prefill change. A parity-safe `q8_gemv` block-size sweep
(64/128/256/512 threads — bit-identical, pure launch config) left decode tok/s **flat
within noise** on 0.6B (122.7–128.7). The batch-1 GEMV is memory-latency-bound, not
occupancy-limited, and the decode CUDA graph already cuts host-side launch overhead.
Further decode gains would need structural work (overlapping the independent Q/K/V and
gate/up GEMVs across streams, or a fused QKV/GU kernel) — higher effort and parity risk,
deferred. Block size kept at the profiled default (256).

## Pushing prefill further — the K=4 ceiling

`q8_gemm_batched` holds every `[warp][token][block]` partial term in shared memory so
lane 0 can sum per block in deterministic order (the parity guarantee), costing
`warps(8) × K × blocks_per_row × 4` bytes. At the FFN down-projection that caps K under
the 48 KiB static-shared limit: 0.6B (bpr 96) → K≤15, 1.7B (bpr 192) → K≤8, **4B (bpr
304) → K≤5** — i.e. the largest, slowest row is already near the ceiling at the current
K=4. Going beyond needs dynamic-shared opt-in (≈100 KiB on sm_86) plus a prefill-sized
scratch, or a register-accumulating GEMM that avoids storing all block terms. Deferred:
moderate effort, and the row that most wants it (4B) benefits least without the opt-in.

## CI gate (the Phase 0 guardrail)

CI runners have no GPU, so the token-identical parity itself runs locally on a CUDA
host (`scripts/validate-cuda-prefill-row.sh`), mirroring the Gemma4 precedent. Two
automated layers guard against silent regression of the GPU path:

1. **Compile gate (pre-existing).** The `rust` job runs `cargo clippy --all-targets
   --all-features -D warnings` and `cargo test --all-targets --all-features` on ubuntu
   + windows. `--all-features` includes `cuda` (cudarc dynamic-loads, no toolkit
   needed), so the resident path and the `#[ignore]`d CUDA parity tests are *compiled*
   on every push (they self-skip without a device). API/refactor drift fails CI here.
2. **Wiring guard (new).** `scripts/check-cuda-prefill-parity-gate.mjs` (wired into the
   `public-scrub` job) fails CI if the batched-prefill optimization, the shared
   `run_batched_layer_stack` (single source of truth, must keep per-head QK-norm), the
   server routing, or the equivalence tests are removed/weakened. This is the
   GPU-less backstop for "a CPU-opt commit silently broke the GPU path" — it can't
   prove parity, but it guarantees the parity machinery stays wired so a local GPU run
   will catch a real divergence.

Nsight orchestration note: profiling the on-demand HTTP server is fragile (kernel
replay resets in-flight requests); profile the startup **warmup forward** instead
(`scripts/ncu-profile-decode.sh` pattern, but capture warmup launches with `-s` small,
no request). `--clock-control none` is required on this box.
