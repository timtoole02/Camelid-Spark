# STAMPEDE Phase 6 — CUDA multi-stream overlap: verdict NULL (WDDM), ships default-off

Mission: overlap the independent K/V and FFN-up GEMV chains per Full layer in
`src/cuda_resident.rs::forward_pass` on side CUDA streams with event joins, behind
`CAMELID_CUDA_STREAMS` (default OFF), live path only. Plan:
`docs/perf-deep-dive/PHASE6_CUDA_STREAMS_PLAN.md` (pinned 41d47433; anchors verified live at
implementation HEAD). Gate (STAMPEDE_CONDUCTOR.md): **>= +8% decode at low ctx on both gate
models AND no depth regression**.

**Verdict: NULL on this box (RTX 3060 Laptop 6GB, WDDM 576.83, CUDA 12.9). The overlap is
correctness-proven and engaged, and decode REGRESSES at low ctx at both rungs of the risk
ladder. The flag stays default-OFF; every kernel launches unchanged when it is off (no side
streams are even constructed).**

## Measured deltas (median-of-5 two-point decode, fresh same-session OFF baselines, OFF/ON
back-to-back per cell, ON legs engaged-checked via CAMELID_RESIDENT_TRACE)

### Rung A — side streams + cudarc default multi-stream event tracking (commit 9247a380)

| cell | OFF tok/s | ON tok/s | delta |
|---|---|---|---|
| Llama-3.2-3B Q8_0, low ctx | 48.43 | 43.82 | **-9.5%** |
| Llama-3.2-3B Q8_0, depth ~1881 | 26.71 | 24.67 | **-7.6%** |
| Qwen3-4B Q8_0, low ctx | 32.17 | 31.35 | **-2.5%** |
| Qwen3-4B Q8_0, depth | (INVALID cell: prompt overshot the 2090-pos resident cap; CPU-fallback tail measured — receipts renamed `*.INVALID-cap-overflow.json`) | | n/a |

Receipts: `qa/perf/stampede-p6-cuda-streams-{off,on}-{llama3b-q8,qwen3-4b-q8}-{lowctx,depth}-20260709.json`.
(Receipt caveat: the harness stamps every receipt with its hardcoded note string
"Pre-optimization baseline"; disregard that field — the labels and this memo are the record.)

### Rung B — + `unsafe ctx.disable_event_tracking()` (commit 35677285), 4B depth resized to 1550

| cell | OFF tok/s | ON tok/s | delta |
|---|---|---|---|
| Llama-3.2-3B Q8_0, low ctx | 42.25 | 38.43 | **-9.0%** |
| Llama-3.2-3B Q8_0, depth ~1881 | 23.34 | 23.80 | +2.0% (sd 2.55 — noise) |
| Qwen3-4B Q8_0, low ctx | 31.78 | 30.44 | **-4.2%** |
| Qwen3-4B Q8_0, depth ~1550 | 18.13 | 17.99 | -0.8% (noise) |

Receipts: `qa/perf/stampede-p6-cuda-streams-{off,on}-*-20260709-rungB.json`. OFF baselines sit
below Rung A's absolute numbers (hours of sustained load on a thermal-limited laptop); the
within-cell OFF->ON comparison is the controlled quantity.

## Mechanism: the event-tracking A/B localizes the cost to WDDM, not cudarc

A correction to the plan's premise, established during review against the vendored cudarc
0.19.7 source: the engine's context is ALWAYS in cudarc multi-stream mode (the main stream
itself comes from `ctx.new_stream()`), so per-slice-arg event tracking is paid by flag-off
engines too — creating the side streams flips no mode. That makes the A/B sharper, not
weaker: Rung B's ON legs ran with tracking disabled (strictly LESS host bookkeeping than
their own OFF baselines, which keep tracking) and still regressed -9.0%/-4.2% at low ctx
(vs Rung A's -9.5%/-2.5% with tracking on). The residual regression is therefore the
cross-stream structure itself on this driver: WDDM's software scheduler does not
co-schedule the sub-100us GEMV kernels, and the per-layer cuEventRecord/cuStreamWaitEvent
traffic (7 extra enqueues/layer) breaks WDDM's launch batching, adding overhead instead of
overlap. Depth was flat at Rung B (both cells within noise); Rung A's one valid depth cell
(3B, -7.6%, tracking on) regressed — consistent with tracking overhead scaling with launch
count, which disable_event_tracking then removed.

**Falsifiable follow-up (not chased here): on Linux or a TCC-mode Windows GPU the same code
should show the predicted +8-15% — the hardware-scheduled launch path has none of WDDM's
batching behavior. The flag exists; a capable host can A/B it in one session.**

## Correctness evidence (both rungs)

Byte-identical greedy corpora OFF==ON (5-prompt AB corpus incl. a ~1772-2080-server-token
depth prompt; corpus key-count asserted; ON legs required the "overlap ENGAGED" trace):

- Legs: Llama-3.2-3B Q8_0 (low+depth, split-K attention exercised); Qwen3-4B Q8_0 (QK-norm
  on side_a); Qwen3-4B Q4_K_M (K-quant gemv lanes + Q8_K scratch); 3B Q8_0 with
  CAMELID_RESIDENT_NO_FUSION=1 (unfused chain); ornith-1.0-9b Q4_K_M qwen35
  (device-side decode loop / forward_token_device; interleaved Full/SSM layers — overlap
  engages only on Full layers). Legs 1-4 use the 5-prompt corpus with the depth prompt;
  leg 5 uses the 4-prompt corpus (no depth prompt — the qwen35 device-loop lane is
  exercised for the forward_token_device entry point, not for context depth).
- Rung A (tracking on): 5/5 at 9247a380. Rung B (tracking off): 5/5 at 35677285+ — legs
  1+5 in the first run, legs 2/3/4 re-proven from a pinned worktree after a harness
  incident (below).
- Device tests: 34/34 ignored CUDA tests pass flag-off AND flag-on at both steps.
- Parity risk is zero by construction: kernels launch unchanged (same grid, same reduction
  order); only the enqueue stream differs, with event joins before every dependent read.
  Verify/graph/batched paths are untouched (single-stream), so no `splitk_verify_active()`
  analogue is needed.

## Receipts-integrity note

Three contamination events were caught and fenced during measurement; all receipts above
postdate the fences:

1. A concurrent session's `git checkout` in the shared repo checkout swapped the AB probe
   mid-validation (reflog timestamp == probe mtime), truncating the corpus and fabricating
   a "divergence". Fix: pinned worktree for all receipts; probe hard-fails on non-string
   content; validate script asserts exact corpus key counts (both would have caught it).
2. A leaked GPU-resident server from another session timeshared the GPU during the first
   Rung B bench (3B read 6.6 tok/s vs true ~42-48). Fix: a clean-box precondition (GPU has
   no other compute processes; no cargo/rustc running) — enforced by a session wrapper for
   the Rung B matrix above, and now built into `scripts/bench-cuda-streams-matrix.sh`
   itself for every future run.
3. undici's default 300s fetch timeout killed a leg whose K-quant depth prefill
   legitimately took 302s under load. Fix: probe speaks node:http with no client timeout.

## Disposition

- `CAMELID_CUDA_STREAMS` stays **default OFF**. Flag-off constructs nothing and enqueues the
  byte-identical launch sequence (lazy construction buys provable inertness — NOT a
  multi-stream-mode switch; the context is in multi-stream mode either way, see Mechanism).
- The implementation + event graph are correct and stay in-tree for the Linux/TCC follow-up,
  exactly like the CUDA-graphs precedent (correct, parity-clean, measured no-win here).
- `disable_event_tracking` remains tied to the flag and scoped to this engine's CudaContext
  (other lanes build their own contexts); `enable_offload_scratch` carries a one-time drain
  (before the state is armed) so the offload copy stream is safe under tracking-off, and
  forward_pass drains the context on any Err while the overlap is armed so no unjoined
  side-stream work can outlive an error into a drop/rebuild.
