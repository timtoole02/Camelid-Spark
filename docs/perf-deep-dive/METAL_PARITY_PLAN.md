# METAL_PARITY_PLAN.md

**Phase 2 deliverable — Camelid vs llama.cpp Metal parity campaign**
Spec: `METAL_PARITY_AGENT_SPEC`

## Goal

Close the prefill GEMM gap on Apple M4 — Camelid 3B prefill is **~0.79x** of llama.cpp (~452 vs ~572 tok/s; ~2.9 vs ~3.67 TFLOPS effective) — **without breaking bit-exact greedy parity**. Decode is already TIED and memory-bandwidth-bound (no lever there); the entire recoverable surface is in the prefill FFN GEMMs.

### Pins (reproduce exactly)

| Component | Pin |
|---|---|
| Hardware | Apple M4 |
| llama.cpp | `acd79d603` |
| Camelid | `2323033` |
| Quant | Q8_0 |
| Decoding | greedy |
| Reference prefill prompt | same prompt set as `llama-bench` (601-token prefill used in Phase 1 measurements) |

### Measured Phase-1 baseline (carried forward)

- **Decode (3B):** Camelid ~28.6 tok/s, gpu_busy 35 ms/tok, ~98 GB/s ≈ 82% of the ~120 GB/s wall. llama 28.85. **TIED, memory-bound — NOT a lever.**
- **Decode (small, 0.6B):** Camelid ~115 vs llama ~131 (~0.88x), ~74 GB/s — small-model dispatch / output-proj overhead. **Secondary, lower-value lever.**
- **Prefill (3B):** Camelid ~452 tok/s (2.9 TFLOPS) vs llama ~572 (3.67 TFLOPS, riding the ~3.4 TFLOPS M4 GEMM wall). **0.79x with headroom — the real gap.**
- **Prefill trace (per layer, 601-token prefill):** `gemm_gateup` ~49 ms (dominant) + `gemm_down` ~26 ms (second) dominate. Cumulative across 28 layers.

### Hard constraints (from the contract — apply to every candidate below)

1. **Bit-exact greedy parity is non-negotiable.** int8 accumulation is associative and SAFE; **any reduction RE-ORDERING that changes f32 rounding is OFF-LIMITS.** The existing tiled-MMA `q8_0_block_wire_mm` is already NOT byte-exact with the scalar k-split path (different accumulation order) and is gated behind `CAMELID_METAL_MM` — that gate (and its established baseline) is the parity reference any new kernel must match, not the scalar path.
2. **Rust-native MSL only.** Study, do not copy, llama/ggml. The findings below are study output; ports must be original.
3. **No M5+ tensor API.** M4 has no tensor cores — `simdgroup_float8x8` is the only matrix path. No candidate may assume tensor hardware.
4. **No fine GPU profiling (no Xcode / entitlement-gated counters).** Candidates are measured by **prefill tok/s + gpu_busy + `CAMELID_PREFILL_TRACE` stage timings + bit-exact greedy diff** only. Occupancy/ALU-stall claims stay 'suspected' until Xcode lands on the T7.

### Parity-class definitions

- **Parity-class A (free):** does NOT change the f32 reduction order of the *currently-shipping prefill kernel* (`q8_0_block_wire_mm` tiled MMA, the baseline behind `CAMELID_METAL_MM`). Output is bit-identical to today's Camelid. Always allowed.
- **Parity-class B (gated, re-baselined):** changes accumulation order vs today's kernel → new bit pattern. **Rejected unless** it can ship behind its own flag with a freshly-captured greedy baseline AND it is the *only* way to get the win. Default-off until the new baseline is the documented reference.
- **Parity-class C (rejected):** re-associates f32 reductions on the default path with no clean baseline story. Off-limits by contract.

---

## Ranked candidates

Ranked by **(suspected prefill-tok/s recovery) × (parity-safety) × (1/effort)**. The FFN gate/up GEMM dominates the trace (~49 ms/layer), so its tiling sits at #1.

---

### #1 — Fuse gate + up FFN GEMM into a single weight-streaming pass

- **Bottleneck class:** compute-bound prefill GEMM (the dominant `gemm_gateup` stage, ~49 ms/layer).
- **Why it's #1:** gate and up are computed *sequentially* (metal.rs:11392–11413) from the **identical** input `ffn_y` (normalized residual, `n_tokens × 3072`). They are two independent `[n_tokens × 8192] @ [8192 × 3072]ᵀ` projections. Today each pass re-stages and re-streams its own ~26.7 MB Q8_0 weight and re-reads the same activation tile. The activation tile (`ffn_y`) can be staged once and reused across both weight matrices in one kernel.
- **Expected win:** *suspected* 12–18% on the `gemm_gateup` stage. The gate/up stage is the single largest line item, so this is *suspected* ~6–10% on whole-prefill tok/s for 3B. (Estimate basis: the activation B-tile stage and per-tile setup are paid twice today and once after fusion; the weight A-stream — the larger traffic — is unchanged, so the win is bounded by activation-reuse + dispatch/setup amortization, not halved-weight-BW. Earlier study text floated "halve weight-load" — that is **not** correct here because gate and up have *different* weights; only the activation reuse and the single-dispatch setup are saved. Marked conservative on purpose.)
- **Parity risk:** **Parity-class A (free) — IF** each output fragment keeps the exact same K-accumulation order as today's `q8_0_block_wire_mm`. Fusion only changes *which threadgroup* emits which output column block and *when* the shared B-tile is loaded; it does NOT change the per-output-element f32 reduction sequence. The two output matrices (gate_buf, up_buf) are written from independent `simdgroup_float8x8` accumulators with the same inner K-loop as the standalone kernel. **No re-association → bit-exact with current Camelid.** This must be asserted by the greedy diff, not assumed.
- **Rust-native implementation sketch:**
  - New kernel `q8_0_block_wire_mm_gateup` (and `_f16o` variant to match the existing `q8_0_block_wire_mm_f16o` path) derived from `q8_0_block_wire_mm` (metal.rs:1165–1289).
  - Keep tile `NR0=64` weight rows, `NR1=128` tokens, `NK=32`, 256 threads / 8 simdgroups, 2×2 quadrant layout — unchanged so the K-loop reduction order is byte-identical.
  - Threadgroup memory: stage B (activation, the shared `ffn_y` tile, 4 KB) **once**; carry **two** weight A-regions (gate, up). This raises threadgroup memory from 12 KB toward ~16–20 KB (8 KB A_gate + 8 KB A_up + 4 KB shared B). Verify against the M4 32 KB threadgroup limit and the scratch budget that the `use_mm` gate already checks (metal.rs `use_mm` selection); if the dual-A footprint trips the budget gate, narrow to `NR0=64` with a single A-region double-buffered across the two weights (stage gate-A, MMA, restage up-A, MMA) — still one B-stage, still one dispatch.
  - Each simdgroup keeps its existing `mc[16]` accumulators for gate; allocate a second `mc[16]` for up (register pressure check — *suspected* OK at 256 threads, but if occupancy drops, fall back to the double-buffered-A single-accumulator variant above).
  - Bind both weights and both output buffers in one dispatch (replaces the two `gemm(...)` closure calls at 11392–11413 with one `gemm_gateup(...)` call).
- **Files touched:** `metal.rs` (new kernel source near 1165–1289; new dispatch path; the `gemm`/`use_mm` closure region 10508–11509, specifically the gate/up call site 11392–11413). Possibly a new pipeline entry in the kernel registry.
- **How to measure:**
  1. Warm the prefill graph (see Measurement protocol) so kernel compile is excluded.
  2. `CAMELID_PREFILL_TRACE=1`, capture `gemm_gateup` ms/layer before/after; confirm the stage shrinks and total prefill tok/s rises.
  3. gpu_busy delta over the 601-token prefill.
  4. **Bit-exact greedy diff** of full output token stream vs current Camelid (must be identical — this is Parity-class A) AND vs llama.cpp exact-row reference.
  5. Confirm `use_mm` still selects this path for the 8192-row gate/up shape; confirm fallback to k-split still byte-matches when dims aren't 128-multiple.

---

### #2 — N-dimension tile retune for the gate/up shape (NR0=64 → 128/256)

- **Bottleneck class:** compute-bound prefill GEMM (gate/up, N=8192).
- **Expected win:** *suspected* 3–8% on `gemm_gateup` if the current 64-row tile is occupancy- or dispatch-overhead-limited at 128 threadgroups wide (8192/64). Wider tiles (NR0=128 or 256) reduce grid width and per-threadgroup launch overhead. *Suspected* and **must be measured** — could be net-zero or negative if the kernel is already bandwidth-bound (the weight A-stream is the bulk traffic and a wider tile doesn't reduce it). Lower confidence than #1.
- **Parity risk:** **Parity-class A (free)** *only if* the K-loop reduction order per output element is preserved. Widening NR0 adds more output rows per threadgroup but each output element still accumulates over K in the same sequence. **No re-association.** Confirm with greedy diff. (Note: changing NR1 token tiling similarly is parity-safe by the same argument, but NR1=128 is already the weight-streaming sweet spot per the findings, so leave it.)
- **Rust-native implementation sketch:**
  - Parameterize `q8_0_block_wire_mm` tile via MSL function constants or a second specialized kernel `q8_0_block_wire_mm_nr128`.
  - NR0=128 → 4×4 quadrant layout per threadgroup (16 output 8×8 fragments per simdgroup region) or keep 2×2 quadrants with more simdgroups; pick whichever keeps the inner K-loop fragment order identical.
  - Threadgroup memory grows with NR0 (A-region doubles for NR0=128 → ~16 KB A + 4 KB B). Check 32 KB limit + scratch gate.
- **Files touched:** `metal.rs` (kernel 1165–1289; tile constants; `gemm`/`use_mm` selection so the new tile is chosen only for the gate/up N=8192 shape).
- **How to measure:** A/B `gemm_gateup` and `gemm_down` stage ms/layer at NR0 ∈ {64, 128, 256}; warmed prefill tok/s; gpu_busy; greedy bit-diff each variant. Keep the winning NR0 *only if* parity-clean AND tok/s strictly improves; otherwise discard (it's a tuning knob, not a guaranteed win).

---

### #3 — Down GEMM K-streaming / tile retune (K=8192)

- **Bottleneck class:** compute-bound prefill GEMM (`gemm_down`, ~26 ms/layer, second dominant).
- **Expected win:** *suspected* 4–10% on `gemm_down`. Down is `[n_tokens × 8192] @ [3072 × 8192]ᵀ` — same FLOPs as gate/up but with a **2.67× larger K (8192 vs 3072)**, which the findings flag as causing more L2 eviction / K-traffic. Better K-blocking or a K-aware tile could recover the per-layer 26 ms. *Suspected* it is partly compute-limited (SiLU input is an f16/f32 mix), so the ceiling is lower than #1.
- **Parity risk:** **Parity-class A (free)** if K is still consumed in the same block order with the same per-element f32 accumulation. **Parity-class B (rejected unless gated)** if any K-split / split-K reduction is introduced (splitting K across threadgroups and summing partials re-orders the f32 reduction → new bit pattern). Prefer the class-A retune; do NOT introduce split-K on the default path.
- **Rust-native implementation sketch:**
  - Reuse `q8_0_block_wire_mm` with a K-blocking that improves L2 residency for the 8192-deep K (e.g., ensure the weight A-stream for the down weight stays contiguous per 128-token tile; verify the existing stride=64 swizzled load is optimal at K=8192).
  - Optionally a `_down` specialization with NR1 (token tile) tuned for the larger K so the activation tile is re-read fewer times. Parity-safe as long as accumulation order is fixed.
- **Files touched:** `metal.rs` (kernel 1165–1289; down call site 11443–11453; `gemm`/`use_mm`).
- **How to measure:** `CAMELID_PREFILL_TRACE` `gemm_down` ms/layer A/B; warmed prefill tok/s; gpu_busy; greedy bit-diff. Reject any split-K variant outright (parity).

---

### #4 — Fuse SiLU(gate ⊙ up) into the gate/up kernel epilogue

- **Bottleneck class:** prefill GEMM epilogue / memory traffic (couples to #1).
- **Expected win:** *suspected* 2–4% whole-prefill. The findings note a `silu_mul_f16o` dispatch exists separately today. Emitting already-activated `SiLU(gate) ⊙ up` directly from the fused #1 kernel saves one `640×8192` buffer store + the separate `silu_mul` dispatch + its readback. Only meaningful if built on top of #1 (needs both gate and up fragments resident in the same kernel).
- **Parity risk:** **Parity-class A (free)** — SiLU and the elementwise multiply are per-element f32 ops; computing them in-kernel vs in a separate dispatch does NOT change the order of any reduction. The GEMM reductions are untouched. **Bit-exact** provided the SiLU is computed in the same precision (f32 then cast, matching the standalone `silu_mul` path's rounding). Verify rounding parity carefully — if the standalone kernel does f16 intermediate and the fused one does f32 (or vice versa), the bits differ → re-baseline. Match the existing path's precision exactly.
- **Rust-native implementation sketch:**
  - Add an epilogue to the #1 fused kernel: after both `mc` accumulators are finalized, compute `silu(gate_frag) * up_frag` per output element before the store, emitting one fused buffer instead of two.
  - Mirror the exact SiLU formulation + precision of the current `silu_mul_f16o` kernel.
  - Removes the standalone `silu_mul` dispatch between gate/up and down (region ~11413–11443).
- **Files touched:** `metal.rs` (the #1 fused kernel; remove/bypass the standalone silu_mul dispatch; down GEMM input now reads the fused buffer).
- **How to measure:** stage trace should show `9:resid+silu` shrink and no separate silu dispatch; warmed prefill tok/s; gpu_busy; **greedy bit-diff is the gate** (SiLU precision is the parity risk here).

---

### #5 — Small-model decode dispatch / output-proj overhead (0.6B)

- **Bottleneck class:** small-model overhead (decode, not prefill).
- **Expected win:** *suspected* recover part of the 0.6B decode 0.88x gap (~115 → toward ~131). At ~74 GB/s the 0.6B decode is NOT at the BW wall, so the gap is per-token dispatch / output-projection overhead, not bandwidth. **Secondary, lower-value lever** per Phase-1 context — explicitly de-prioritized vs prefill.
- **Parity risk:** **Parity-class A (free)** if it only reduces dispatch count / fuses the output projection without changing reduction order. Greedy diff confirms.
- **Rust-native implementation sketch:** reduce per-token command-buffer/dispatch overhead on the decode path for small hidden dims; consider fusing the final RMSnorm + output projection dispatch. Do not touch the 3B decode path (it's TIED/BW-bound — no headroom).
- **Files touched:** `metal.rs` decode path (`ResidentDecodeState` decode, distinct from `prefill_tokens` 10508–11509).
- **How to measure:** 0.6B decode tok/s via **gpu_busy** (not HTTP wall); greedy bit-diff. Lower priority — only after #1–#4.

---

### Rejected outright (Parity-class C / B-without-justification)

- **Half-precision (f16) accumulate for gate/up FFN** — the study lists this as a *suspected* 10–15% bandwidth win, BUT f16 accumulation **changes the f32 rounding of every output element** → new bit pattern → Parity-class B at best, and there is no clean reason to re-baseline the whole engine for it. **REJECTED by the contract** (changes greedy output). int8 *quant* dot-products are fine (int8 accumulation is associative); the f32→f16 *accumulator* swap is the forbidden re-association.
- **Split-K / K-split parallel reduction on the default prefill path** — splitting K across threadgroups and summing partials re-orders f32 reductions. **REJECTED** on the default path (would only be admissible behind a gated, re-baselined flag with a proven exclusive win, which it is not — #1/#3 get the gains parity-free).
- **Prefill-only repacked Q8_0 wire layout (144B/4-block)** — *suspected* L2/BW win, but on-load deinterleave + a prefill-only format is **high-effort** and risks dequant-rounding drift vs the decode-compatible 34B blocks. The dequant is per-element `int8 * half(scale)` either way, so it *can* be parity-safe (Parity-class A) — but the effort/risk is not justified while #1–#4 are open. **Deferred, not rejected.** Revisit only if #1–#4 don't close the gap.
- **Any switch of the default path to the scalar k-split kernel for "parity"** — the scalar `q8_0_block_linear_ksplit_f32y_wire_gemm` is byte-exact to CPU but ~2.6 TFLOPS (slower). The tiled MMA at ~3.3 TFLOPS is the established prefill baseline behind `CAMELID_METAL_MM`. We optimize *within* that baseline; we do not regress to the scalar path.

---

## Measurement protocol

1. **Warm the prefill graph first.** Phase 1 found a methodology bug: the first prefill request pays Metal kernel compilation, contaminating tok/s. Issue at least one throwaway warm-up prefill (same model, same prompt length class) before any timed run, so all timed prefills hit compiled pipelines.
2. **Decode: use `gpu_busy`, not HTTP wall.** HTTP wall includes scheduling/IO jitter. Report gpu_busy ms/tok and derive tok/s + GB/s from it (decode comparisons in this campaign are gpu_busy-based; 3B = 35 ms/tok = ~98 GB/s reference).
3. **Prefill: warmed tok/s + `CAMELID_PREFILL_TRACE=1` stage timings.** Capture per-layer `2:gemm_qkv`, `5:gemm_o`, `7:gemm_gateup`, `8:gemm_down`, `9:resid+silu` before/after each candidate. The candidate is credited only if its targeted stage shrinks AND total warmed prefill tok/s rises.
4. **Same prompt set as `llama-bench`.** Use the 601-token prefill prompt used in Phase 1 so numbers compose with the baseline (Camelid ~452 / llama ~572 tok/s).
5. **Parity gate — two diffs, every candidate, every time:**
   - **vs current Camelid (`2323033`)** greedy token stream — Parity-class A candidates MUST be byte-identical. Any divergence demotes the candidate to class B and triggers reject-or-rebaseline review.
   - **vs llama.cpp (`acd79d603`) exact-row** greedy reference — must remain within the established Camelid↔llama parity envelope (no new divergence introduced).
6. **A/B on a quiesced box.** Per the auto-loop finding, a contended box (parallel dg/parity loops) poisons A/B. Reboot / quiesce before a credited measurement.
7. **Numbers tagged.** Every projected figure in this plan is *suspected* until measured by this protocol. Only the Phase-1 baseline numbers are measured.

---

## Phase 3 first implementation: **#1 — Fuse gate + up FFN GEMM**

**Implement #1 first.** Rationale:

1. **Largest target.** `gemm_gateup` (~49 ms/layer) is the single dominant prefill stage; even a conservative 12–18% stage win is the biggest single-candidate prefill-tok/s recovery on the board.
2. **Parity-free (class A).** It reuses the existing `q8_0_block_wire_mm` K-loop reduction order verbatim — only the dispatch shape and shared-activation staging change. The win does **not** require touching f32 accumulation, so it ships on the default path with a clean bit-exact greedy diff (no re-baseline, no gate gymnastics). This is the rare high-value + zero-parity-risk combination the contract rewards.
3. **Unblocks #4.** The SiLU-epilogue fusion (#4) depends on both gate and up fragments being resident in one kernel — which is exactly what #1 builds. Doing #1 first compounds.
4. **Bounded, measurable scope.** One new kernel + one dispatch-site change in `metal.rs`, measured cleanly by `gemm_gateup` stage delta + warmed prefill tok/s + the two parity diffs.

**Go/no-go for Phase 3:** ship #1 only if greedy output is byte-identical to current Camelid (Parity-class A) AND warmed prefill tok/s strictly improves on the 601-token llama-bench prompt. If the dual-A threadgroup footprint trips the `use_mm` scratch budget, fall back to the single-A double-buffered variant (still one B-stage, one dispatch) before abandoning the candidate.
