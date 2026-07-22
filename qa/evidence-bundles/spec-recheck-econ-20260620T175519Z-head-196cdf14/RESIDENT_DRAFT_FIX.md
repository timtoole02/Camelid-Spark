# Resident draft-model path — diagnosis + fix + honest 6 GB verdict

Follow-up to Phase 1 Path B / Phase 3: "fix the resident draft path so the 0.6B stays on GPU."
Date (UTC): 2026-06-20. Machine: RTX 3060 Laptop, 6 GB (≈5122 MiB free after desktop), CUDA 12.9.

## Diagnosis (why the 0.6B fell to CPU)

With `CAMELID_RESIDENT_TRACE=1` the draft engine build failed:

```
[resident-cuda] VRAM sizing: free 482 MiB, weights 639 MiB ... headroom 512 MiB ... fits 0 pos -> cap 0
[resident-cuda] VRAM too small for resident decode even with 28/28 layers offloaded (cap 0 < 256); using CPU path
```

Root cause: the 4B target builds its resident engine FIRST and sizes its KV greedily (cap 1045,
~294 MiB), taking free VRAM down to **482 MiB**. The 0.6B draft then can't satisfy the fixed
**512 MiB headroom** the sizing reserves (and the `cuda_vram::evaluate` backstop's 512 MiB
post-load floor), so it builds 0 KV positions and falls back to the CPU forward (~253 ms/token).
Not a model/eligibility problem — purely VRAM budgeting between two engines.

## Fix (committed)

Three changes, all default-off / gated so the single-model path is byte-for-byte unchanged:

1. **Speculative-coexistence VRAM policy** (`build_resident_cuda_engine` + a process reserve set
   by `ModelDrafter::new`). When a draft is in play, the **target reserves the draft's footprint**
   (weights + a capped KV at `CAMELID_SPEC_DRAFT_CONTEXT`, default 512, + a small margin) so the
   draft can build **fully GPU-resident** beside it; the draft engine itself uses a small headroom
   and a capped KV. **Honored only when the target still fits FULLY resident after the reserve** —
   see the gate below.
2. **The offload gate.** If honoring the reserve would force the target to **offload** trailing
   layers, the reserve is dropped and the target builds full-resident (draft → CPU, the prior
   lossless behavior). Two reasons offloading the target is the wrong trade: it slows the target
   forward ~3× (measured 41 → 14 t/s with 5 layers offloaded), and it triggers (3).
3. **Verify-offload correctness guard** (`verify_drafts_gpu`). The batched verify
   (`run_batched_layer_stack`) reads each layer's resident VRAM slice directly; for an **offloaded**
   layer that slice is a 1-byte placeholder (real weights stream into scratch only on the
   single-token path). A batched verify over an offloaded target therefore read garbage and broke
   losslessness. Now `verify_drafts_gpu` returns `None` when the engine `is_offloaded()`, routing
   to the lossless CPU chunk verify. (Latent bug — the resident draft path had never actually run
   before, so it was never exposed.)

## Measurements

**The mechanism works** — with the reserve engaged (gate temporarily bypassed for the test), the
0.6B built **fully GPU-resident** and draft latency dropped **253 ms/tok → 13 ms/tok (≈19×)**:

```
[resident-cuda] VRAM sizing: free 994 MiB, weights 639 MiB (resident 639 MiB; 0/28 offloaded) -> cap 512   ← draft fully resident
draft 13.2 ms/tok | f_draft 0.55   (was 253 ms/tok, f_draft 0.90)
```

But it required the target to offload 5 layers, which (a) dropped the target to 14 t/s and (b) hit
the offloaded-verify path → with the guard, verify falls to CPU; without it, output diverged. Net
on 6 GB: **still slower than plain full-resident decode**, so the gate (2) correctly **declines**:

| path (this box) | target | draft | verify | result |
|---|---|---|---|---|
| matrix / default | full GPU-resident (41 t/s) | CPU (~253 ms/tok) | GPU batched | **lossless ✓, draft-bound (unchanged)** |
| coexist, gate ON (6 GB) | full GPU-resident | CPU (declines) | GPU batched | **lossless ✓ — same as above (no regression)** |
| coexist, gate OFF (forced) | 5 layers offloaded (14 t/s) | **GPU-resident (13 ms/tok)** | CPU (after guard) | lossless ✓ but net-slower; not a win |

Regression check (gate on): n-gram 1.09× lossless ✓; draft-gpu matrix lossless ✓, GPU verify, target
full-resident — identical to the committed Phase-1 behavior.

## Honest verdict — the 0.6B cannot stay on GPU on THIS 6 GB box

The two models' **weights alone are 4315 + 639 = 4954 MiB**; with CUDA context/kernel/scratch
overhead (~200–300 MiB) that already approaches the **5122 MiB free**, leaving no room for either
KV cache. So Camelid cannot hold both **fully** resident here — and any target **offload** to make
room is a net loss (slow target) and forces verify off the fast GPU path. The fix is therefore
**correct but dormant on 6 GB**: it engages on a GPU with enough free VRAM to hold both fully
resident (≈8 GB+), where the draft is ~free (13 ms/tok) and verify stays GPU-batched and lossless.

This matches Phase 3: llama.cpp wins on the same 6 GB box because its **f16 KV cache** (half of
Camelid's f32) plus lower overhead let both models sit resident. The real 6 GB unlock for Camelid is
**f16 resident KV** (a resident-attention kernel change) — and even then the margin is thin
(weights + overhead alone ≈ 5.2 GB). Until then, on ≤6 GB the lossless draft-model path stays
draft-bound; **n-gram remains the shippable win** (Phase 1/4).

## Reproduce

```
# draft fully resident only engages where both models fit resident (≥~8 GB free):
camelid bench-speculative <4B> --drafter draft --draft-model <0.6B> --spec-only \
  --draft-tokens 4 --workload code --max-tokens 128 --prompt-file prompts/code.txt
# CAMELID_RESIDENT_TRACE=1 shows the coexist gate decision + per-engine VRAM sizing.
# CAMELID_SPEC_DRAFT_CONTEXT caps the draft KV (default 512).
```

---

# Part 2 — f16 KV + coexistence: the 0.6B now stays on GPU (commits 683238fc, 1e987b85, de130301)

Part 1 concluded the 6 GB unlock needed f16 resident KV. Done — and it turned out to be a free,
bit-identical change because the resident engine ALREADY f16-rounds every K/V value (`kv_scatter`
calls `f16_round`); it just stored them in f32 containers. Storing the f16 bits (u16) directly
halves KV VRAM with zero numerical change (the attention kernels read back via `f16_bits_to_f32`).

## What landed

1. **f16 resident KV** (683238fc): KV cache stored as `u16` (f16 bits) instead of `f32`. All 13
   `cuda_resident` GPU parity tests pass (per-kernel + full_forward + verify_batch + prefill→decode)
   — bit-identical. Halves KV VRAM for every resident model; general win.
2. **Coexistence headroom + bounded spill** (1e987b85): `CAMELID_SPEC_COEXIST_HEADROOM_MB` (small
   per-engine margin under coexistence) + `CAMELID_SPEC_COEXIST_SPILL_MB` (bounded WDDM spill).
   Default-off / unchanged for single-model.
3. **Fast resident draft loop** (de130301): the draft's pending re-ingest + sequential steps now
   ride the GPU-argmax lane (no full-logits copy). Restored draft accept 48% → 92.5% and improved
   the draft-model path S_sync 0.10× → 0.77×.

## Result: both models fully GPU-resident on 6 GB, lossless

With f16 KV + tight contexts (no spill needed), the trace shows BOTH resident, 0 layers offloaded:

```
[resident-cuda] VRAM sizing: ... weights 4315 MiB (resident 4315; 0/36 offloaded) -> cap 384   (target)
[resident-cuda] VRAM sizing: ... weights  639 MiB (resident  639; 0/28 offloaded) -> cap 256   (draft)
draft γ=4 spec-only | accept 92.5% | gpu/cpu verify 14/0 | LOSSLESS ✓
```

The 0.6B **stays on GPU** — the literal goal. Lossless, GPU batched verify.

## Honest ceiling: still not a *speed* win on this 6 GB laptop GPU

| draft | latency | why |
|---|---|---|
| 0.6B alone (bench-generate) | **7.5 ms/tok** | native resident decode |
| 0.6B as draft, coexisting with 4B | **~31 ms/tok** | GPU resource contention when both models are resident |
| 4B target plain decode | 27 ms/tok | the bar a draft token must beat |

S_sync on 6 GB peaks at **0.77×**. The blocker is now neither residency, paging, nor the draft
loop (all fixed) but **coexistence contention**: a resident 0.6B that decodes at 7.5 ms alone slows
to ~31 ms when it shares the 6 GB card with the resident 4B. Since 31 ms/draft-token > the 4B's own
27 ms/token, **drafting costs more than it saves on this hardware, for any draft window** — so spec
cannot beat plain decode here. (llama.cpp's draft runs ~4 ms/tok; its 1.37× win depends on the
draft staying that fast under coexistence, which Camelid's stack does not achieve on this card.)

## Net

- f16 KV: shipped, bit-identical, halves KV VRAM everywhere — a real general win and the enabler.
- The 0.6B now stays GPU-resident on 6 GB, lossless (goal met).
- Draft-model spec improved 0.10× → 0.77× on 6 GB; it becomes a **win on a GPU without the
  coexistence contention** (more headroom / sustained clocks), where the draft keeps its ~7.5 ms.
- Remaining work for a 6 GB win: eliminate the coexistence per-token contention (GPU profiling;
  likely clock/occupancy/stream-scheduling on the shared laptop GPU) — beyond residency/VRAM.
- On ≤6 GB today, **synchronous n-gram remains the shippable win** (1.3–1.6× code/JSON, single model).

---

# Part 3 — profiling the "coexistence contention" (commit 16eec2d2): it isn't contention

Part 2 attributed the remaining S_sync < 1 to "GPU resource contention when both models are
resident." Profiling disproves that. Added per-draft-step counters (`[draft-profile]`: summed GPU
forward µs vs wall draft µs, resident vs CPU-fallback step counts) + `nvidia-smi` clock/throttle
logging.

## What the profile shows

| signal | finding | conclusion |
|---|---|---|
| GPU SM clock, 0.6B alone | 1755 MHz, throttle 0x0 (none) | — |
| GPU SM clock, coexistence | ~1740 MHz, throttle 0x4 (SW power cap) | ~1% slower — **clocks not the cause** |
| draft GPU-forward fraction | **100%** (wall draft == summed forward µs) | **no sync stalls / overhead around the forward** |
| draft resident vs CPU steps | 237 resident, **0 CPU-fallback** | draft genuinely runs resident |
| draft per-step (short prompt) | **~10.5 ms/step** | the 0.6B's native resident decode rate |

The "7.5 ms alone vs 31 ms coexist" gap from Part 2 was a **conflation**: 7.5 ms was a warmup-boosted
number; steady-state 0.6B decode is ~10–13 ms, and the draft runs at exactly that. The earlier
~31 ms/tok was inflated by the **first round re-ingesting the 88-token prompt as 88 sequential
decode steps** (the batched-prefill alternative was tried and **desyncs the drafter's resident
engine → accept 90% → 41%**, so token-by-token re-ingest is kept). With a short prompt the draft
is a clean ~10.5 ms/tok.

## The real reason draft-model spec doesn't win here

It's the **draft/target speed ratio**, not contention. Camelid's 0.6B decodes at ~10.5 ms vs the
4B's ~26 ms — a **0.4× ratio**. Clean steady-state economics (short prompt, γ=6, accept 54%):

```
spec round = draft 65 ms + verify 58 ms = 123 ms  →  4.16 tokens
plain      = 4.16 × 26.1 ms             = 109 ms  →  4.16 tokens     → spec 0.89×
```

The draft (65 ms, 6 drafts at 10.5 ms of which ~2 are rejected) + the batched verify (58 ms) exceed
plain-decoding the same tokens. **Why the 0.6B is "only" 0.4×:** Camelid's resident decode carries
~8–12 ms of **FIXED per-token overhead** (≈168 kernel launches across 28 layers + per-layer CPU
orchestration + one host sync) — bandwidth alone is ~2 ms. That fixed cost is ~constant across model
sizes, so it dominates the small draft. llama.cpp's 0.6B runs ~4 ms (CUDA graphs / fused launches →
near-zero fixed overhead), a **0.22× ratio**, which is exactly why it wins 1.37× on this same card.

## The lever for a 6 GB win (next, separate work)

Cut the fixed per-token resident-decode overhead — **CUDA graphs** (capture the per-token launch
sequence once, replay it) and/or kernel fusion — which collapses the ~168 per-token launches into
~one replay. This helps small models most (the draft), pulling the 0.6B toward llama's ~4 ms and the
ratio toward 0.2×, at which point the draft-model path wins on 6 GB. It also speeds the 4B target
decode (closing part of the 0.73× base-kernel gap). This is decode-kernel work, independent of the
VRAM/residency/coexistence changes in Parts 1–2.

## Net (all parts)

- f16 KV (683238fc): shipped, bit-identical, halves KV VRAM everywhere.
- Both 4B + 0.6B fully GPU-resident on 6 GB, lossless (goal met); verify-offload correctness guard.
- Draft-model spec improved 0.10× → ~0.88× on 6 GB (resident draft + fast draft loop).
- Profiled the residual: it is NOT coexistence contention but the 0.6B's per-token decode overhead
  (fixed launch/orchestration cost). The remaining win needs CUDA-graph/fused decode, not more
  VRAM work. On ≤6 GB today, synchronous n-gram remains the shippable win.

---

# Part 4 — CUDA graphs: already built, and CONFIRMED not the lever (corrects Part 3)

Part 3 named CUDA graphs as the fix for the small-draft decode overhead. Investigated — and it's
wrong. The CUDA graph decode path **already exists** in `cuda_resident.rs` (`decode_graph`,
`forward_token_greedy_graphed`, capture-once/replay), gated `CAMELID_CUDA_GRAPHS=1`, default-off
with a documented prior finding: on the RTX 3060 it saved nothing (3B 53.2→52.5, TinyLlama
129→124 tok/s — slightly *slower*).

Measured it on the small model, the case Part 3 assumed would differ:

| config | tok/s |
|---|---|
| 0.6B alone, graphs OFF | 81.78 |
| 0.6B alone, graphs ON | 80.72 |
| coexistence spec, graphs ON | draft falls to CPU (0 resident steps), S_sync 0.44× (worse) |

**No benefit on the small model either.** The earlier "~168 kernel-launch overhead" framing was
the error: `forward_us == wall (100%)` does NOT mean launch-bound — it means the host **waits for
the GPU to finish**, so the ~12 ms IS GPU *execution* time. The ~168 launches enqueue ahead
asynchronously, so their host cost is already hidden behind GPU execution (precisely why graphs,
which only remove launch overhead, save nothing). The GPU is genuinely busy ~12 ms running the
small kernels: ~3–6 ms real work (weight reads, attention) + ~6 ms inefficiency (small-GEMV
occupancy, per-kernel startup, inter-dependent-kernel gaps across ~168 kernels/token).

## The actual lever: kernel FUSION (not graphs)

To cut GPU *execution* time for the small draft, fuse the ~6 kernels per layer (rmsnorm+quantize,
the QKV/gate/up GEMVs, rope+kv_scatter, attention, silu+down, residual) into a few larger kernels —
fewer launches AND fewer dependent-kernel gaps AND better occupancy. This is what gives llama.cpp's
0.6B ~4 ms (vs Camelid's ~12 ms). It is a substantial resident-decode kernel rewrite with full
parity re-validation — a different and much larger effort than graphs, and the real remaining work
for a 6 GB draft-model speculation win. (It also speeds the 4B target, narrowing the 0.73× base gap.)

## Corrected net

- CUDA graphs: already implemented + gated; **confirmed not helpful on this GPU** (decode is
  GPU-execution-bound, not launch-bound) — keep default-off.
- The small-draft decode cost is GPU kernel-execution time → needs **kernel fusion**, not graphs.
- Everything from Parts 1–2 stands (f16 KV shipped; both models resident on 6 GB, lossless).
- On ≤6 GB today, **synchronous n-gram remains the shippable win**.

---

# Part 5 — kernel fusion (commits 6f69acfa, dcc8d53a): real but ~5%, glue isn't the bottleneck

Part 4 named kernel fusion as the lever for the small-draft decode cost. Implemented three fusions
in the resident single-token decode, each **bit-identical** (all 13 cuda_resident parity tests pass,
incl. full_forward_token_matches_cpu), default-on with `CAMELID_RESIDENT_NO_FUSION=1` to A/B:

- **F1** `rms_norm_quantize`: fuse each RMS-norm with the following Q8_0 quantize (2/layer + output).
- **F2** residual-into-GEMV: the O- and down-projection GEMVs write `out[row] += acc` straight into
  the hidden buffer, dropping the residual_add launch + projection round-trip (2/layer).
- **F3** `silu_mul_quantize`: fuse SiLU(gate)*up with the down-proj input quantize (1/layer).

Together they remove ~5 launches/layer (~140/token on a 28-layer 0.6B) plus several f32 round-trips.

## Result: +5.3% on the small model, ~0 on the large

| model | fusion off | fusion on | Δ |
|---|---|---|---|
| 0.6B decode (bench-generate) | 78.0 t/s | 82.1 t/s | **+5.3%** |
| 4B decode | 36.7 t/s | 36.95 t/s | +0.8% (bandwidth-bound) |
| coexistence spec S_sync | 0.88× | 0.88× | within noise |

Real and free (bit-identical, no regression), and it helps small models most — but **~5% is the
ceiling of glue-kernel fusion**, and it's within run-to-run noise for the spec, which stays 0.88×.

## Why fusion didn't move the needle (the honest finding)

The per-kernel scheduling gap is only ~5 µs, so removing ~140 launches saves ~0.7 ms of ~12 ms
(~5%). The other ~11 ms is the **GEMV and attention kernels' own execution time** — real memory/
compute work plus their inefficiency (the q8_gemv runs at ~76% of peak DRAM; the decode attention
is a simple non-tensor-core kernel). That is the **0.73× base-kernel gap** vs llama.cpp, and it is
what makes Camelid's 0.6B ~12 ms vs llama's ~4 ms. Glue fusion can't touch it.

## The actual remaining lever (deeper, separate)

Faster **GEMV and attention kernels**: tensor-core / MMA Q8 GEMV, a FlashAttention-style fused
attention, better occupancy/vectorization. That is what closes the 0.73× base gap (helping every
model, target and draft) and is the only thing that gets the 0.6B draft near llama's ~4 ms so
draft-model speculation wins on 6 GB. It is a substantial kernel-engineering effort of its own.

## Net (Parts 1–5)

- f16 KV (bit-identical, halves KV VRAM) + coexistence → both 4B+0.6B fully GPU-resident on 6 GB,
  lossless (the literal goal, met); verify-offload correctness guard.
- draft-model spec 0.10× → ~0.88× on 6 GB.
- Profiled the residual to GPU kernel-execution time (not contention, not launch overhead).
- CUDA graphs: already built, confirmed not helpful (decode is GPU-bound). Kernel **glue** fusion:
  done, bit-identical, +5.3% small-model — but glue isn't the bottleneck.
- The last lever is GEMV/attention **kernel-execution** efficiency (the 0.73× base gap) — a deeper
  effort. On ≤6 GB today, **synchronous n-gram remains the shippable win**.

---

# Part 6 — q8_gemv memory micro-opt (commit 3fa34f75): marginal, and the draft can't benefit

Tensor cores were ruled out (batch-1 GEMV is M=1 + memory-bound). The honest "faster GEMV" target is
the measured ~76%→86% peak-DRAM gap. The kernel is latency-bound (~60% DRAM at coalesced access — too
few in-flight loads), so the lever is memory-level parallelism: unroll the per-lane block loop to
issue several weight loads before the dp4a.

Implemented (U=4), bit-identical (13/13 parity tests). Result:

| model | before | after | Δ |
|---|---|---|---|
| 4B decode | 36.95 t/s | ~37.4 t/s | +1.3–1.6% |
| 0.6B decode | 82 t/s | 82 t/s | flat |

**It does not close the gap, and the draft can't benefit — for a structural reason.** The unroll only
has room when `blocks_per_row` is large. The 0.6B's projections have few blocks/row (hidden 1024 → 32
blocks; 32 lanes cover them in ~1 pass), so U degenerates to 1 — no MLP to gain. The small-model GEMV
is **not** within-warp-latency-bound; it is **work-starved**: batch-1 of a small model has too few
rows/blocks to saturate the GPU per kernel. No GEMV micro-opt fixes "not enough work." The 4B gains a
little (80 blocks/row → some unroll room), but even there the gap to llama's 86% needs a ground-up
kernel redesign (different work decomposition), not a loop tweak — and that still wouldn't help the
draft.

## Final conclusion of the kernel-perf arc (Parts 4–6)

Every decode-kernel lever has now been tried and measured:
- CUDA graphs — no benefit (decode is GPU-execution-bound, not launch-bound).
- Glue-kernel fusion (F1–F3) — +5.3% small models, bit-identical.
- GEMV MLP unroll — +1.5% large, flat small.
- Tensor cores — ruled out by hardware (M=1, memory-bound).

None makes draft-model speculation win on 6 GB, because the draft's cost is the **0.6B's batch-1
decode being work-starved on the GPU** — an architectural property of single-token decode of a small
model, not a kernel inefficiency. llama.cpp's ~4 ms comes from a fundamentally more optimized
small-batch decode path (kernel design + scheduling), a large from-scratch rewrite with uncertain
payoff and no bearing on the shippable result.

**Shipped & real across the whole effort:** f16 KV (bit-identical, halves KV VRAM), both models
resident on 6 GB lossless, verify-offload correctness guard, glue fusion (+5.3% small), GEMV unroll
(+1.5% large) — and the synchronous **n-gram win (1.3–1.6× code/JSON), which remains the shippable
≤6 GB speedup.** Draft-model speculation on ≤6 GB is not reachable by these levers.
