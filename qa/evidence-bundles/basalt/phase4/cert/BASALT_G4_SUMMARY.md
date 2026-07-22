# BASALT Gate G4 — NVFP4 CUDA decode: CERT + perf (measured, this box)

Status: **CERT PASS; Option B EXECUTED (Phase 4b dp4a kernel upgrade landed).** The
pre-registered `__byte_perm` + `__dp4a` inner-loop upgrade recovered the speed the byte
reduction should have bought: NVFP4-mm CUDA decode **14.64 → 26.51 tok/s (+81 %)**, now
**faster than Q8_0 (26.51 vs 25.80)** and still 2.08 GB lighter in VRAM — parity held
**46/46 bit-identical**.

Engine: `basalt/phase4-cuda-decode` — G4 CERT/perf at `8c2de5bb` (kernel impl `892672ca`);
Phase 4b dp4a upgrade re-measured on the current branch HEAD (this commit).
Hardware: RTX 3060 Laptop, sm_86, 6144 MiB, driver 576.83, CUDA 12.9. All numbers
**measured on this box**, never general. Receipts: `qa/evidence-bundles/basalt/phase4/cert/`.

## 1. CERT — the wiring is correct end-to-end (PASS)

NVFP4-mm greedy, Camelid **CPU wire lane vs CUDA-resident lane**, 9-prompt lane-native pack:
**6/9 token-identical.** All 3 divergences are textbook near-tie argmax flips — in each the
CUDA token is exactly the CPU's **#2** candidate at a 0.047–0.111 raw-logit gap, attributable
to the accepted **CUDA f16-KV vs CPU f32-KV** difference (the same greedy-token contract the
shipping Q8_0 CUDA row runs under). No non-near-tie divergence → no kernel/wiring bug. The
first-ever full NVFP4 CUDA generation ran clean. Combined with the impl's 46/46 bit-identical
same-bytes gate, the kernel + raw-wire upload + dispatch + residency are proven correct.

## 2. Perf — measured (median of 5 warm runs, 128 greedy tokens)

| lane | tok/s | per-token read | achieved BW | % of 336 GB/s roofline | peak VRAM |
|---|---:|---:|---:|---:|---:|
| Q8_0 CUDA-resident (shipping baseline) | **25.80** | 5.182 GB | 133.7 GB/s | **39.8 %** | 5559 MiB |
| NVFP4-mm CUDA-resident (v1, scalar LUT) | 14.64 | 3.05 GB | 44.6 GB/s | 13.3 % | 3479 MiB |
| **NVFP4-mm CUDA-resident (Phase 4b, `__dp4a`)** | **26.51** | 3.05 GB | 80.8 GB/s | **24.0 %** | **3479 MiB** |
| NVFP4-mm CPU wire lane (reference) | 1.57 | — | — | — | — |
| pin-GPU (cross-engine context) | skipped | — | — | — | — (6.06 GB full-offload unsafe on 6144 MiB) |

Phase 4b re-measures **only** the NVFP4-mm CUDA row (dp4a touches nothing else); the Q8_0 and
NVFP4-cpu rows are carried unchanged from the G4 run. NVFP4 dp4a runs were tight
(26.41–26.86 decode tok/s across all 5; steady 2nd-half median 26.13), peak VRAM identical to
v1 at 3479 MiB on every run, VRAM verified freed to 0 after each load.

## 3. The honest headline — and what it means for Option B

You chose **Option B** (continue on **space/speed** grounds, quality cost disclosed). G4 + the
Phase 4b dp4a upgrade now deliver **both** halves of that thesis:

- **SPACE: confirmed (unchanged by 4b).** NVFP4-mm resides in **3479 MiB vs Q8_0's 5559 MiB —
  2.08 GB more free VRAM.** Per-token weight read shrinks a measured **1.70×** (format-isolated
  1.647×, matching the pre-registered ~1.6×; the extra comes from an incidental inp_gate/proj
  precision difference between the two files, disclosed).
- **SPEED at v1: not there.** The v1 scalar-LUT kernel ran at **0.57× the Q8_0 lane's speed**
  (14.64 vs 25.80 tok/s) *despite* moving 1.70× fewer bytes, because it was **compute-bound**:
  13.3 % of the DRAM roofline vs Q8_0's 39.8 %.
- **SPEED at Phase 4b: won.** The pre-registered `__byte_perm` + `__dp4a` inner-loop upgrade
  (the pin's `get_int_from_table_16` codebook expansion + 4-way int8 dot, replacing the scalar
  nibble-unpack + KV-LUT multiply) lifts NVFP4-mm CUDA decode to **26.51 tok/s — +81 % over v1
  (1.81×)** — and moves the kernel from 13.3 % to **24.0 %** of the roofline. It **did not reach
  the roofline** (Q8_0 still sits higher at 39.8 %, so the kernel is not yet fully
  memory-bound), but the ~1.8× throughput lift is enough to **overtake Q8_0: 26.51 vs 25.80
  tok/s (1.03×)** — because NVFP4 reads 1.70× fewer bytes/token, so a merely-competitive
  roofline fraction still wins in absolute tok/s. **Parity held: `nvfp4_gemv_matches_oracle`
  stays 46/46 bit-identical, worst rel diff 0.000e0** (the dp4a i32 sumi is identical to the
  scalar one by construction), so this speed came at **zero** cost to the CERT that passed.

**Claim-lint consequence (binds Phase 6):** the honest statement is now **"NVFP4-mm on this
box is both faster than Q8_0 (1.03×) and 2.08 GB lighter in VRAM"** — a measured decode-speed
win, not just headroom. It is a *narrow* win (1.03×, and still below the DRAM roofline, so a
future memory-bound kernel could widen it), and it is a **decode** figure on this specific
6 GB card; no surface may over-generalize it. The G3 quality delta still travels with every
NVFP4 surface unchanged.

## 4. Decision — Option B EXECUTED (Phase 4b landed)

Tim chose **(B) Phase 4b — the dp4a kernel upgrade** at G4. It is now done:

- **Executed:** `nvfp4_gemv`'s per-sub-block integer dot was rewritten from the scalar
  nibble-unpack + KV-LUT multiply to the pin's `get_int_from_table_16` `__byte_perm` codebook
  expansion + `__dp4a` 4-way int8 dot (ported exactly; `sm_86` has `__dp4a`, no arch change).
  The v1 scalar loop is preserved as a code comment for the before/after receipt.
- **Parity held:** `nvfp4_gemv_matches_oracle` = **46/46 bit-identical, worst rel diff
  0.000e0** (plus the sentinel-decode, residual-fusion, and even-bpr guards all green). The
  dp4a i32 result equals the scalar one by construction, so the CERT is untouched.
- **Result:** NVFP4-mm CUDA decode **14.64 → 26.51 tok/s (+81 %, 1.81×)**, 13.3 % → 24.0 % of
  the roofline. It **did not reach** the memory roofline (Q8_0 is 39.8 %) but the lift is
  enough to make NVFP4-mm **faster than Q8_0** (26.51 vs 25.80 tok/s, 1.03×) while keeping the
  2.08 GB VRAM headroom. **The "speed" half of Option B is now real** — a narrow, honest,
  measured win on this box.

Still-open follow-ons (not part of 4b, unchanged):
- **(C) The gpu_head lever** (bigger): the tied Q8_0 head is ~23 % of NVFP4-row bytes and
  stays resident. Attacking it (NVFP4 head / NVFP4-all residency) compounds the space win and
  the per-token byte reduction. Larger scope; a follow-on campaign bite.
- Closing the remaining roofline gap (24.0 % vs Q8_0's 39.8 %): the dp4a kernel overtook Q8_0
  by moving fewer bytes, not by saturating DRAM. A further memory-bound rework could widen the
  margin — noted, not scheduled.

Phase 5 stays **BLOCKED-HW** (no Blackwell) and Phase 6 (surface alignment) now carries the
updated honest story: NVFP4-mm is faster **and** lighter than Q8_0 on this card, with the G3
quality delta disclosed.

## 5. Also still awaiting your signature (carried, non-blocking)
- **D-B6 (TK3)** per-tensor admission — draft banked (`scratchpad/basalt-db6/`), Option A
  recommended (~5-line change).
- **§2.4 matrix-mechanism deviation** — disclosed in DECISIONS.md, needs your explicit nod.

## 6. Safety note
~20 GPU model loads, **zero incidents**: one GPU process at all times, VRAM checked before
every load and verified freed to 0 after every one; peak 5559 MiB (Q8_0); box clean at exit.
The pin-GPU comparator was **skipped on memory-safety grounds** (6.06 GB full-offload on a
6144 MiB card is not comfortable headroom) — a deliberate omission, not a failure.
