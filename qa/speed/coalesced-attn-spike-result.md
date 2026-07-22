# Spike result: coalesced split-K attention K-dot under greedy-token parity

**Date:** 2026-06-21 · **Box:** RTX 3060 Laptop 6GB (sm_86) · **Model:** Qwen3-4B-Q8_0
**Flag:** `CAMELID_ATTN_COALESCED` (default OFF; `attn_sk_scores_coalesced` in `src/cuda_resident.rs`)

## Hypothesis
The 0.51× depth gap (26.2 vs llama.cpp 51.1 t/s @ ctx~1881) is the uncoalesced K-dot in
`attn_sk_scores` (each thread owns a whole key position; adjacent threads stride 256B → no
coalescing, ~5% of DRAM peak). Greedy-token parity (newly adopted) was thought to unblock
the previously-reverted coalesced read (it flipped tok26 under bit-exact parity). Spike:
warp-coalesced K-dot (lanes own d=L,L+32,...; warp-shuffle reduce) inside split-K.

## Result
- **PARITY: PASS.** Output token-id sha identical baseline vs candidate across synthetic,
  real-code, and 5 interleaved depth rounds (sha `702249DFF7DFF85D` / `DA8BDC0B…` / `9702…`).
  Determinism confirmed. The warp-shuffle dot re-association did NOT flip the greedy sequence
  on any tested prompt → greedy-token parity holds for this kernel.
- **SPEED: SPEED-NULL.** Interleaved A/B (5 rounds, throttle-controlled): candidate/baseline
  median **1.10×** (23.4 vs 21.3 t/s); at matched 1740 MHz clocks (round 1) only **+3.6%**.
  Below the +15% kill bar. (Earlier "divergence" seen during a 2-process VRAM thrash was a
  measurement artifact, not a real flip — its baseline sha also differed from clean.)

## Conclusion
Coalescing the K-dot does **not stack** on split-K: split-K already captured the occupancy
win, and the residual attention-read inefficiency recovers only ~+4–10% — confirming the
red-team projection and the campaign's "coalesce-only (26.8) and split-K-only (26.2) each
land ~26.5" finding. **The depth-attention coalescing lever is closed.** Kernel kept gated
default-off (additive, parity-safe, marginal long-context option); reverting is also clean.

## Pivot (where the real wins are)
- **Lane A — lossless tree verification** on the existing batched-verify kernel (no parity
  risk, ~0 VRAM): n-gram 1.25× → ~1.3–1.6× code/JSON. CPU seam done + 35 tests green.
- **Lane B — Q4_K fused-dequant GEMV**: headline = makes **8B fully VRAM-resident on 6GB**
  (kills host-offload, >2× for 8B); ~1.6–1.75× for 4B. All-Rust (dp4a).

Harness: `qa/speed/spike_speed_ab.sh` + `ab_summary.mjs` + `parity_check.mjs` (reusable).
