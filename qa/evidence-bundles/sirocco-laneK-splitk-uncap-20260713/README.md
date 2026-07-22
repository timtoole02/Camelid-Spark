# SIROCCO Lane K — `SPLITK_MAX` uncap (16 → 32)

**Machine:** RTX 3060 Laptop (GA106, CC 8.6, 30 SM), 6 GB, CUDA 12.9, WDDM, Win11
**Change:** raise the decode split-K cap `SPLITK_MAX` from **16 to 32** (`src/cuda_resident.rs`), mirrored in lockstep in the two CUDA spec-verify emulations (`attention_batched`, `attention_tree_batched`).
**Result:** **+5.96 %** end-to-end decode at ctx≈8.8k (thermal-robust interleaved A/B — the one measured e2e point), **token-identical** to the CPU oracle and **bit-identical** to the spec-verify path. Expected to grow with context (V-read byte-share rises; not A/B'd across contexts). No change below ctx 512.

## Why (occupancy, not vectorization)

This is the **opposite** of the rejected experiment #6 (which *vectorized* the split-K V read with `uint4` and lost, −18%, by dropping the active thread count to a quarter-warp). Here we keep the kernel and add **more split blocks**. The decisive microbench (`micro-splitk.cu`, `microbench-summary.txt`) shows why:

- `attn_sk_partial` (the **V read**) uses only `head_dim`=64 of its 256 block threads, so it is **occupancy-limited** — it runs at 92 GB/s pinned at the old cap of 16 splits, and rises to **110 GB/s (+19 %) at 32**, plateauing near 116 by 128.
- `attn_sk_scores` (the **K read**) is already coalesced/optimal at 16 and *regresses* with more splits — so 32 is the knee: it banks the V-read plateau with minimal K-read give-back and combine overhead.

At ctx≈8.8k the V read is only a fraction of total decode bytes (model weights + the full KV cache dominate), which is consistent with the clean +19 % kernel win (measured at ctx 32k) showing up as **+~6 %** end-to-end. The +6 % at 8.8k is the one directly measured e2e point; by the byte-share model the V-read share — and the win — should rise with context, but that trend was not A/B'd across contexts.

## Correctness (`gate-results.txt`)

The uncap is **token-parity, not byte-identity** — it re-groups the split-K f32 reduction (n_splits 16→32), so attention output bits differ while greedy tokens do not. It is held to the same multi-prompt bar the rejected #8 dp4a change failed:

1. **Gate 1 — bit-identity spec-verify** (`splitk_spec_verify_bit_identical`, device test): decode ≡ the unchanged emulations across the new **17..22** split counts (sweep widened past the old cap; ceiling is the tree emulation's 48 KB shared budget, not the real path). **Passes** ⇒ lossless speculative decode preserved.
2. **Gate 2a — reassociation robustness**: same binary except the cap; **96/96** greedy tokens identical, old n_splits=16 vs new n_splits=26 @ ctx 6602.
3. **Gate 2b — saturated (n_splits=32) vs CPU oracle**, three distinct prompts: **64/64** identical at ctx 8802 / 10232 / 12211.

Why token-parity is safe here but wasn't for #8: #8 changed the GEMV *arithmetic* (per-layer precision loss compounding across 16 layers → flipped 8/8 prompts). This changes **no GEMV** — only the split *count* of the online-softmax reduction, split-invariant up to one f32 rounding per attention, and it additionally clears the bit-identical Gate 1 the dp4a kernel never could.

## Files

- `micro-splitk.cu` — the decisive n_splits sweep (lifts `attn_sk_scores` + `attn_sk_partial` verbatim).
- `microbench-summary.txt` — the sweep numbers (K-read vs V-read vs n_splits).
- `gate-results.txt` — the three correctness gates in full.
- `interleaved-ab.txt` — the 4-pair thermal-robust end-to-end A/B (median +5.96 %).

_Measurements: monitored-boost (hardware clock pin admin-gated on WDDM); mem clock stable at 6000 MHz. A/B is interleaved OLD/NEW per pair so the per-pair ratio cancels SM-clock drift._
