# SIROCCO Lane K — opening result — `camelid.speed-receipt/v1`

**Date (UTC):** 2026-07-12
**Machine:** RTX 3060 Laptop GPU (GA106, CC 8.6), 6 GB, driver 576.83, CUDA 12.9, WDDM · Win11
**Camelid:** `main` @ 334653d8
**Lane:** K (kernel campaign) — selected by [SIROCCO Phase 0](../sirocco-phase0-roofline-20260712/README.md) (G0: `C ≈ 0`, kernel-bound).

## What Lane K recon found (and refuted)

Phase 0 said the ctx≈0 headroom is in the kernels, not the host. Lane K localized it with a standalone
Q8-GEMV microbenchmark (`micro-gemv.cu`, replicates `q8_gemv`'s access pattern) and direct flag A/Bs:

| Finding | Evidence | Consequence |
|---|---|---|
| **The Q8 GEMV is already near-peak on big matrices** | microbench: LM head **101%**, FFN gate/up **91%**, FFN down **84%** of `BW_peak` (270.9 GB/s); `U=4` unroll already ≈ optimal | A `U`-bump PR would win ~nothing — **hypothesis refuted before writing it.** |
| **Small attention-projection GEMVs under-saturate** | microbench: attn q/o **68%**, k/v **43%** (occupancy-limited; `grid = rows/8` → k/v = 64 blocks on 30 SMs) | Real GEMV target is split-K on the narrow projections (bit-exact-preservable; ~13% of B → modest). |
| **CUDA Graphs HURT at ctx≈0** | direct A/B: `CAMELID_CUDA_GRAPHS=1` = **−5%** (128→121 tok/s, 3/3 rounds) | Graphs force worst-case fixed-shape attention sizing (empty-block tax). **Confirms Lane A/A2 is empty — by test, not inference.** |
| **★ Coalesced attention wins at long context** | this receipt | Promote `CAMELID_ATTN_COALESCED` to default-on for the split-K path. |

## The win — `CAMELID_ATTN_COALESCED` (existing kernel, default-off)

Phase 0's PC3 found the decode **attention KV read runs at ~10–20% of peak** — scalar `unsigned short`
f16 loads through a software converter (`f16_bits_to_f32(kp[d])`), no coalescing. Camelid already ships
a coalesced variant `attn_sk_scores_coalesced` (one warp per key position, adjacent-`half` coalesced K
loads + warp-shuffle reduction) in the split-K path, but it is **gated off** (`CAMELID_ATTN_COALESCED`,
default false). Turning it on:

| ctx (tok) | baseline tok/s | coalesced tok/s | Δ (interleaved) |
|---|---|---|---|
| ~1542 | 87.46 / 81.73 | 89.15 / 84.21 | **+1.9% / +3.0%** |
| ~2002 | 81.10 / 79.45 | 86.88 / 83.32 | **+7.1% / +4.9%** |
| ~3302 | 61.44 / 56.17 | 65.10 / 60.64 | **+6.0% / +8.0%** |

**The gain scales with context** (more KV positions read → more benefit from coalescing the read).
It is a **no-op at ctx≈0** (split-K activates only at `attn_shared > 512`) — **zero headline regression.**

### Correctness — token-identical (the gate)

Greedy decode, `temperature 0`, on a **varied coherent prompt** (ctx 1542, output tokens
`[578,15140,49267,279,34681,...]`): baseline vs `CAMELID_ATTN_COALESCED=1` produce **byte-identical
output token ids** (both 309 bytes, `diff` clean). The attention path is token-parity by design, and the
runtime parity self-check (`ensure_resident_parity_verdict`, `bit_exact` greedy probe) gates it. This is a
speedup with **no output change**.

## ⛔ DEFAULT-ON REJECTED — it breaks lossless speculative decode (the important finding)

I flipped the default to on, rebuilt, and it passed the obvious gates — GPU-resident load, ctx≈0
no-regression (126.96 vs 127.23), +4.8%/+7.4% long-ctx, and **greedy token-identity (decode-on vs
decode-off)**. Those all looked green. **Then the stricter CI test caught the real problem:**

`splitk_spec_verify_bit_identical` (`cuda_resident/tests.rs:1111`) asserts three paths are **bit-identical**:
the **decode** path (`launch_attention_splitk`), the **linear spec-verify** (`attention_batched`), and the
**tree spec-verify** (`attention_tree_batched`). The verify paths use **non-coalesced** kernels. With
`CAMELID_ATTN_COALESCED` on, the decode path re-associates the per-position dot (warp-shuffle) → decode
is **no longer bit-identical to spec-verify** → **the lossless speculative-decode guarantee breaks at
ctx>512.** The test *self-skips* with coalesced on (message quotes exactly this), so a naive default-flip
would have **silently shipped a correctness regression while turning off the test that guards it.**

My greedy token-identity check only compared decode-vs-decode; it never compared decode-vs-spec-verify,
which is the actual lossless property. **The default flip was reverted.** The kernel is correctly OPT-IN:
enabling it trades the ctx>512 lossless-spec-decode guarantee for +5–8% long-context greedy throughput.

**Correct recommendation:** keep `CAMELID_ATTN_COALESCED` **opt-in** (default-off). Document it as a
"long-context greedy speedup; not compatible with lossless speculative decode at ctx>512." The
default-on-able Lane K win must be a **bit-identity-preserving** KV-read vectorization (same math order,
just wider loads) — see the follow-up work; that keeps decode == spec-verify and can ship default-on.

**Next Lane K targets (ranked):** (1) extend coalescing / `half2`-vectorize the V read and
`attention_decode` (non-split-K) KV loads — the bulk of the ~10–20% under-saturation remains; (2) bit-exact
split-K on the narrow attention **projection** GEMVs (68%/43% → ~85%). The big FFN/LM-head GEMVs are done.

**Files:** `micro-gemv.cu` (GEMV microbench), `micro-gemv-results.txt`, `coal-ab.txt` (A/B logs),
`tokens-base.json` / `tokens-coalesced.json` (identity proof), `speed-receipt.json`.
