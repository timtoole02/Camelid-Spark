# SIROCCO Phase 0 — DENOMINATOR — `camelid.roofline-receipt/v1`

**Date (UTC):** 2026-07-12
**Machine:** RTX 3060 Laptop GPU (GA106, CC 8.6, 30 SM), 6 GB GDDR6, driver 576.83, CUDA 12.9, **WDDM** · i7-11800H (16T), 15.7 GiB RAM · Win11
**Camelid:** `main` @ 334653d8 (`target/release/camelid.exe`, CUDA-default Windows build)
**Question (SIROCCO §0):** Camelid does ~100–130 tok/s on Llama-3.2-1B/CUDA. *What is the denominator, and which lane closes the gap?*

## Verdict (G0)

> **Branch = Lane K (kernel campaign).** `C ≈ 0`, the whole decode token is a GPU forward pass that
> saturates only **67%** of memory bandwidth. Overhead-targeting lanes (A/A1/A2) are **empty on this
> build**. The headroom lives in the GEMV and attention kernels. **Phase 0 steered off the illustrative
> plan's Lane-A assumption — which is exactly its purpose.**
>
> **⚠ Two escalations:** (1) P3/Q4_K_M confound **fails** — K-quant CUDA decode is kernel-bound.
> (2) Decode **attention** reads the KV cache at ~10–20% of peak — a second, worse under-saturating kernel.

---

## I4 DEVIATION — read first

**The hardware SM-clock pin required by I4 could not be applied** (`nvidia-smi` clock control is
admin-gated and denied for the session user; persistence mode N/A on WDDM). Substituted a
**monitored-boost protocol**, justified and validated:

- Decode is **memory-bandwidth-bound**, and the **memory clock held at 6000 MHz across all 1959
  active samples — a single value, never floated** (session throttle log). The *governing* clock
  was stable; only the SM clock floated (1462–1755), and SM governs **`C`, measured ≈ 0**.
- **Thermal control:** interleaved A/B, **25 s cooldown between rounds**, 250 ms session throttle log;
  runs showing a HW/SW **thermal** bit (0x08/0x20/0x40/0x80) rejected. Variance bands reported.
- **Validation:** branch-critical window (regression + PC1) had **31 thermal samples / ~1000 active
  (~3%)**, per-run rejected; the of-record tok/s carry a **0.6–0.7% variance band**. 92% of all
  thermal events were in the truncated hot PC3 high-ctx prefills. SM sagged ≤ 8% under thermal —
  and SM multiplies `C ≈ 0`, so the bandwidth-bound figures are unaffected.

A hard-pin re-run (if an elevated shell is provided) would tighten `C`/`BW_eff` but **cannot change
the branch**, whose driver (`C ≈ 0`, mem clock stable) is clock-robust.

---

## 1. `B` — bytes read per decode step (exact, from the GGUF tensor table)

Summed `n_bytes` of every tensor **touched in one decode step** (dense Llama, **tied embeddings** →
`token_embd.weight` is the LM head, read in full each step). Not file size; not the §0 estimates.

| Point | Model | Quant | `B` (bytes) | `B` (GB) | LM head | attn | ffn |
|---|---|---|---|---|---|---|---|
| **P1** | Llama-3.2-1B | Q8_0 | 1,313,251,456 | **1.3133** | 21.3% | 13.6% | 65.2% |
| **P2** | Llama-3.2-3B | Q8_0 | 3,414,061,312 | **3.4141** | 12.3% | 21.9% | 65.8% |
| **P3** | Llama-3.2-1B | Q4_K_M | 799,862,912 | **0.7999** | 26.9% | 12.1% | 61.0% |

All three within 0.1% of the §0 sanity estimates. LM-head share at P1 = 21.3% (confirms §Lane-B "21%").
P3 requantized locally from P1's Q8_0 (`llama-quantize --allow-requantize`, 5.18 BPW) — no download.

## 2. `BW_peak` — measured, not spec sheet

Device-to-device STREAM triad (`a=b+q·c`, 3 words/element, 512 MB/array), nvcc 12.9, sm_86.
Nameplate = 192-bit × 6001 MHz × 2 / 8 = **288 GB/s**.

> **`BW_peak` = 270.9 GB/s** (median of 3×100 iters; min 264.8, max 272.1), SM 1755 / mem 6000
> = **94% of nameplate** (§0 expected 80–90%; this GPU does better).

## 3. `t` and the regression (0.1)

n = 256 output tokens, greedy, ctx ≈ 0, P1/P2/P3 **interleaved**, 5 rounds, 25 s cooldowns, thermal-rejected. p50.

| Point | tok/s p50 | ms/tok p50 | var band | **MBU** = B/(t·BW_peak) |
|---|---|---|---|---|
| P1 1B Q8_0 | 126.44 | 7.878 | 0.7% | **0.615** |
| P2 3B Q8_0 | 51.24 | 19.441 | 0.6% | 0.648 |
| P3 1B Q4_K_M | 108.04 | 9.401 | 6.5% (1 reject) | **0.314** |

Regress `t(ms)` vs `B(GB)` across the two **Q8_0** points:

> **slope = 5.50 ms/GB → `BW_eff` = 181.7 GB/s** · intercept `C` = 0.65 ms (regressed) · **`BW_eff/BW_peak` = 0.671**

Note MBU *rises* with size (P1 0.615 → P2 0.648): larger GEMVs occupy the GPU better — the small-matrix
under-occupancy the branch is about. The regressed `C = 0.65 ms` is not real overhead; it is the
2-point line absorbing that size-dependent efficiency. The **true** overhead is the direct measurement:

### Direct `C` (clock-robust) — `CAMELID_DECODE_TIME`

| | forward | forward % | sample | in-step+loop other | **C** |
|---|---|---|---|---|---|
| P1 | 7.67 ms | **100.0%** | 0.00 | 0.00 | **0.00** |
| P2 | 18.70 ms | **100.0%** | 0.00 | 0.00 | **0.00** |
| P3 | 8.87 ms | **100.0%** | 0.00 | 0.00 | **0.00** |

> **`C` = 0 ms → `C/t` = 0.** The entire decode token is the GPU forward pass. Sampling already runs
> on-device; host submission is hidden. **There is no overhead to remove.**

## 4. The branch (0.2)

Inputs: `MBU@P1` = **0.615** · `C/t_P1` = **0** (direct) · `BW_eff/BW_peak` = **0.671**.

| Condition | This box |
|---|---|
| `C/t ≥ 0.35` → overhead-bound → Lane A | **NO** (C/t ≈ 0) |
| `C/t < 0.15` & `BW_eff/BW_peak < 0.55` → kernel-bound → Lane K | near (0.671, just above 0.55) |
| `C/t < 0.15` & `BW_eff/BW_peak ≥ 0.75` → at the wall → Lane B,C | NO (0.671 < 0.75) |
| else → mixed → A1+A2, re-run | ← **table lands here** (0.55 ≤ 0.671 < 0.75) |

**Table verdict: mixed → A1+A2.** But the table's [0.55, 0.75) cell prescribes A1+A2, and **A1+A2 are
provably empty on this build**: `C ≈ 0`, on-device sampling already ships (A1's premise — a host-side
full-vocab sort eating 30–40% — is a **no-op** here), host submission gaps are 0 ms (A2's target).
Recoverable-from-Lane-A = `1/(1−0.9·C/t)` = **1.00× (0%)**.

> ### ▶ MECHANISM-HONEST VERDICT: **Lane K (kernel campaign)** — *not in the SIROCCO doc; open it.*
>
> With `C ≈ 0`, the whole token is a forward pass saturating only **67%** of bandwidth — the GEMV
> under-saturating DRAM. Corroborated by the committed **Phase-3 Nsight receipt** (`q8_gemv` 42% DRAM
> on narrow attention projections, never > 66%; 128 blocks / 30 SMs = under-occupied) and by
> **llama.cpp reaching 0.86** on the same 4B-Q8 vs Camelid's 0.65–0.67. Lane K work: vectorized loads,
> occupancy, in-register dequant, `cp.async` on the Q8 MLP/attention GEMVs — **narrow attention
> projections first** (and the decode attention kernel, §PC3).
>
> **Phase 0's whole point:** §Composite assumed Lane A (`C/t ≈ 0.5`); the box is the opposite
> (`C ≈ 0`, kernel-bound). **Running Lane A would have been running the wrong lane.**

## 5. Poison checks (0.3)

**PC1 — quadratic sweep** (n = 64/256/1024/2048): tok/s **130.60 / 127.05 / 124.71 / 124.55**.
n=2048 is **4.6%** below n=64 and the rate **plateaus** at high n (124.71→124.55). The f32 KV-growth
term (avg-position read) predicts ~4.8% → residual ≈ 0. → **PASS — no O(n²).** *(Incidentally confirms
the KV cache is f32: the f16 prediction of 2.4% would leave a 2.2% residual; f32's 4.8% matches.)*

**PC2 — wire ratio** (`serve` SSE vs engine, 5 cold prompts): engine 126.44, wire median **124.11** →
**`wire_ratio` = 0.982** (**PASS**, ≥ 0.90). One SSE flush/token (no coalescing). **API path is not the
bottleneck; `API_INVERSION_CONDUCTOR` does not block this campaign.**

**PC3 — context sweep** (n=256 decode; truncated at ctx 3153 — 16k/32k skipped: prefill > 3 min each
+ thermal at 74 °C; linear trend already established, **not** a silent cap):

| ctx (prompt tok) | tok/s | ms/tok | µs / ctx-token |
|---|---|---|---|
| 0 (2) | 127.06 | 7.87 | — |
| 790 | 102.00 | 9.80 | 2.44 |
| 3153 | 58.89 | 16.98 | 2.89 |

Per-ctx-token cost ≈ constant (~2.7 µs) → **LINEAR in ctx, no O(ctx²) bug.** **But** at ctx=3153 the
attention step reads ~206 MB of f32 KV, which at `BW_eff` should cost ~1.1 ms — it costs ~9 ms. **The
decode attention kernel reads the KV cache at ~10–20% of peak bandwidth** → a second, worse
under-saturating kernel. **Major Lane K target for long-context decode.**

## 6. ⚠ ESCALATION — P3 confound FAILS (§0.1 "stop and escalate")

P3 (Q4_K_M) sits **+86%** above the Q8 line — **MBU_P3 = 0.314** vs MBU_P1 = 0.615. On CUDA the doc
expects dequant ≈ free (P3 on the line). It is **not**: **Camelid's Q4_K_M CUDA decode is
kernel/dequant-bound**, ~half the bandwidth of Q8_0.

- **Independently corroborated** by the committed `llama-3.2-3b-q4_k_m-…-cuda-resident-speed.json`
  (26.6 tok/s → MBU ≈ 0.20; kernels `q4k_gemv`/`q6k_gemv`). Not an artifact of the local requant.
- **Consequence for Lane B/C:** §Lane-C's "r ≈ 0.62" is a **byte** ratio; the **time** cost of a Q4
  draft/head on this engine is worse (compounds T7/T8). Lane B1 (Q4 head) and Lane C3 (Q4 draft)
  economics must be re-derived from **measured Q4 time**; Lane K must cover the K-quant GEMVs.

## G0 gate

- [x] `roofline-receipt.json` + this README: device/driver/WDDM, clock state (**I4 deviation:
  monitored-boost, validated**), `BW_peak` measured, `BW_eff`/`C` regressed **and** direct, `MBU@P1`, all 3 sweeps.
- [x] PC1 PASS, PC2 PASS.
- [x] Branch selects **exactly one lane: Lane K** (table cell "mixed" disambiguated by direct `C ≈ 0`).
- [x] Escalations logged: **P3/Q4_K_M kernel-bound** (reshapes Lanes B/C) and **decode-attention
  under-saturation** (Lane K long-context).

**Next:** open the Lane K kernel campaign (GEMV + decode attention); do **not** run Lane A/A1/A2 as
first movers. Re-derive Lane B/C Q4 economics from measured Q4 time.

**Files:** `roofline-receipt.json` · `phase0-regress.json` · `phase0-pc1.json` · `phase0-pc3.json` ·
`wire-result.json` · `decode-time.json` · `B-values.json` · `triad.cu` · `phase0-measure.mjs` ·
`session-clocklog.csv`.
