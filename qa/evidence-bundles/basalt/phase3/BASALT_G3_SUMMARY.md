# BASALT Gate G3 — CPU inference + pre-registered quality eval

Status: **NO-GO** (pre-registered §5.2 rule applied verbatim). Campaign's central question
answered. **Awaiting Tim: scope decision (postmortem-and-stop vs continue-to-Phase-4 on
bandwidth grounds with the quality cost disclosed).**

Frozen engine: **SHA = `8038abba`** (branch `basalt/phase3-cpu-eval`). All legs ran on this
single SHA per Amendment 3 §3. Protocol: `basalt_eval_protocol.md` (incl. Amendments 2, 3).
Receipts: `legs/` in this bundle (`eval_legs_summary.json` is the master; per-row
`legs/<row>_metrics.json`; `legs/baseline_continuations.json`; `legs/legB_parity.json`).

**Phrasing lock (Amendment 3 §4, binding on every figure below):** all quality figures are
measured **vs the Q8_0 parent, at matched 4.5 bpw**. No figure here may be compressed to an
unqualified "NVFP4 ≈ Q4_K_M" or any absolute-quality claim. Absolute-vs-BF16 is a separate,
not-yet-registered eval.

## 1. The result

Teacher-forced top-1 agreement with the Q8_0 parent's greedy continuations, over the 9
lane-native prompts (296 emitted positions — EOS stopped some prompts before their token
budget; the denominator is actual emitted positions, per the protocol):

| row | agreement (vs Q8_0 parent, 4.5 bpw) | matches | mean KL (nats) | median | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| Q8_0 (baseline, sanity) | 100.0 % | 296/296 | 0 | 0 | 0 | 0 |
| **NVFP4-mm** (gated subject) | **88.5 %** | 262/296 | 0.1112 | 0.0055 | 0.454 | 6.25 |
| **Q4K-mm** (gated comparator) | **92.6 %** | 274/296 | 0.0646 | 0.0020 | 0.372 | 1.51 |
| Q4_K_M-df (report-only) | 90.5 % | 268/296 | 0.1169 | 0.0040 | 0.491 | 6.11 |
| Q4_K_M-im (report-only) | 95.3 % | 282/296 | 0.0335 | 0.0018 | 0.170 | 0.567 |

Sanity anchor exact: Q8_0-vs-itself = 100.0 %, KL = 0, all 296 forced-decode bins
byte-identical to the Leg A free run. Harness sound.

## 2. GO/NO-GO — applied verbatim

Pre-registered rule (protocol §5.2, signed at G0): **GO iff `agreement(NVFP4-mm)` ≥
`agreement(Q4K-mm)` − 2.0 percentage points.**

- `agreement(NVFP4-mm)` = 88.5
- `agreement(Q4K-mm)` = 92.6 → threshold = **90.6**
- Sanity guard: `agreement(Q4K-mm)` = 92.6 ≥ 80.0 → holds → the rule applies (no
  suspect-basis escalation).
- **88.5 ≥ 90.6 → FALSE → NO-GO.**

The gap is **4.1 points** — more than twice the 2.0-point tolerance. This is not a marginal
miss: NVFP4-mm is the **worst of the four produced rows** on agreement, and worse than the
format-isolated Q4_K comparator on mean KL as well (0.111 vs 0.065 nats). The result is
recorded as measured; no threshold was adjusted after seeing it.

## 3. What the format-isolated design bought us (why this NO-GO is trustworthy)

`NVFP4-mm` and `Q4K-mm` differ in **exactly one thing**: the 294 matmul weights are NVFP4
vs uniform Q4_K, both at 4.5 bpw, from the same Q8_0 parent, same tool, same keep-list,
byte-identical everywhere else (proven at G2). So the 4.1-point agreement gap and the ~1.7×
mean-KL gap are attributable to the **weight format alone** — not to a mixture, an imatrix,
or a different source. On this pilot, at matched bit-width, **Q4_K is the better format.**

Corroboration the answer is real, not a harness artifact:
- **Ordering is physically sensible**: imatrix-calibrated (95.3 %) > data-free Q4_K
  (92.6 %) > standard Q4_K_M mixture (90.5 %) > NVFP4 (88.5 %). Calibration helps most;
  NVFP4 helps least.
- **Consistent with upstream** (recon §8): published NVFP4-vs-Q4_K_M comparisons are
  "unsettled, possibly worse for small models"; a calibrated Qwen3.6-27B NVFP4 came out
  *behind* unsloth Q4_K_M. E4B is a small model. Our controlled result sharpens that folk
  knowledge into a matched-bpw number.
- **Engine sound**: the non-disturbance tripwire passed (Leg A byte-identical to the prior
  freeze), sanity anchor exact, all five rows ran.

## 4. Cross-engine token parity (Leg B) — passed independent of the quality verdict

Camelid NVFP4-mm CPU greedy vs the pin's `llama-completion` CPU greedy on the same NVFP4
file, 9 prompts: **8/9 token-identical.** The single divergence (village-story, step 13:
Camelid ` his` vs pin ` always`) is a **near-tie argmax flip** — the pin's token is
Camelid's own #2 at a **0.084 raw-logit gap**, inside the lane's probe-and-attribute
envelope; top-32 retained (`legs/legB_divergence_village-story.json`), no tolerance
widened. This confirms Camelid's NVFP4 decode faithfully reproduces the reference engine;
the quality NO-GO is a property of the **format**, not of our implementation of it.

## 5. What survives the NO-GO, and what doesn't

- **Quality parity with Q4_K at matched bpw: REFUTED for this pilot.** This is the G3 gate,
  and it failed. NVFP4 does not earn a "quality-competitive" claim here.
- **Decode-bandwidth motivation: untouched by this result.** NVFP4-mm still moves ~1.6×
  fewer matmul bytes per token than Q8_0 (measured G2 sizes; the tied Q8_0 head keeps it
  below the 1.889× matmul-only figure). That lever lives in Phase 4 (CUDA) and is a
  *separate axis* from quality. A NO-GO on quality does not by itself kill a
  "smaller + faster, at a stated quality cost" value proposition — but it does mean any such
  framing must carry this quality delta, front and center.
- **The engine work stands**: NVFP4 loads, admits (gemma4-scoped), decodes bit-exact vs the
  pin, and runs a full forward. That is real, receipted capability regardless of the verdict.

## 6. Decision — RESOLVED: Option B (Tim, 2026-07-17)

**Tim chose (B): continue to Phase 4 on decode-bandwidth grounds**, with the quality cost
disclosed on every surface. NVFP4 proceeds as a *space/speed* format, explicitly not a
quality-competitive one. The G3 NO-GO stands as recorded — Option B is a forward choice on a
separate axis, not a reversal. Binding disclosure (claim-lint, Phase 6): every user-visible
NVFP4 surface carries the G3 delta (88.5 % vs 92.6 % top-1 agreement, 0.111 vs 0.065 mean-KL
nats, vs the Q8_0 parent at matched 4.5 bpw). The original options, for the record:

Per the conductor, a G3 NO-GO "pivots to a postmortem + upstream comparison note, and that
is a publishable result too." Two honest paths:

- **(A) Postmortem-and-stop.** Seal G3 as the campaign's answer: at matched 4.5 bpw on
  Gemma-4-E4B, NVFP4 is measurably behind Q4_K on top-1 agreement and KL vs the Q8_0
  parent. Phase 4/5 (CUDA/Blackwell) do not run. The engine capability is documented as
  "supported, not quality-competitive," or held unshipped. Cleanest close.
- **(B) Continue to Phase 4 on bandwidth grounds, quality cost disclosed.** If the value of
  NVFP4 is bytes-per-token (fully-CUDA-resident headroom, decode throughput) and a
  disclosed ~4-point agreement / ~1.7× KL cost vs Q4_K is acceptable for that, Phase 4
  measures the actual speed/residency win and every surface carries the G3 quality delta.
  This treats NVFP4 as a *space/speed* option, not a *quality* one. The Phase 4 kernel
  design is already reconned and ready.

My read: the pre-registered question was **quality**, and the answer is a clean NO-GO —
(A) is the honest default. (B) is legitimate only if the campaign's real goal was bandwidth
all along and quality was the bar to clear rather than the point; that reframing is Tim's to
make, not mine. Either way, the number ships with its qualifier.

## 7. Items still requiring Tim's signature (carried from earlier gates, non-blocking)

- **§2.4 matrix-mechanism deviation**: the invariant-matrix enforcement is compile-time
  file-binding + test-time fn-name assertion (a fn *rename* fails the meta-test, not the
  build). Amendment 3 §2.4 permits substitutions only if strictly stronger; this is weaker
  on the rename axis. Disclosed in DECISIONS.md, flagged for your explicit nod.
- **D-B3 carve-out does not auto-sunset at G3**: gemma4 is outside `COVERED_ARCHITECTURES`,
  so deleting the pilot carve-out would re-refuse the pilot on the architecture axis. If the
  campaign continues, the carve-out must be kept or gemma4 deliberately promoted (D-B6
  territory). Draft options in `scratchpad/basalt-db6/D-B6_DRAFT.md`.

## 8. Bundle contents (`legs/`)

`eval_legs_summary.json` (master: every command verbatim, SHA `8038abba`, sha verifications,
73 RAM readings, per-leg durations) · `baseline_continuations.json` (Leg A, the forced-decode
oracle) · `<row>_metrics.json` ×5 (agreement + KL, top-32 excerpts) · `legB_parity.json` +
`legB_divergence_village-story.json` · `ram_log.csv`. Raw full-logit bins deleted after KL
computation (≈1.7 GB). Safety: one model process at a time, two-phase pin/Camelid capture,
≥4 GB free floor (all readings logged), zero box crashes, nothing written to git by the eval.
