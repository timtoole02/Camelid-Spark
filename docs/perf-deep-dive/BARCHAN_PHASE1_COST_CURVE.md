# BARCHAN — Phase 1: the verify-cost curve

**Verdict: GATE 1 = KILL.** Per-round verify cost at k=15 is **9.25×** the k=1 cost, against a
KILL threshold of 3×. The width thesis is dead on this host.

N=5, all repetitions in one contiguous verified-quiet window.

**Consequence, per conductor Gate 1:** publish the curve, **drop Phases 3 and 4**, record the
negative. No latch re-tune, no branch sweep, no serve wiring.

---

## 1. What was measured

`bench-speculative`, Metal resident tree lane, `Llama-3.2-3B-Instruct-Q8_0`, column
`repetitive_extraction` (the column with the most verify rounds to average over).

`--draft-tokens k` sets the per-round tree budget to `(min(k+1,16), k)` = `(max_nodes, max_depth)`
(`src/main.rs`, `full_tree`). So the sweep is a **tree-width sweep**:

| `--draft-tokens` | 1 | 3 | 5 | 7 | 11 | 15 |
|---|---|---|---|---|---|---|
| max_nodes | 2 | 4 | 6 | 8 | 12 | **16** |

k=5 is the shipped default. k=15 saturates `TREE_MAX_NODES`.

- Binary `camelid v0.3.1-155-g38e0aaf`, **clean worktree**,
  sha256 `2efbbbc123aa627335693a87efb9800de4f6aa74795a2100dd0e07eb1c0d05cd`.
- Env: conductor §2.1 block, `CAMELID_SPEC_TREE_GATE=0` (ungated), `CAMELID_SPEC_CPU_VERIFY=0`.
- Configs **interleaved within each repetition**, `--warmup` on every run.
- Per-round verify cost uses the **A4-corrected** `verify_ms` (commit `38e0aaf`). The pre-A4 field
  conflated plain-step time and would have flattened this curve toward a wrong answer.

**Sample size: N=5, satisfied.** All five repetitions ran in one contiguous window on the pinned
clean binary (the runner refuses to start on a SHA mismatch). Load average and foreign-process
count were sampled before and after every run into `environment_audit.tsv`: across all 60 samples,
**zero** `rustc`/`cargo` processes were observed and load stayed within 2.10–3.84. Conditions were
homogeneous across reps, not merely interleaved within them.

An earlier attempt was aborted at run 18/30 by a concurrent session's builds and GPU benchmarks;
its clean reps 1–2 are retained in `target/barchan-p1-sweep/` as a preliminary read and agree with
this one (9.55× vs 9.25×, slope 40.18 vs 38.24 ms/node). A still earlier 13-run set was discarded
for having been built from a dirty worktree. Both are quarantined, not deleted.

---

## 2. The curve

Every cell: `lossless = true`, `first_divergent = -1`, `cpu_verify_rounds = 0`.

| k | max_nodes | actual mean n | verify ms/round — median [range] (spread) | ms/node | acc/round | s_sync — median [range] |
|---:|---:|---:|---|---:|---:|---|
| 1 | 2 | 2.00 | 58.4 [56–62] (9.2%) | 29.2 | 1.81 | **1.031** [1.021–1.080] |
| 3 | 4 | 4.00 | 119.7 [113–121] (6.6%) | 29.9 | 2.94 | 0.870 [0.854–0.898] |
| 5 | 6 | 5.92 | 193.1 [185–197] (6.3%) | 32.6 | 3.62 | 0.706 [0.679–0.720] |
| 7 | 8 | 7.77 | 285.9 [280–305] (8.8%) | 36.8 | 4.27 | 0.576 [0.567–0.595] |
| 11 | 12 | 11.52 | 425.0 [415–448] (7.7%) | 36.9 | 4.48 | 0.429 [0.418–0.440] |
| 15 | 16 | 14.95 | 540.7 [530–571] (7.5%) | 36.2 | 4.95 | 0.377 [0.373–0.381] |

The drafter genuinely fills the requested width — mean n tracks max_nodes to within 7% at every
point — so this is a real width sweep, not a saturated one.

**Internal control:** `normal_step_ms / normal_steps` held at **35.47 – 36.17 ms** across all six
configurations (a 2.0% band) while the verify curve moved 9.25×. The control did not move; the
measured quantity did.

### Least-squares fit over actual mean node count

```
verify_ms_per_round = 38.24 × nodes − 23.74        R² = 0.99715
```

- **Marginal cost per additional verified node: 38.24 ms**
- **Fixed per-round cost (intercept): ≈ 0** (slightly negative, −23.7 ms)
- **R² = 0.997** — the cost is linear in width to within measurement noise.

(Fit is over the *actual* mean node count from the `[metal-tree-verify]` traces, anchored on the
full trace prefix. A loose `n=(\d+)` pattern also matches the trailing `n=` of `round_seen=` in the
`[spec-tree]` trace and silently poisons the fit — 284 spurious matches vs 38 real ones on one log.)

---

## 3. What this refutes

**The §0.3 thesis is refuted.** It predicted that on a memory-bound M4, a k-row verify reads the
same weights as a 1-row decode, so verifying up to ~15 tokens should cost about what verifying 1
costs. Measured: cost is **linear in k** (R² = 0.997), at **38.24 ms per row against a 35.72 ms
plain decode step — a ratio of 1.07×**. Each additional verified row costs *slightly more than an
entire independent decode*. The
batched verify delivers **zero weight-read amortization**. A k-row tree verify is strictly worse
than k sequential decodes.

**The PIVOT hypothesis is also refuted.** The conductor's 1.5–3× branch — and my own Phase 0
prime suspect, per-round resident-engine teardown/reseed, plus `compact_tree_kv_path` — predicted a
*fixed* per-round cost dominating. The intercept is ≈ 0. There is essentially **no** fixed
per-round cost. Whatever is expensive is paid **per row**, not per round. That rules out the
mechanisms the PIVOT branch was designed to catch, and it is why this is a KILL rather than a
redirect at KV compaction.

**This also explains the 3060's 1.28×** without needing a drafter explanation: if per-row verify
cost is near a full decode there too, acceptance can never outrun it.

**Not yet attributed: GPU vs host.** The conductor's KILL wording names "the residual (not
gpu_busy) as the driver". This sweep establishes the *shape* (pure per-row, zero intercept) but
does not split GPU-busy from host time — the `gpu_busy_us` instrumentation was not built, because
the curve answered the gate without it. Both a non-amortizing GEMM and per-row host work are
consistent with linearity. **The attribution is open; the verdict is not.** No mechanism that costs
a full decode per row can make width pay.

---

## 4. Why the verdict is not sample-size-limited

The gate is 3×. The measurement is **9.25×** — a 3.1× margin over the threshold, at N=5:

- Between-rep spread is 6.3–9.2% of the median at every k; the widest single range (k=15,
  530–571 ms) is nowhere near the 3× line.
- s_sync reproduced in every rep: k=1 stayed above 1.0 in all five (1.021–1.080); k=15 landed in
  0.373–0.381.
- The curve is monotone across six configurations with R² = 0.997.
- The internal control (plain-step ms) held a 2.0% band throughout.
- The preliminary N=2 run in a different thermal window agrees (9.55×, 40.18 ms/node).

---

## 5. The one positive result

**k=1 — a 2-node tree — is the only configuration that beats plain decode**, at
**s_sync ≈ 1.031** (all five reps in 1.021–1.080). The arithmetic is consistent: 58.4 ms buys
1.81 emitted tokens = **32.3 ms/token against 35.7 ms/token plain**.

So the lane is not worthless — but **width is the wrong knob, and the optimum is the minimum
width**, which is the exact opposite of the campaign's premise. Every widening step past 2 nodes
loses, monotonically, because acceptance grows sublinearly (1.81 → 4.95, a 2.7× gain for 8× the
nodes) while cost grows linearly (9.25×).

Note the shipped default is k=5, which measures **0.706** on this column — i.e. the current default
is a **29% regression** versus plain decode whenever the ungated tree lane actually engages. The
`SpecLatch` exists to skip exactly these rounds, which is why the shipped gated path does not show
this; but it means the latch is doing load-bearing damage control, not fine-tuning.

---

## 6. Recommendation

1. **Close the width lane.** Do not run Phases 3–4. Record the negative.
2. **Do not ship a width change.** k=1 beating k=5 is a real signal, but it is one column on one
   host, and the gated path already suppresses the loss. Any default change needs its own
   evidence across the full column pack.
3. **If the lane is ever reopened, the question is "why does a verified row cost a full decode?"**
   That is a single, well-posed instrumentation question (`gpu_busy_us` in `verify_batch_tree`),
   and it is the only thing that could revive tree speculation on Apple Silicon. It is a *kernel*
   question, and conductor §1 currently rules the Q8 Metal kernel lane closed — so reopening the
   width lane requires reopening that first.

---

## 7. Artifacts

**`target/barchan-p1-n5/`** — the result of record. `records/` (30 JSON records, k{1,3,5,7,11,15}
× rep{1..5}), `logs/` (per-run stderr with `[metal-tree-verify]` and `[spec-tree]` traces),
`run-n5.sh` (SHA-gated on the pinned binary), `environment_audit.tsv` (load + foreign-process
count, pre and post every run), `sweep.log`.

`target/barchan-p1-sweep/` — preliminary, aborted at run 18/30 by host contention. Only
`k*-rep{1,2}` are admissible; agrees with the N=5 result. Also holds
`records-INADMISSIBLE-dirty-binary/` (13 runs discarded for a dirty-worktree build).
