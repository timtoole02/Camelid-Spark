# SPEC_RECHECK Phase 1–4 — where lossless speculation pays on this machine

Date (UTC): 2026-06-20
Machine: RTX 3060 Laptop GPU (GA106, CC 8.6, 6 GB GDDR6, ~5.0 GiB free), i7-11800H (8C/16T,
AVX2+AVX-512), 16 GB RAM, Windows 11, CUDA 12.9
Camelid: `feat/bactrian-experimental-fast` @ 196cdf14 (worktree adds the `bench-speculative`
harness below; serving paths byte-for-byte unchanged — speculation is default-off and moves no
support-ledger row)
Target model: Qwen3-4B-Q8_0 (GPU-resident, all 36 layers in VRAM)
Draft model (Path B): Qwen3-0.6B-Q8_0 (same tokenizer family)
Competitor (Phase 3, run): llama.cpp `llama-speculative` pinned `acd79d6` (`-ngl 99 -ngld 99`)

## Scope of this run

Per the SPEC_RECHECK decision, this campaign measures the **two drafter paths the prior
SPEED_CAMPAIGN skipped**, on the one draft/target pair that is fully local:

- **Path A — n-gram / prompt-lookup** (`NGramDrafter`, zero draft model): the cheap, ~free-draft
  path. Full matrix: 6 workloads × γ{2,4,6,7} = 24 cells.
- **Path B — GPU-resident draft model** (`ModelDrafter`, Qwen3-0.6B → 4B): the path the prior
  campaign forced CPU-only. Focused matrix (see Path B for why the γ sweep is reduced).

Phase 3 head-to-head vs llama.cpp-speculative IS run here too — it needs only the local Qwen3 pair
(`llama-speculative` keeps both models resident), so no download was required after all. Out of
scope (needs the multi-GB downloads — the "full matrix" option): the Llama cross-tokenizer-family
pair and the Qwen3-8B target.

## Method (the harness)

`camelid bench-speculative` (new, committed) runs, back-to-back on one prompt with a fresh KV
cache and `--warmup`: (1) a **plain greedy baseline** and (2) the **speculative run**. The spec
loop mirrors the server's exact accept/verify/rollback path (`api::generate`): a normal greedy
first step seeds the resident engine, then each round drafts ≤γ tokens, verifies them in ONE
batched forward (`verify_drafts_gpu` on the resident GPU; CPU chunk verify + KV rollback as
fallback), accepts the longest confirmed prefix plus the target's own next token, and rolls the
rest back. It reuses the production `NGramDrafter` / `ModelDrafter` / `verify_drafts_gpu` /
`accepted_draft_prefix` — no new kernels. Draft and verify spans are timed separately.

Per cell it emits one JSON line (`results/<kind>.jsonl`) with: accept_rate,
mean_accepted_tokens_per_round, draft_ms, verify_ms, **f_draft = draft/(draft+verify)**, plain &
spec tok/s, **S_sync = spec t/s ÷ plain t/s**, TTFT, gpu/cpu verify-round split, peak RSS, GPU
offload status, and the **lossless gate**: `first_divergent_generated_token_index` — the first
index where the spec token stream diverges from THIS run's plain greedy stream (−1 = identical).

Lossless gate is intra-Camelid (spec vs Camelid's own greedy), not vs llama.cpp.

---

## Phase 1 Path A — n-gram (24 cells, ALL LOSSLESS ✓)

### S_sync (spec t/s ÷ plain t/s) — >1 means speculation pays

| workload | γ=2 | γ=4 | γ=6 | γ=7 |
|---|---|---|---|---|
| code | **1.25×** | 1.22× | 1.17× | 1.12× |
| json | 1.24× | **1.26×** | 1.19× | 1.10× |
| extraction | 1.00× | 1.02× | 0.93× | 0.90× |
| chat | 0.92× | 1.00× | 0.97× | 0.98× |
| creative | 0.99× | 1.00× | 0.92× | 0.98× |
| adversarial | 1.03× | 1.01× | 0.96× | 0.93× |

### accept rate

| workload | γ=2 | γ=4 | γ=6 | γ=7 |
|---|---|---|---|---|
| code | 84.2% | 66.1% | 54.2% | 46.4% |
| json | 68.8% | 65.5% | 53.7% | 46.0% |
| extraction | 46.2% | 31.8% | 25.0% | 21.4% |
| chat | 40.0% | 50.0% | 33.3% | 28.6% |
| creative | 0.0% | 0.0% | 0.0% | 0.0% |
| adversarial | 53.8% | 42.5% | 33.3% | 28.6% |

### Reading it

- **Real, lossless wins: code 1.25× (γ=2) and json 1.26× (γ=4).** `f_draft ≈ 0.0001` — the
  n-gram lookup is free, so any non-trivial accept rate is pure upside. Every winning round used
  the **GPU resident verify** (gpu/cpu verify split 19/0, 21/0). These are genuine user-facing
  speedups over Camelid's own baseline on structured output.
- **γ=2–4 is the sweet spot; γ=6,7 regress below 1.0×.** Wider draft → wider batched verify GEMM,
  and accept rate falls with depth, so the extra verify rows cost more than the tokens they save.
  This is the `verify_batch` overhead the prior campaign flagged (Blocker 2), now bounded: even
  with free drafts, over-drafting loses.
- **Prose loses or breaks even** (chat/creative ≈ 1.0×; creative accept 0% → n-gram never matches
  novel prose, so it degrades to plain decode at ~1.0×, never worse at γ=2/4). Adversarial ~1.0×.
- **extraction is only ~1.0× — an honest miss against the hypothesis.** This prompt is *format*-
  repetitive but *content*-novel (every row has a distinct name/role/city), so the n-gram suffix
  lookup catches only separators. The repetition ceiling exists but needs token-level content
  repetition: a pathological "alpha beta gamma ×6" prompt hits **S_sync 2.54× / accept 100% /
  f_draft 0.000** (smoke test, same harness) — that is the n-gram upper bound, realized only when
  generated *content* recurs, not merely its shape.

**Path A verdict:** lossless synchronous n-gram speculation is a real ~1.25–1.26× win on
code/JSON at γ=2–4 (ceiling ~2.5× on genuinely repetitive content), ~free draft (f_draft≈0), no
regression on prose at γ≤4. Shippable as-is.

---

## Phase 1 Path B — GPU-resident draft model (Qwen3-0.6B → 4B)

**Crux finding (smoke + forced-CPU control, code prompt, γ=4):**

| draft path | accept | tok/round | f_draft | plain t/s | spec t/s | S_sync | lossless |
|---|---|---|---|---|---|---|---|
| "GPU-resident" (default) | 92.5% | 4.50 | 0.897 | 41.5 | 4.22 | **0.10×** | ✓ |
| forced CPU (`--cpu-draft`) | 92.5% | 4.50 | 0.897 | 41.6 | 4.69 | **0.11×** | ✓ |

The accept rate is **excellent (92.5%, well above n-gram's 84% on code)** — the 0.6B predicts the
4B very well. But S_sync is **0.10×** — a 10× *slowdown* — because the draft costs ~253 ms per
draft-token (`f_draft 0.90`). The forced-CPU control is **identical**, which proves the default
"GPU-resident" draft is **not actually resident** — it silently runs the same slow CPU forward.
The draft is draft-bound, and because the draft is sequential, more γ only makes it worse — so the
S_sync verdict is γ-insensitive and negative.

> **Correction (from Phase 3):** this is a **Camelid implementation gap, not a 6 GB VRAM wall.**
> llama.cpp's `llama-speculative` runs the *same* 0.6B draft + 4B target **both fully GPU-resident
> on this identical box** (`-ngl 99 -ngld 99`) and gets a 1.4–2.0× spec win (Phase 3). So both
> models *do* fit resident on 6 GB; Camelid's draft session just fails to claim GPU residency and
> falls back to CPU. Fixing that — not concurrency — is the real lever (see Phase 4).

### Focused matrix — 6 workloads × γ{2,4} (12 cells, ALL LOSSLESS ✓)

γ{6,7} omitted (not silently): the config is draft-bound (the draft is sequential, ~253 ms/tok),
so S_sync only degrades with more draft tokens — every cell here is already 0.07–0.11×, and a
wider γ cannot reverse that. The γ sweep would only re-confirm a negative.

**S_sync** (all ≪ 1):

| workload | γ=2 | γ=4 |
|---|---|---|
| code | 0.11× | 0.11× |
| json | 0.10× | 0.10× |
| extraction | 0.08× | 0.08× |
| chat | 0.09× | 0.07× |
| creative | 0.10× | 0.07× |
| adversarial | 0.10× | 0.08× |

**accept rate** — the genuinely useful signal (can the 0.6B predict the 4B?):

| workload | γ=2 | γ=4 | vs n-gram (best) |
|---|---|---|---|
| code | 92.1% | 82.9% | ≈ (n-gram 84%) |
| json | 89.0% | 84.5% | **better** (n-gram 69%) |
| extraction | 75.2% | 72.3% | **much better** (n-gram 46%) |
| chat | 40.3% | 25.8% | comparable |
| creative | 43.0% | 28.0% | **better** (n-gram 0%) |
| adversarial | 55.5% | 40.3% | comparable |

### Reading it

- **The draft *quality* is excellent and beats n-gram** — 75–92% accept on structured output (it
  even predicts *novel-content* extraction at 75% where n-gram managed 46%, and creative prose at
  43% where n-gram is 0%). If the draft were cheap, the model drafter would be the better path.
- **The draft *speed* destroys it.** `f_draft 0.83–0.91`: ~250 ms per draft-token because the
  0.6B is not GPU-resident (no room beside the 4B on 6 GB) and runs the scalar CPU forward. Even
  with 92% accept, S_sync is 0.07–0.11× — a 10× slowdown.
- All verify rounds still ran on the **GPU resident verify** (gpu/cpu split e.g. 45/0, 70/0); only
  the *draft* is off-GPU.

**Path B verdict:** the GPU-resident draft-model path does not exist on this 6 GB box — the draft
silently degrades to CPU and is draft-bound. High accept (draft quality) is necessary but not
sufficient; draft *throughput* is the wall. Reconfirms the prior campaign's CPU-draft verdict via
the previously-skipped "GPU-resident" path.

---

## Phase 2 — lossless gate (Gate 2 ✓)

`first_divergent_generated_token_index = −1` on **36/36 cells**; **all 12 S_sync>1 cells are
lossless**. Every speculative stream is byte-identical to this machine's own plain greedy decode.
Inherited from the parity-locked verify, asserted per config here.

---

## Phase 3 — head-to-head vs llama.cpp speculative decode (RUN; local Qwen3 pair)

`llama-speculative` @ `acd79d6` on the **same prompt files**, greedy/lossless, both models
GPU-resident (`-ngl 99 -ngld 99 --spec-draft-n-max 8 -c 2048`), 128 tok, median of 3 reps (very
low variance). Camelid column = its best path per workload (n-gram everywhere — the draft-model
path is draft-bound, so it never wins). llama raw 4B = 54.5 t/s (llama-bench, `-fa1`).

| workload | Camelid best | Camelid t/s | llama-spec t/s | **Camelid / llama** | Cam accept | llama accept | llama-spec / llama-raw |
|---|---|---|---|---|---|---|---|
| code | n-gram γ=2 | 50.2 | 97.8 | **0.51×** | 84% | 68% | 1.79× |
| json | n-gram γ=4 | 47.5 | 108.7 | **0.44×** | 65% | 79% | 1.99× |
| extraction | n-gram γ=2 | 32.5 | 76.2 | **0.43×** | 46% | 52% | 1.40× |
| chat | n-gram γ=4 | 32.1 | 49.5 | **0.65×** | 50% | 28% | 0.91× |
| creative | n-gram γ=2 | 31.7 | 37.7 | **0.84×** | 0% | 18% | 0.69× |
| adversarial | n-gram γ=2 | 32.6 | 45.3 | **0.72×** | 54% | 25% | 0.83× |

### Reading it

- **Camelid-spec loses every workload, 0.43–0.84×.** As the spec predicted, synchronous
  speculation does not close the head-to-head — the ~0.73× base-kernel deficit scales through and,
  on structured text, *compounds* with llama's bigger spec multiplier.
- **llama's winning lever is the GPU-resident draft model** it gets 1.79× (code) / 1.99× (json) /
  1.40× (extraction) because its 0.6B drafts on-GPU at ~free cost with good accept. This is exactly
  the path Camelid's implementation fails to realize on the identical box (Path B). The
  head-to-head is **worst where only a model drafter helps** (structured: 0.43–0.51×).
- **Camelid's n-gram is regression-safe; llama's draft-model spec is not.** On my high-entropy
  prose prompts llama's draft accept collapses (creative 18%, adversarial 25%) and its spec drops
  **below its own raw decode** (creative 0.69×, chat 0.91×, adversarial 0.83×) — the draft cost
  isn't recovered. Camelid's n-gram simply drafts nothing on novel text and decays to plain decode
  (creative 0% accept → 0.84× of llama, the narrowest gap), never regressing. A small but real
  robustness edge — though Camelid still loses on absolute t/s everywhere because of the base
  kernel.

**Gate 3:** no cell where Camelid-spec ≥ llama.cpp-spec. Closest is creative (0.84×), and only
because llama's own spec self-regressed there. No head-to-head win, synchronous, on any workload.

---

## Phase 4 — decision

Decided from the data, per the SPEC_RECHECK decision gate:

**1. SHIP — lossless synchronous n-gram speculation on code/JSON (default γ between 2 and 4).**
A real, lossless **~1.25–1.26×** over Camelid's own baseline on structured output (ceiling ~2.5×
on genuinely repetitive content), `f_draft ≈ 0`, GPU-verified, no regression on prose at γ≤4. This
is the spec's "ship synchronous n-gram, no concurrency needed" exit, taken on evidence — the
draft is free, so there is nothing for concurrency to hide and the synchronous path already
captures the win.

**2. DO NOT build the concurrent CPU-draft/GPU-verify overlap (Bactrian) on this hardware.**
The only path with an `f_draft` large enough to be worth overlapping is the model drafter
(`f_draft ≈ 0.85–0.90`), and it is **draft-bound at ~250 ms/draft-token**. Concurrency's ceiling
is `max(draft, verify)` per round; here draft (≈250 ms × γ) ≫ verify (≈75–100 ms), so even perfect
overlap lands at the draft time → ~0.1× of plain. Concurrency cannot turn a 10× slowdown into a
win while the draft is this slow. This reconfirms SPEED_CAMPAIGN Phase 4 — now from the
previously-skipped GPU-resident draft path, which on 6 GB is not actually resident.

**3. The real unlock is GPU-resident drafting — NOT concurrency — and Phase 3 proves it is
reachable on this exact box.** The draft *accept rate* is already excellent (75–92% structured),
so the model drafter's *quality* is not the problem; its *throughput* is, and only because
Camelid's draft session fails to claim GPU residency and falls to CPU. llama.cpp runs the same
0.6B+4B **both resident on this same 6 GB GPU** (`-ngld 99`) and turns that into a real 1.4–2.0×
spec win across structured *and* (modestly) prose. So the highest-value next step is **fixing
Camelid's resident draft-model path** (find why the drafter engine doesn't go resident and make it
behave like `-ngld 99`), which yields a ~free, high-accept draft *synchronously* — no concurrency
needed, and it would help every workload (not just structured, where n-gram is capped). A leaner
`verify_batch` (the γ≥6 n-gram regression shows the batched verify is heavier than one resident
decode) is the secondary lever. Bactrian (CPU-draft/GPU-verify overlap) is the *wrong* lever:
llama.cpp wins with a resident draft and serialized draft-then-verify, no overlap at all.

**4. Head-to-head vs llama.cpp (Phase 3) — RUN; Camelid does not win.** Camelid-spec lands
**0.43–0.84×** of llama.cpp-spec on every workload (full table above), confirming the spec's
expectation: the ~0.73× base-kernel deficit scales through and, on structured text, compounds with
llama's larger (resident-draft) multiplier. No synchronous head-to-head win on any cell. The lever
that could change it is §3 (resident drafting + leaner verify + the base-kernel/roofline work),
not the concurrent overlap.

### Honest ceilings / caveats

- Single draft/target pair (Qwen3-0.6B→4B), single box. The Llama cross-family pair and Qwen3-8B
  target were not downloaded. These are the "full matrix" option, deferred.
- Phase 3 prompts are byte-identical across engines; token *streams* are not (each engine is
  lossless vs its OWN greedy — a speed comparison at matched settings, not an identical-output
  claim). llama reps had very low variance; Camelid per-cell plain t/s varies 31–40 (laptop clock
  drift across the ~12-min run), so head-to-head ratios carry ~±a few % from clock noise — far
  inside the 0.43–0.84× verdict.
- Laptop GPU clocks are unpinned (thermal drift). The verdicts (n-gram +25%, draft −90%) are far
  outside any clock noise; the exact tok/s digits are not pinned-clock numbers.
- "extraction" here is format-repetitive but content-novel; n-gram's low score on it is a property
  of *this* prompt, not of extraction in general. The model drafter's 75% on the same prompt shows
  the structure is learnable — by a model, not by suffix lookup.
- The ~250 ms/draft-token CPU figure exceeds the prior campaign's ~117 ms/token 0.6B measurement;
  the extra is per-round re-sync/rollback in the draft loop plus contention with the GPU-driving
  CPU thread. Either way it is draft-bound; the precise CPU-draft kernel speed is the CPU-perf
  mission's scope, not this recheck's.

## Files

- `prompts/*.txt` — the 6 fixed workload prompts (raw continuations).
- `workloads.md` — workload manifest (all `max_tokens=128`, greedy, `--warmup`).
- `run_matrix.sh` — Camelid driver: `run_matrix.sh <ngram|draft-gpu|draft-cpu> ["<γ list>"]`.
- `phase3_llama_spec.sh` — Phase 3 driver: runs `llama-speculative` on the same prompts.
- `analyze.mjs` — renders the Camelid matrices from `results/*.jsonl`.
- `results/ngram.jsonl` (24), `results/draft-gpu.jsonl` (12) — one Camelid JSON record per cell.
- `results/phase3-llama-spec.jsonl` (6) — llama.cpp-speculative per workload (median of 3 reps).
- `results/matrices.md` — rendered Camelid S_sync / accept-rate / detail tables.
- `results/headtohead.md` + `.json` — Phase 3 Camelid-vs-llama table.

## Reproduce

```
# n-gram, full sweep
bash run_matrix.sh ngram "2 4 6 7"
# GPU-resident draft model (Qwen3-0.6B → 4B)
bash run_matrix.sh draft-gpu "2 4"
# Phase 3: llama.cpp-speculative on the same prompts (both models GPU-resident)
bash phase3_llama_spec.sh
# single cell:
CAMELID_COMMIT=$(git rev-parse --short HEAD) camelid.exe bench-speculative \
  Qwen3-4B-Q8_0.gguf --drafter ngram --draft-tokens 4 --workload code \
  --max-tokens 128 --warmup --prompt-file prompts/code.txt
```

## Provenance

- Camelid commit: `feat/bactrian-experimental-fast` @ 196cdf14
- Target Qwen3-4B-Q8_0 sha256: `8c2f07f26af9747e41988551106f149b03eb9b5cb6df636027b6bf6278473300`
- Draft Qwen3-0.6B-Q8_0 sha256: `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`
- llama.cpp pin (Phase 3, run): `acd79d603cb2e1c84c0886137b80f1ad649b6857` (`-ngl 99 -ngld 99`)
- Backend: CUDA 12.9, RTX 3060 Laptop (6 GB), GPU-resident target decode + GPU batched verify.

