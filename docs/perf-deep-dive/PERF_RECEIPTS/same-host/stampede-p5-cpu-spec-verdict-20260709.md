# STAMPEDE Phase 5 — CPU spec-tree verify: Stage A verdict (infrastructure GO, economics blocked)

Host: i7-11800H CPU-only lane (CUDA hidden), 3B Q8_0, camelid 85f18f24 (Phase 5 Stage A on
main 8f197347), bench-speculative --warmup, CAMELID_SPEC_TREE=1, suffix drafter, greedy.
Single-engine by construction (one process, one model); free-RAM 7.5-7.7GB, no orphans.

## What landed (P5.1 + P5.2)
The spec-tree lane now verifies on the CPU when the resident GPU verify is unavailable:
tree's primary chain -> batched chunk forward -> KV rollback (the linear lane's shipped
pattern), one-way ratchet off the resident paths, drafter switches to linear chains,
kill-switch CAMELID_SPEC_CPU_VERIFY=0. The acceptance latch is extracted to
speculative::SpecLatch (constants 2/4/2/1/64) with 5 unit tests; GPU and CPU rounds drive
one policy. LOSSLESS: first_divergent == -1 on every cell below (spec text == plain greedy).

## Measured (k=7 drafts, verify batch 8 rows)

| workload | s_sync | verify rounds (CPU) | acc drafts/round | lossless |
|---|---|---|---|---|
| repetitive_extraction (256 tok, x3 reps) | 1.095 / 1.099 / 1.061 | 33 | 5.45 (64% rate) | yes x3 |
| code_completion (128 tok) | 0.821 | 6 | 1.67 | yes |
| structured_json (128 tok) | 0.902 | 5 | 1.40 | yes |
| normal_chat (128 tok) | 0.937 | 4 | 1.50 | yes |
| adversarial_lowaccept (128 tok) | 0.929 | 3 | 1.00 | yes |

k sweep (repetitive, 128 tok, single runs, thermal-noisy): k=3 1.11x / k=4 0.93x /
k=6 0.86x / k=7 0.99x.

## The economics blocker (measured, reproducible x3)
verify_ms_per_round / plain_ms_per_token = **7.05 / 7.03 / 7.28** — one 8-row chunk verify
costs ~7.1 plain decode steps. The campaign brief's premise was ~one weight pass (~2.5-3
decode steps, from the large-M prefill amortization receipts). At 7.1x, even PERFECT
acceptance (8 tokens/round) caps at 8/7.1 = 1.13x; the measured-excellent 5.45+1 = 6.45
tokens/round yields ~0.91x on verify rounds. THE DRAFTER SIDE IS PROVEN (5.45 accepted
drafts/round on repetitive is outstanding); the blocker is purely the small-M (4..8 rows)
chunk-forward cost — per-row cost at M=8 is ~117ms vs 133ms plain decode, i.e. batching
amortizes almost nothing at small M on this path (the prefill receipts' ~2.5x amortization
was measured at M~512).

## Gate verdict
Per the Phase 5 gate (any workload >= 1.3x -> default-on; all < 1.1x -> KILL):
**economics KILLED at current verify cost; the lane ships DEFAULT-OFF** (CAMELID_SPEC_TREE
is opt-in; the latch bounds the loss on latched-off classes to the O(1) probe cost, which
a 128-token run overstates - code's 0.82x is 4 probe verifies at 7x on a short run).

## The staged follow-up this receipt de-risks
Profile and fix the M in [4,8] chunk verify (candidates: per-call activation-quantize and
scratch overheads repeated per projection, owner GEMM behavior at M=8 far below its
ROW_BLOCK=64 design point, attention over full history per row). At the premise cost
(~3x), repetitive = (5.45+1)/3 ~= 2.1x and the gate flips to GO with the latch already in
place. That investigation is its own kernel slice; the acceptance histogram above is the
receipt that it is worth doing.

---

## UPDATE 2026-07-09 — small-M chunk-verify fixes (P5 follow-up slice)

Component profiling (env-gated CAMELID_SPEC_VERIFY_TIMINGS split of the chunk forward) found
two culprits and fixed both:
1. **The bespoke gate/up prefill arms have no small-M amortization**: at 8 rows they ran at
   ~6x a single weight pass (gate+up ~330-376k us over 28 layers) while ffn_down — which
   flows through the cascade to the tiled Q8 MATMUL OWNER — amortized fine (~63k us). FIX:
   the gate/up arms now decline 2..=16-row batches so they flow to the owner (large chunks
   unchanged; predicate boundary pinned by test; both routes carry bitwise proofs vs the
   same packed-rows4 baseline, so the swap cannot change bits).
2. **The verify chunk never ran on the wider prefill pool** (every other batched prefill
   path does, P0.6). FIX: the layer loop now runs under run_on_prefill_pool (pool choice
   never changes math — the established contract).

Re-measured (same protocol):

| cell | before | after |
|---|---|---|
| repetitive s_sync (256 tok) | 1.06-1.10x | **1.265x** (spec 11.04 tok/s > plain 8.72) |
| code / json / chat / adversarial | 0.82-0.94x | 0.905-0.952x |
| gate+up per verify (28 layers) | ~330-376k us | ~66k us |
| logits | ~150k us | ~62k us |
| lossless | yes x5 | yes x5 |

Large-chunk prefill sanity (bench-owner-sweep, 516-token prompt): off 27.3-27.4 /
owner+vnni4x8 30.4-30.6 tok/s — matches the Lane A revalidation receipts within thermal
noise; the routing change is invisible outside 2..=16-row batches.

Remaining gate distance: repetitive 1.265x vs the 1.3x default-on bar. The dominant
remaining cost is attention context (actx ~265-276k us of ~440k layer time at 8 rows,
position ~700) — inherently per-row KV work; batching/parallelizing small-M chunk attention
is the next (and last cheap) lever. The lane stays opt-in.

Post-review-fix confirmation (predicate 4..=16 + owner-would-take guards): repetitive 256-tok
cell s_sync = 1.309x (spec 10.07 / plain 7.69, lossless, 5.45 acc/round) — cells straddle
1.26-1.31x across thermal conditions, i.e. AT the 1.3x gate line. Default-on remains
not earned (latched-off classes 0.905-0.952 on 128-token runs = the O(1) probe cost);
the lane stays opt-in.
