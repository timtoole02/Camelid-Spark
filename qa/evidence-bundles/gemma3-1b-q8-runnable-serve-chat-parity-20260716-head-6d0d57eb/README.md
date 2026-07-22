# Gemma 3 1B-It Q8_0 — runnable serve chat parity (MUSTER M-A1)

Row: `gemma_3_1b_it_q8_0` (filename-anchored) — exact file `gemma-3-1b-it-Q8_0.gguf`,
sha256 `b205840c5dcef55078e37d344677869a714ffd42a4ae448c48dcfb52e4bb10d5`, 1,069,306,368 B,
upstream-verified exact against ggml-org/gemma-3-1b-it-GGUF (license gemma). First gemma3
support surface: runnable serve lane (`CAMELID_RUNNABLE_SERVE=1`, CPU), gemma3 marker
renderer, EOG-stopping dense decode.

## Phase 3 gates (before any parity run)
- Renderer byte-locked: `qa/prompt-packs/gemma3-chat-template-shapes-v1.json` — 6 shapes
  captured from the pinned oracle's `/apply-template` (--jinja, solo session), locked by an
  in-src test; PASSED FIRST RUN. Covers the trim and untrimmed-system-prefix distinguishers.
- Gate pack committed: `qa/prompt-packs/gemma3-chat-gate-pack-v1.json` (5 plain-English
  prompts avoiding the SPM merge-order divergence classes; all sequences far under the
  512-token sliding window).
- Determinism sanity: two independent serve sessions byte-identical (det-run1/2.json).

## Phase 4 parity (two-phase, engines never co-resident)
Oracle: llama.cpp llama-server version 9632 (acd79d603), CPU backend, binary sha256
6c787bf07ac1d7e1bbaa1ee176c3ef0df58ea86494c8c1b1d2d9f4a9176b19ae, flags
`-ngl 0 -ctk f32 -ctv f32 -fa off --no-repack -c 2048`, greedy /completion, captured solo
(gemma3-oracle.json), then camelid compared via scripts/chat-parity-gemma3.mjs.

Results (gemma3-parity.json, committed AS-IS with all_pass=false):
- **Cross-engine prompt tokenization identical 5/5** (llama /tokenize vs camelid encode,
  BOS at token level on both) — the recon's SPM merge-order risk did NOT fire.
- **4/5 prompts token-AND-text identical at every 1/5/50 depth**, including a natural
  early-stop leg that exercises the new dense EOG stop semantics.
- **One divergence**: prompt 4 ("sky color"), position 16 of the 50-token leg only
  (1/5-depth legs identical). Probed in solo oracle sessions (near-tie-analysis.json +
  probe-*.json): camelid's token is the oracle's immediate #2 at a **0.3416-nat** top-2
  gap; the oracle is BIT-STABLE across thread-count (default/4/2) and runtime-repack
  kernel controls, so no oracle-side flip exists.

## Near-tie disclosure (read before citing this bundle)
The single flip's gap (0.3416 nat) EXCEEDS the Ornith Q4_K_M precedent's <=0.33-nat
soft-position line, and no oracle-backend flip was found. The attribution rests on the
conductor's second named precedent (Llama-3.2-3B Q4_K_M: "documented benign near-ties
with logprob gaps stated" — that accepted bundle carried one measured 0.18-nat gap, two
unmeasured flips, and no controls). This is the weakest single attribution in the MUSTER
campaign to date, stated in the contract row, COMPATIBILITY.md, and the promotion PR.

## Phase 5 smoke
- Full model-promotion-smoke-bundle attempt (api-webui/): every API step green
  (load, capabilities, /v1/completions, /v1/chat/completions) — the run then failed at
  the generation-timings HELPER because the runnable lane emits no camelid.timings_ms
  (pre-existing lane gap; the Ornith precedent rows never ran this bundle). The failure
  record is committed as-is.
- The §7.1 expectation leg (frontend/scripts/smoke.mjs) was then run directly
  (frontend-smoke.txt): **exit 0, all four expectations asserted** —
  --expect-compatibility-row gemma_3_1b_it_q8_0, --expect-compatibility-status
  supported_exact_row_smoke, --expect-contract-supported true, --expect-webui-chat
  enabled — plus real streaming + non-streaming chat round-trips. Serve ran with
  CAMELID_SKIP_FIT_CHECK=1 (advisory-only fit check double-counts an already-resident
  model; same note as the M-B1 bundle).
- Observation recorded for follow-up (outside this row's claimed surface):
  /v1/completions still routes runnable-served gemma3 to the mis-bound optimized engine
  (no runnable bridge and no fail-closed gate on that endpoint).

## Perf reality (recorded, NOT a claim)
~5 s/token on the pure-f32 runnable CPU lane (the 262k-vocab tied head dominates).

## Environment
Head 6d0d57eb tree (branch muster/ma1-gemma3-1b). Windows 11 build 26220; 15.7 GiB RAM.
CPU lane only (runnable CUDA is qwen35-gated).
