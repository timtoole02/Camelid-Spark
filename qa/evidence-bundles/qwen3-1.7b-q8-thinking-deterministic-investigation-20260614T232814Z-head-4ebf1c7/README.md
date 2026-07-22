> **CORRECTION (artifact-free re-measurement).** The decode-vs-prefill mechanism described below (the "dominant, 3/4 probes" attribution) was a **measurement artifact**: the prefill comparison was fed a detokenized prefix with `parse_special=false`, mis-tokenizing the `<think>`/`<|im_start|>` special tokens. Re-measured with **exact token ids** (`camelid_prompt_token_ids`), **camelid decode == camelid prefill** (capital@73 gap **0.003**, not 0.7; 3/3 captured probes agree). The thinking-trace divergence vs llama.cpp is **entirely cross-implementation rounding**, not decode-vs-prefill. The bundle's **verdict still stands** (full-trace parity is not achievable via a shared-order config) and is strengthened. See `correction-decode-vs-prefill-artifact.json`.

---

# Qwen3-1.7B Q8_0 — thinking-mode deterministic full-trace parity: investigation (negative result)

**Question.** Can Qwen3 thinking-mode *full-trace* token-parity vs the pinned
llama.cpp reference be achieved by putting camelid and the reference on a shared
deterministic reduction order — promoting the shipped *leading-trace* claim?

**Verdict: No — not as a configuration.** The block-level reduction order is
already shared and stable on both sides; the residual divergence is fixed
per-kernel f32 arithmetic, which is not a reduction-order knob. The conservative
leading-trace claim is retained. No semantic/text-level (path-b) fallback was used.

## What was measured (`findings.json`)

1. **Within-engine reduction-order noise is ~0.** The ggml reference's top-2 logit
   gaps at the divergence positions are **bit-identical under `-t 1` and `-t 8`**
   (ggml parallelizes over output rows without splitting a reduction). camelid
   `--deterministic` is bit-exact across runs/threads by design (DECISIONS.md D9).
   So the divergence is **not** explained by within-engine reduction-order noise.

2. **`--deterministic` does not move the divergence.** camelid `--deterministic`
   reproduces the *exact* same first-divergence tokens (73 / 27 / 205 / 26) as the
   cpu_reference path used in the shipped leading-trace bundle. The cpu_reference
   path was already on the order-stable CPU kernels; pinning determinism changes
   nothing cross-engine.

3. **The divergences are small near-ties** (reference top-2 gap 0.0218–0.1796
   logit) flipped by **accumulated cross-kernel f32 differences far larger than
   reduction-order noise** — e.g. at "capital of France" token 73, camelid prefill
   ranks " landmarks" over " sites" by **+0.7064** logit while the reference ranks
   it by **+0.1656** (a ~0.54 cross-implementation difference), and camelid's
   *decode* flips to " sites".

4. **Two mechanisms (teacher-forced at each divergence):**
   - **Dominant — camelid decode-vs-prefill (3/4 probes).** Feeding the agreed
     prefix, camelid **prefill reproduces the reference argmax**, but camelid
     incremental **decode flips it**. camelid's forward pass (prefill) is correct;
     its autoregressive decode path (GEMV vs GEMM + step-wise KV accumulation)
     diverges. This is camelid-internal.
   - **Cross-implementation (1/4 — primary-color@27, gap 0.0218).** camelid prefill
     *also* disagrees with the reference: fixed camelid-CPU-dot vs ggml-`vec_dot`
     rounding across 28 layers.

## Why a shared-order config can't deliver full-trace parity

The shared block-level order (per-output serial K, ascending Q8_0 blocks,
`--no-repack`) is already in place. The residual is (a) camelid decode kernels ≠
camelid prefill kernels and (b) camelid CPU dot (i8mm/dotprod/vDSP) ≠ ggml
`vec_dot` — fixed, ISA/implementation-dependent f32 arithmetic (DECISIONS.md D9
explicitly notes the values are ISA-dependent). Eliminating these is
**reimplementation** (make camelid decode bit-match its prefill, and camelid
kernels bit-match ggml), not a configuration. And even if decode were unified with
prefill, primary-color@27 (a 0.0218-logit cross-implementation razor-tie) would
still flip — so full-trace parity is unreachable by config regardless.

## Actionable lead (the real discrepancy the kill-condition points at)

The dominant divergence is **camelid decode-vs-prefill numerical inconsistency**
(~0.7 logit by token 73, accumulated over decode steps; argmax-consistent for
tokens 0..72, then flips). The reference does not exhibit this decode-vs-prefill
flip here. Aligning camelid's decode numerics to its prefill path would shrink
long-greedy-generation divergence broadly (not only thinking mode). That is an
engineering change, out of scope for a reduction-order configuration.

## Outcome

Leading-trace thinking-mode claim unchanged (`qwen3-1.7b-q8-thinking-enabled-parity-*`).
No promotion to full-trace parity. No path-(b) semantic agreement.
