# ORNITH_9B_CONSTRAINED_VRAM_CONDUCTOR — reconciliation against merged state

Date: 2026-07-02. Author: conductor execution session.
Purpose: the conductor was drafted against a stale picture of the tree. Before
executing, each item is reconciled against what is already merged on `main`
(PR #350 bringup + the GPU-resident lane, commits `b1ab4753…82caaaa6`). Per ground
rules ("evidence gates everything", "read metadata, don't trust this doc's numbers"),
stale premises are corrected here with receipts, not silently routed around.

## Refuted conductor premises (evidence against)

1. **"The legacy parity pin `acd79d6` predates `qwen35` and CANNOT load this model."**
   FALSE. `acd79d6` (build 9632) defines `LLM_ARCH_QWEN35` at the binary level and is
   the oracle every existing qwen35 receipt was minted against
   (`qa/ornith/G-PARITY-qwen35-vs-llamacpp.md`, PASS 4/4). Consequence: `REF_QWEN35`
   is pinned to `acd79d6`'s CUDA build — see `REFERENCE_PIN_QWEN35.md`. The dual-pin
   rule is preserved trivially (same commit, two build configs, receipts name their
   binary).

2. **"Q4_K_M (5.63GB) will not achieve full residency; do not burn time trying."**
   ALREADY DISPROVEN ON MAIN. The home-requant Q4_K_M (5,629,108,416 B) runs fully
   GPU-resident on this 6GB card: 5259 MiB peak @ 8192 ctx with sparse KV
   (only the 8 full-attention layers hold KV; the 24 DeltaNet layers hold fixed-size
   state), ~700 MiB margin, 12.06 tok/s decode, greedy-token-identical to the CPU
   oracle lane (commits `a8dacf5c`, `82caaaa6`). Consequences:
   - Item 4's kill criterion ("no ≤Q3-class quant achieves full residency…") has a
     live fallback that is *already better* than the kill path assumed.
   - The Item 4 bar that still matters: **16K context + ≥512 MiB headroom**, and the
     Item 5/6 premise (enough headroom to host spec-lane scratch + draft-state ring
     alongside weights). Q4_K_M at 16K sparse-KV KV cost (~8 layers × 4 kv-heads ×
     256 dim × 2 × f16 = 4 MiB/1K tokens ⇒ ~64 MiB @16K… measured, not assumed, in
     the Item 4 receipt) may well pass the bar itself — it becomes the row to beat,
     not a discard.
   - llama.cpp CUDA at `-ngl 99` decodes this Q4_K_M at ~40 tok/s vs our 12 tok/s
     (Item 0 smoke) — confirming the GPU lane's known "not bandwidth-bound, real
     headroom" diagnosis. Draft-speed work in Item 6 has kernel headroom to exploit.

## Item-by-item status vs. merged receipts

| Conductor item | Status | Evidence / gap |
|---|---|---|
| Item 0 — pin + fixtures | **THIS SESSION** | `REFERENCE_PIN_QWEN35.md`; Q6_K downloaded + hash-verified vs HF LFS oid; bf16 downloading; corpora frozen (`FIXTURES_five_prompt_parity.json`, `FIXTURES_agentic_20.json`, `FIXTURES_tokenizer_adversarial.json`); oracle CUDA smoke captured (think-block opening, 40.3 tok/s gen) |
| Item 1 — tokenizer gate | **GENUINELY OPEN** | Existing parity fed identical token-ID arrays to both engines (by design — isolates model forward), so an independent byte-exact text→ID gate was never minted. Known risk: `\p{M}` mark folding deferred in the qwen35 pre-tokenizer (G-PARITY "known deviation"). Harness: `camelid tokenize` subcommand (added this session) + `tokenizer_gate.mjs` |
| Item 2 — qwen35 bringup | **SUBSTANTIALLY EARNED** | CPU: G-PARITY PASS 4/4 greedy token-identical vs `acd79d6` (n=20). CUDA: GPU stack token-identical to certified CPU lane incl. run-twice + long-context (8192/2100-tok prefill) tests, merged on main. Gaps to mint `RECEIPT_ITEM2_*`: extend to 5 prompts × n=64; one direct Camelid-CUDA vs llama.cpp-CUDA five-prompt run |
| Item 3 — serving surface | **SUBSTANTIALLY EARNED** | Native renderer + `<think>`→`reasoning_content` split + qwen3_xml→`tool_calls` lift certified (G-TOOLCALL); 3 distinct-tool agent-eval PASS receipts (G-AGENT, promotion-eligible; battery was single-case-packaged only because the CPU lane was ~1 s/tok). Gaps: full 4/4 battery in one run — now feasible against the 12 tok/s GPU serve; archive one full streamed SSE capture (reasoning_content deltas + include_usage) |
| Item 4 — 6GB budget + quants | **OPEN (this conductor's first real build item)** | imatrix from 20-trace coding corpus over bf16 → IQ3_XXS / Q3_K_M / IQ4_XS + PPL table. NOTE the conductor underspecifies: Camelid's resident engine implements q8_0/q4_K/q6_K gemv only — IQ3_XXS (codebook grid) and IQ4_XS are new kernel families; Q3_K_M also contains q3_K (+q5_K) tensors. Plan: produce all three quants + PPL table first (pure llama.cpp), then implement the ONE winning quant's kernels in Camelid. Existing Q4_K_M full-residency row is the baseline to beat |
| Item 5 — economics gate | OPEN | blocked on Item 4 quant choice; Q6_K weights already local + hash-verified |
| Item 6 — Inverted Bactrian | OPEN | blocked on Item 5 GO |
| Item 7 — TDGP gating | OPEN | blocked on Item 6 |

## Corrections to Item 5/6 assumptions worth carrying

- **CPU verifier kernel reality:** the runnable lane's fast CPU path is the int8×int8
  AVX2 Q8_0 kernel (~0.3 s/tok decode, memory-bandwidth-bound). There is NO fast CPU
  K-quant path in the runnable lane (Q4_K_M falls to generic dequant, 0.64 tok/s-class).
  Q6_K-on-CPU batched verify (prefill-shaped) therefore needs either (a) a q6_K
  AVX2 dot in the runnable lane, or (b) verify via the Q8_0 weights instead of Q6_K
  (9.5GB in 15.7GB RAM — tight but mmap'd; Q8_0 is *higher* fidelity than Q6_K and
  already has the fast kernel). Item 5's harness should measure BOTH verifier options;
  Q8_0-as-verifier may dominate Q6_K on both quality and speed. The conductor's Q6_K
  choice was driven by the published-quant list, which the bf16 download obsoletes.
- **DeltaNet state rollback (Item 6):** the GPU lane resets SSM state per generate
  (`reset_qwen35_state`); the k-deep state ring buffer must cover the 24 DeltaNet
  layers' [32×128×128] f32 states (~6 MiB/position-checkpoint × k — VRAM budget line
  item for Item 4's headroom bar).
