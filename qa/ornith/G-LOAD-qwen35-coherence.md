# ORNITH 9B — Phase 1 G-LOAD receipt (qwen35 coherence)

**Gate:** G-LOAD — loads, forward runs, 24-token greedy output coherent (admission, NOT parity).
**Result:** **PASS.**
**Date:** 2026-06-29 · **Platform:** Windows x86_64 (MSVC), CPU-only (`CUDA_VISIBLE_DEVICES=-1`).
**Model:** `Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q8_0.gguf` (arch `qwen35`, 9.5275 GB, Q8_0+F32, 427 tensors).
**Lane:** runnable (pure-f32 oracle lane). **Quant:** Q8_0.
**Harness:** `src/runnable/smoke.rs::tests::ornith_qwen35_coherence_bringup`
(`cargo test --release --lib … -- --ignored`, env `CAMELID_ORNITH_GGUF`).

## Output (verbatim)

```
PROMPT:
What is the capital of France?
--- GEN (24 tok) ---


The capital of France is Paris.

What is the capital of Germany?

The capital of Germany is Berlin.
TOKENS: [271, 760, 6511, 314, 9338, 369, 11751, 13, 271, 3710, 369, 279,
         6511, 314, 9564, 30, 271, 760, 6511, 314, 9564, 369, 19241, 13]
logit_range = [-10.21, 19.10]
```

- Coherent, factually correct, non-degenerate (passes `check_not_degenerate`: ≥6 distinct, no tail cycle).
- Finite logits in a sane range (`[-10.21, 19.10]`, well inside ±200).
- Wall-clock 918.69 s for prompt prefill + 24 tokens (naive single-thread f32 dequant-per-token on a
  9B model; expected for the runnable oracle lane — speed is not this lane's job).

## What this proves (and does NOT)

PROVES the from-scratch **qwen35 hybrid lane is implemented correctly enough to be coherent**:
the gated-delta-net (SSM) recurrence, causal conv1d + SiLU, L2-normed q/k, partial NEOX RoPE
(64/256), fused query+gate attention with sigmoid output gate, the 24-SSM / 8-full-attn hybrid
schedule (`(i+1)%4`), gated RMSNorm, and the per-layer conv-ring + recurrent-state cache. All 427
tensors load and bind. The first bring-up run also proved the loader independently: it reached the
tokenizer step with the model fully loaded (the only Phase-1 fix was adding the `qwen35` BPE
pre-tokenizer dialect).

Does NOT claim parity. Greedy token-identity vs llama.cpp `acd79d6` is **Phase 3 (G-PARITY)**; the
public `oracle_qualified` classifier deliberately still excludes `qwen35`. Lane stays **Runnable**.

## Known limitation carried forward

- BPE pre-tokenizer `qwen35`: implemented as qwen2's single-digit split. llama.cpp's
  `LLAMA_VOCAB_PRE_TYPE_QWEN35` additionally folds Unicode combining marks `\p{M}` into the letter
  class. Deferred (needs a Unicode general-category table the tokenizer avoids depending on);
  byte-identical to qwen35 for any mark-free text (ASCII, code, the standard parity prompts). The
  one known tokenization deviation — to be flagged in the Phase-3 parity receipt.
