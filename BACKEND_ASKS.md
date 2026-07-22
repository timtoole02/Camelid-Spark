# BACKEND_ASKS.md

Open requests for reference data / undecided tolerances surfaced while building the
runnable lane (`RUNNABLE_LANE_SPEC.md`). Each entry: what's needed, why, blocking phase.

## RA-1 — HF reference harness (transformers + tokenizers) — **RESOLVED (Phases 3 & 5)**
- **What:** A pinned HF `transformers` + HF `tokenizers` reference harness producing
  frozen fixtures (greedy logits/token sequences; string↔id maps) per (arch, quant, tokenizer).
- **Resolution:** Anchored to **HF** (spec-literal). Both halves now stood up:
  - Tokenizer (Phase 3): `scripts/gen-tokenizer-fixtures.py` (HF `tokenizers`,
    `tokenizers==0.23.1`) → `tests/fixtures/tokenizer_hf/`; `tests/runnable_tokenizer.rs`.
  - Transformers (Phase 5): `scripts/gen-hf-parity-fixtures.py` loads the **same GGUF**
    into HF (`transformers==5.12.1`, `torch==2.12.0+cpu`) — it dequantizes Q8_0 to f32
    (= camelid's bit-exact dequant) and un-permutes Q/K — runs greedy, records token
    sequences + first-step logits → `tests/fixtures/hf_parity/tinyllama.json`;
    `tests/runnable_parity.rs` checks camelid. **Note:** transformers 5.12.1 mis-detects
    gguf's version as 'N/A'; the script monkeypatches `modeling_gguf_pytorch_utils.is_gguf_available`.

## RA-5 — SPM leading-whitespace divergence from HF — **RESOLVED (Phase 3, fixed)**
- **Was:** camelid's SPM `normalize_spm_text` suppressed the dummy `▁` prefix when text
  started with whitespace; HF's Metaspace always prepends it → 4 leading/pure-whitespace
  cases diverged.
- **Fix:** removed the `!text.starts_with(char::is_whitespace)` guard
  (src/tokenizer/mod.rs:575) so the dummy `▁` is always prepended when `add_space_prefix`
  is set (matching HF). SPM encode is now **30/30 HF-exact**; BPE remains 30/30.
- **Regression check:** no supported-lane regression — lib unit tests 475/475, existing
  `tests/tokenizer.rs` 25/25, `tests/dg_tokenizer_parity.rs` (llama.cpp anchor) green.
  The change only affects plain-text SPM encode with `parse_special=false` on
  leading-whitespace input; chat tokenization uses `parse_special=true` + control tokens
  and is unaffected. DG/gemma sets `add_space_prefix=0`, so it short-circuits.
- **Decode note (deliberate, not a defect):** camelid's `decode` is STATELESS so it can be
  called per-token during streaming (`api/mod.rs:6880/6901`, `main.rs:1659/2631`). It
  therefore retains the single dummy-prefix space rather than stripping it like HF's
  stateful Metaspace decoder. Consumers strip one leading space per the `add_space_prefix`
  convention to recover exact round-trip — `tests/runnable_tokenizer.rs` does this and
  shows 0/30 round-trip instability for both families.

## RA-2 — ggml dequant reference fixtures — **RESOLVED (Phase 2)**
- **What:** Checked-in block-level reference dumps under `tests/fixtures/dequant/` produced
  by Python `gguf`/`llama-cpp`, for F32, F16, Q8_0, Q4_0, Q4_K_M, Q5_K_M, Q6_K.
- **Resolution:** `scripts/gen-dequant-fixtures.py` emits fixtures under
  `tests/fixtures/dequant/` using the `gguf` package (gguf==0.19.0, the numpy port of
  ggml's dequant) as the reference; `tests/runnable_dequant.rs` bit-checks the runnable
  decoder against them. All 7 formats are **bit-exact** (max_abs=0, max_ulp=0).
  - F32/F16 via numpy; Q8_0/Q4_0 via `gguf.quants.quantize`; Q4_K/Q5_K/Q6_K via
    **synthetic structurally-valid blocks** (random integer fields + sanitized f16
    super-scales). A dequant bit-exactness test is independent of byte provenance.
  - **Why synthetic for K-quants:** the only on-disk K-quant model
    (`diffusiongemma-…-Q4_K_M.gguf`, ~16 GB) cannot be memmapped on this box —
    `GGUFReader` does a full-file `np.memmap` and Windows fails with
    `WinError 1455 (paging file too small)`. Real-model extraction was abandoned for
    the (equivalent, more robust) synthetic path. If real-model anchoring is later
    wanted, it needs a small K-quant GGUF or a streaming (non-memmap) reader.

## RA-3 — tolerances
- **(i) Dequant tolerance — RESOLVED (Phase 2):** every covered format (incl. F16 and all
  K-quants) is **bit-exact** vs ggml reference — `max_abs_diff == 0`, `max_ulp == 0`. No
  tolerance needed; the test asserts bit-exactness for F32/F16/Q8_0/Q4_0 and `max_abs == 0`
  for the K-quants.
- **(ii) Logit max-abs-diff threshold — RESOLVED (Phase 5):** the hard gate is greedy
  token-sequence exact-match (passed 64/64 tokens, 4 prompts, TinyLlama). Observed logit
  max-abs-diff vs HF (same dequantized weights) = **4.673e-5** — pure f32 op-order
  rounding. No numeric tolerance is gated; the diff is reported as evidence in the parity
  artifact (`qa/runnable/tinyllama-parity.json`).

## RA-4 — covered-set vs. code allowlist mismatch — **RESOLVED (Phase 1)**
- **What:** Confirm the runnable v1 covered-set is authoritative:
  spec archs `{llama, qwen2, qwen3, gemma2, gemma3, phi3}` vs. `src/model.rs:52-54`
  `{llama, mistral, qwen2, qwen3, smollm3, gemma3, gemma4, phi3, lfm2}` (note: spec has
  **gemma2**, code does not; code has mistral/smollm3/gemma4/lfm2, spec does not).
- **Resolution:** The **spec's covered-set is authoritative for the runnable lane**
  (the spec declares it so). The admission gate (`src/runnable/admit.rs`,
  `COVERED_ARCHITECTURES`) keys off `{llama, qwen2, qwen3, gemma2, gemma3, phi3}`,
  intentionally independent of `model.rs`'s optimized-lane allowlist. Revisit only if
  a model the supported lane handles (e.g. mistral) must also run runnable.
