# BASALT D-B6 — BF16 admission (Option A) evidence receipt

**Decision:** DECISIONS.md D-B6 (SIGNED by Tim, 2026-07-17) — Option A. Add BF16 to the
runnable lane's covered quant set as an exact-decode type so legitimate mixed-type NVFP4
files (the `gemma-4-E4B-it-NVFP4-mm` pilot's single `per_layer_model_proj` BF16 tensor)
admit under the existing whole-file coverage model. NO support-status change (stays
`not_anchored`, NOT supported).

**Scope boundary held:** this changes ONLY runnable-lane admission (`src/runnable/admit.rs`
covered set + `src/runnable/dequant.rs` dispatch). The gemma4 wire lane `WireQuant`
(`src/gemma4_runtime.rs`) is untouched — it still refuses BF16 as a matmul wire format
(`wire_quant_new_admits_nvfp4_and_still_refuses_uncovered` unchanged, PASS). The D-B3
architecture carve-out (`admit.rs`, gemma4 iff NVFP4) is untouched.

## Engine change

- `src/runnable/admit.rs`: `GgufTensorType::BF16` added to `is_covered_quant`; module doc
  updated (the "requires a BF16-free file" statement retired); generic covered-set refusal
  message now lists BF16.
- `src/runnable/dequant.rs`: `GgufTensorType::BF16 => decode_bf16_tensor(...)` dispatch arm
  added before the `other =>` refusal; reuses `crate::tensor::decode_bf16_tensor` (exact,
  lossless `u32::from(u16) << 16` widening — bf16 is the top 16 bits of f32). No new numeric
  code.

## Fixture (rider 2 — M-B5 exit condition (a))

- `tests/fixtures/dequant/bf16_exact.json`: 24-value golden (LE wire bytes + reference f32
  bits), covering +/-0, +/-1/2, 0.5, subnormals, min-normal, +/-Inf, qNaN payloads, max
  finite, and arbitrary mantissas. Provenance states bf16->f32 is definitionally lossless and
  identical to the pin's `ggml_bf16_to_fp32`.
- Unit tests in `src/tensor/mod.rs::bf16_dequant_parity_tests`:
  `bf16_fixture_reference_is_the_exact_widening` (self-check: `ref == bf16 << 16`),
  `decode_bf16_tensor_matches_golden_bit_exact` (bit-exact on `to_bits`),
  `decode_bf16_tensor_wrong_length_fails_closed`.

## Test / matrix re-sign (rider 3 + 4)

- `gemma4_nvfp4_with_bf16_refuses_on_bf16` INVERTED → `gemma4_nvfp4_with_bf16_admits_fully_after_d_b6`
  (Windows leg: real pilot shape admits fully). Off-Windows twin
  `gemma4_nvfp4_with_bf16_refuses_off_windows_platform_gate` keeps the Amendment 3 §9 platform gate.
- SHA_E `ends_with("IQ4_XS")` byte-identical-message pin RETIRED; relocated as
  `ends_with("IQ4_XS, BF16")` in `rejects_unknown_quant_naming_tensor` (sanctioned covered-set
  widening, IQ4_XS precedent).
- `qa/invariant_lanes.json` L1 `I-carveout` companion test fn renamed to the admission-pin fn;
  cell note + column source updated. `invariant_matrix_binding` meta-tests PASS.

## Surfaces reversed (#475 clause → admits fully; verbatim D-B6 text)

README.md (2 rows), SUPPORT_MATRIX_v0.1.md (row detail + new covered-set note),
docs/architecture/NVFP4_FORMAT.md (overview), src/api/mod.rs (NVFP4 `planned_quantization`
notes). All figures (88.5/92.6, 0.111/0.065, 26.51/25.80, 46/46), the status descriptor, the
"sidecar-bearing and NaN-sentinel files fail closed", and the "non-Windows refuses (TK2)"
statements KEPT UNCHANGED. Ledger regenerated via
`scripts/extract-capabilities-to-ledger.mjs` (CAIRN Amdt1).

## Gate results

- `cargo fmt --check`: clean.
- `cargo clippy --all-targets --all-features -D warnings`: clean.
- `cargo test --all-features`: 1255 passed / 0 failed (full log: `db6_cargo_test.txt`).
- `node scripts/check-ledger-drift.mjs`: PASS (ledger == code contract, no surface contradicts).
