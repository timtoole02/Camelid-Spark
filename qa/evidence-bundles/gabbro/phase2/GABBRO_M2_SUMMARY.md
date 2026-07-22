# GABBRO M2 — macOS platform-gate lift (atomic §9.3 ratchet PR)

**Scope claim:** this phase narrows the Amendment 3 §9 platform gate so NVFP4 admits on **Windows AND macOS** (other targets refuse), and truths-up the refusal message + every support surface in the same PR. It makes **no new support claim**: NVFP4 stays *engine-facts, NOT a supported row, not quality-competitive* (Gate G3 NO-GO stands). macOS scope is the **CPU wire lane only** — the Metal GPU kernel is GABBRO Phase M3 and is not wired.

## Provenance
- Campaign: GABBRO, Gate **G-M2** (STOP for review — **not merged**)
- Host: Apple M4, 16 GiB, macOS 26.5, Darwin arm64; worktree `<worktree>` @ `510fa51` (origin/main) + M2, branch `gabbro/m1-arm-decode`
- Depends on: G-M1 (`qa/evidence-bundles/gabbro/phase1/`); Tim's ruling folding the surface truth-up into M2
- UTC: 2026-07-18

## The change (15 files — see `m2-diff.stat`)
**Behavior (2 gates):** `!windows` → `!windows && !macos` in `src/gemma4_runtime.rs::nvfp4_windows_only_check` and the mirrored `src/runnable/admit.rs` gate.

**Tests (4 files):** 6 cfg-twinned tests moved macOS from the refuse-leg to the admit-leg; 2 new macOS-admit positive controls (`windows_only_check_admits_nvfp4_on_macos`, `admits_gemma4_nvfp4_pilot_on_macos`) that execute on this M4; the `invariant_matrix_binding` platform twin and the `nvfp4_wire_lane_refusals` non-reachability assertion re-gated.

**Message truth-up:** the typed TK2 error → **"NVFP4 is Windows/macOS-only in this release; see SUPPORT_MATRIX"**, updated at every quoting site: `src/gemma4_runtime.rs`, `src/runnable/admit.rs`, `src/api/mod.rs` (live `/api/capabilities` note), `ledger/camelid-ledger.json` (mirror), the two test files, `qa/invariant_lanes.json`.

**Surfaces:** `SUPPORT_MATRIX_v0.1.md`, `CAPABILITY_MATRIX.md`, `STATUS.md`, `COMPATIBILITY.md`, `README.md` (both NVFP4 rows), `DOCS.md`, `docs/architecture/NVFP4_FORMAT.md`, `DECISIONS.md` (**D17 addendum 9** + a §9.1 cross-reference). Each states: macOS = CPU wire lane, bit-exact (G-M1); Metal GPU = Phase M3, not wired; CUDA dp4a + RTX-3060 perf remain Windows-only; not-supported / not-quality-competitive preserved; macOS receipt cited.

## Conductor discrepancies recorded (§7: pin wins)
- **L4-metal "flip na→enforced" is a no-op.** BASALT already shipped the L4-metal `I-plat`/`I-unknown-type`/`I-sidecar`/`I-carveout` cells as `enforced` ("UPGRADE over the prescribed na", S1). GABBRO's conductor assumed they were `na`; the pin was ahead. Recorded in addendum 9.
- **Function name retained.** `nvfp4_windows_only_check` is now a mild misnomer (admits macOS); kept to avoid churn/risk in the ratchet — it is `pub(crate)`, not a user surface. Optional rename is a flagged follow-up.

## Verify (this box) — all green
- `cargo fmt --all --check`: PASS (exit 0)
- `cargo clippy -p camelid --lib --tests`: clean, no warnings
- Tests: **975 passed / 0 failed / 19 ignored** — lib 861, gemma4_capabilities 88, api_vertical_slice 4, invariant_matrix_binding 9 (the §2.4 meta-test), nvfp4_format 9, nvfp4_e4b_spotcheck 1, nvfp4_wire_lane_refusals 3. Full log: `m2-verify.txt`.

## Not done here (by design)
- Full `--all-targets` suite on ARM (scoped run only).
- CAIRN entries / Evidence-Chip / frontend (GABBRO M4 surface alignment).
- Metal NVFP4 kernel (GABBRO M3).
- No merge — STOP at G-M2 for human review.
