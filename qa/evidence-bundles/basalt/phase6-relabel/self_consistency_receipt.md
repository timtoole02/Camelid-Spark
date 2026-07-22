# BASALT NVFP4 relabel + execution-truth pass — self-consistency receipt

Date: 2026-07-17. Branch: `basalt/nvfp4-relabel-execution-truth` (off `main` = `bb8382a6`, post-#474).
Conductor: "BASALT — NVFP4 relabel + execution-truth pass (post-#474 reconciliation)".
Scope: diff-only. Two word-retirements + four nuance-adds + two verify-only surfaces.
**No figure changed. Drift gate green.**

## 1. Edits applied (5 files)

| file | edit | type |
|---|---|---|
| `README.md` :218 (NVFP4 model row) | `**Experimental (BASALT) …**` → `**Pilot lane (gemma4 wire + CUDA), Windows-only — receipted engine facts, NOT a supported row, not quality-competitive.**`; + compact execution-truth clause after "(46/46 bit-identical)" | word-retire + nuance |
| `README.md` :204 (Runnable/experimental lane row, NVFP4 entry) | + compact clause after "…NaN-sentinel scale bytes"; **lane name unchanged** | nuance only |
| `SUPPORT_MATRIX_v0.1.md` :45 (NVFP4 pilot row) | status `Runnable/experimental, NOT a supported row, Windows-only` → `Pilot lane (gemma4 wire + CUDA), Windows-only — receipted engine facts, NOT a supported row`; + compact clause after "gemma4-E4B pilot carve-out" | word-retire + nuance |
| `src/api/mod.rs` :2831 (NVFP4 `planned_quantization` notes) | + execution-truth clause after "…no parity claim;" | nuance only |
| `ledger/camelid-ledger.json` (NVFP4 entry) | **regenerated** from the edited `mod.rs` literal via `scripts/extract-capabilities-to-ledger.mjs` | derived |
| `docs/architecture/NVFP4_FORMAT.md` (overview) | + full-form execution-truth clause | nuance only |

**Method note (deviation from conductor §3.4, correct per CAIRN Amendment 1):** §3.4 says
"edit `ledger/camelid-ledger.json`". The ledger is a *derived* artifact — CODE (`src/api/mod.rs`)
is the source of truth (CAIRN Amendment 1), and editing the JSON directly would fail drift
check A (freshness: ledger ≠ code). So the clause was added to the `mod.rs` notes literal and
the ledger regenerated. Result is identical intent, drift-green.

## 2. Self-consistency pass (QUANT_TRUTH-style)

- **"experimental" retired for NVFP4:** ✓ both occurrences replaced (README:218 "Experimental (BASALT)",
  SUPPORT_MATRIX:45 "Runnable/experimental"). Grep confirms both strings gone.
- **Execution-truth clause present on every surface tying NVFP4 to the runnable lane:** ✓
  README:204, README:218, SUPPORT_MATRIX:45, `mod.rs`/ledger notes, docs page. Each states the
  gemma4-E4B pilot executes via `gemma4_runtime` (CPU wire + CUDA-resident), **not** the generic
  runnable serve bridge, and refuses generic runnable admission on its BF16 tensor (`per_layer_model_proj`,
  G2 §6b).
- **No surface says or implies NVFP4 is runnable-served or "Supported":** ✓ verified. CAPABILITY_MATRIX:107
  and STATUS:203 (verify-only) already made no runnable-serve claim → unchanged.
- **Figures preserved verbatim:** ✓ 88.5%/92.6%, 0.111/0.065, 26.51/25.80, 3479/5559, 46/46, 6/9,
  type id 40, sha256 `eb293344…` — all present, none altered (grep-confirmed present in the same
  file counts as before the pass).
- **DO-NOT-TOUCH audit:** ✓ the non-NVFP4 "experimental" usages (README Q8_0/Q4_0 E4B CUDA lanes,
  Phi-3 HOLD, DiffusionGemma, the generic "Runnable/experimental lane" name, CAPABILITY_MATRIX
  lane-history lines) are unchanged. `git diff` touches **only** NVFP4-adjacent lines: 0 non-NVFP4
  lines changed in README.
- **Drift gate:** ✓ `check-ledger-drift.mjs` → "ledger drift check passed". `cargo fmt --check` clean;
  `cargo build --lib` compiles (the `§` in the notes literal is valid UTF-8).

## 3. Carry-forward for Tim (surfaced, not resolved — conductor §6)

1. **§3.2-vs-§5 descriptor nit (one-word).** §3.2's explicit SUPPORT_MATRIX replacement string
   ("…NOT a supported row") is shorter than §2(a)/§3.1's README descriptor ("…NOT a supported row,
   **not quality-competitive**"), yet §5 asks the two to "read identically." I applied §3.2 verbatim
   (specific instruction governs). The descriptor **stem** is byte-identical on both surfaces, and
   "not quality-competitive" is present on both (README status cell + SUPPORT_MATRIX detail cell) —
   so neither surface under-states. If you want the status *cells* byte-identical, append
   ", not quality-competitive" to SUPPORT_MATRIX:45's status column (trivial, truth-preserving).
2. **Generic-lane rename** — not done (conductor §4/§6: "Runnable/experimental lane" name unchanged;
   renaming touches every format in the lane, your call).
3. **Q8_0/Q4_0 E4B CUDA "experimental" lanes** — a separate honesty question (they lack committed
   bundles); left as-is per §6.
4. **Ornith-9B "certified-exact" vs ORNITH Phase-0 recon** — not noticed as a conflict during the
   NVFP4 edits (CAPABILITY_MATRIX was verify-only, unedited); not investigated (out of this pass's scope).
5. **D-B6 (gemma4 promotion) and §2.4 (matrix-mechanism deviation)** — still unsigned, orthogonal, unchanged.

Gate: STOP for Tim. This pass changes labels and adds one execution-truth clause; it asserts
nothing new about capability and moves no number.
