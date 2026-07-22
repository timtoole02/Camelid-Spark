# Neural Field — evidence index and gate decision

Conductor: `OBSERVATORY_NEURAL_FIELD_CONDUCTOR.md`. Branch: `feat/neural-field`.

## Phase 5 gate — PASSED 2026-07-02 → default renderer flipped to Neural Field

| Gate item | Status | Where |
| --- | --- | --- |
| 1. Frame captures (all contract states) | ✅ | `frames/` — see table below |
| 2. Perf receipt, real 50-token TinyLlama Q8_0 run @ DPR 1 and 2 | ✅ p95 2.3ms @DPR1 (≤16.7ms) | `PERF.md` |
| 3. Truthfulness audit incl. RUN_STALE_MS settle | ✅ | `TRUTHFULNESS.md` |
| 4. Validation commands | ✅ `npm run build` green (no `lint` script exists in frontend/package.json — recorded in `PHASE0_RECON.md` §5) | — |

Flow Bench remains available behind the header toggle; a previously stored renderer
choice in `localStorage["camelid.observatory.renderer"]` always wins over the default.

## Frames (all from real TinyLlama 1.1B Q8_0 generations; capture harness =
`scripts/neural-field-evidence.mjs`)

| State | GPU-resident lane | CPU lane (`CUDA_VISIBLE_DEVICES=-1`) |
| --- | --- | --- |
| idle (connection live) | `idle-live.png` | `idle-cpu.png` |
| prefill (motes + plane fill) | `prefill.png` | `prefill-cpu.png` |
| decode | `decode.png` (token-paced multi-front sweep + honesty note) | `decode-cpu.png` (layer-event front) |
| sampler bloom + KV column | `sampler-bloom.png` | `sampler-kv-cpu.png` |
| error (real `inference_error` mid-run) | `error.png` | — |
| finished ok (post-exhale rest) | `finished-ok.png` | `finished-cpu.png` |
| receipt burst (real sealed receipt, `camelid_receipt: true`) | `receipt-burst.png` | — |
| reduced motion (discrete steps, no motes) | `reduced-motion.png` | — |
| unavailable (backend down, 0.04 dim + notice) | `unavailable.png` | — |

Perf raw numbers: `capture-summary.json`.

## Out-of-scope observations for follow-up (no backend changes made, per conductor)

- `prefill_started.path` reports `"auto"`, not one of the four values documented in
  `src/telemetry.rs`.
- `layer_started` is nearly always throttled away on fast CPU decode (shared 15ms gap
  with `layer_completed`); the front is driven by both events (see `TRUTHFULNESS.md`).
