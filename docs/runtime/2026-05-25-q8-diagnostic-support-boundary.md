ARCHITECTURE SCOUT CAMELID: Q8 diagnostic support-boundary handoff

Artifact type: deepening opportunity note
Status: architecture/deepening scout note; no support-contract, frontend readiness, throughput, RSS, parity-envelope, portability, default-on, or broad model-support claim.

## File Path

`docs/runtime/2026-05-25-q8-diagnostic-support-boundary.md`

## Current Shape

Camelid now has a useful separation between internal Q8 route selection and user-facing support readiness, but the separation is still mostly enforced by convention:

- `src/tensor/mod.rs` owns backend-owned packed Q8 runtime storage through `Q8_0RuntimeStorage::PackedRows4`, tensor-family repack eligibility, and tensor shape reinterpretation.
- `src/inference/q8_runtime.rs` owns fail-closed Q8 route flags and the first plan-owned schedule value through `Q8PackedRows4MatmulSchedule`.
- `src/inference.rs` consumes those routes, records Q8 schedule telemetry, and still contains route-family-specific denial behavior for attention, FFN, and output projections.
- `src/execution_plan.rs` owns profile selection, managed env cleanup, and owner-gated passthrough knobs. It exposes selected backend/path diagnostics through `ExecutionPlan`.
- `src/api/mod.rs` returns both `execution_plan` diagnostics and the `support_contract` / `model_compatibility` rows in `/api/capabilities`.
- `frontend/src/lib/chatGate.js` correctly unlocks chat only when runtime readiness and exact support-contract compatibility both pass.
- `frontend/src/lib/capabilities.js` already treats template, checked-context, and throughput lanes as row-scoped readiness, not broad route or model-family evidence.

The gap is not that the current frontend is over-promoting Q8 routes. The gap is that the backend now exposes more internal execution-plan detail next to support-contract rows, while the codebase has no named boundary for which runtime diagnostics are allowed to influence product readiness. As more Q8 route-ledger and scheduler-policy work lands, this boundary needs a small, testable contract so internal diagnostics do not accidentally become support claims.

## Deepening Opportunity

Introduce a private **runtime diagnostic boundary** concept that states what may cross from Q8/inference internals into API/frontend surfaces.

Suggested rule:

```text
Runtime diagnostics may explain the selected internal route.
Only support-contract rows may unlock or widen user-facing support.
```

The implementation does not need a public type first. A focused slice can start with tests and helper naming:

- Keep `ExecutionPlan.selected_q8_path`, `prefill_path`, `decode_path`, and `reasons` diagnostic-only.
- Keep Q8 schedule telemetry inside generation diagnostics and evidence bundles, not compatibility rows.
- Keep `/api/capabilities.support_contract` and `.model_compatibility` as the only compatibility source consumed by frontend gates.
- Keep frontend row-lane helpers keyed to exact rows and evidence fields, not route names such as `x86_experimental_q8_0_avx2` or `mac_validated_q8_0_repack`.
- Add a targeted regression test that a loaded model with an optimized Q8 execution plan remains chat-blocked when its exact compatibility row is not supported.

This is a deep-module boundary for product truth, not a performance slice.

## Recommendations

1. Treat `ExecutionPlan` as route diagnostics and support-contract context, not as a support oracle.
2. Add a named helper or test fixture in the frontend for "diagnostic plan present, contract unsupported" so future UI work has a clear blocked state to reuse.
3. Keep Q8 route-ledger, decode scheduler, and packed matmul scheduler notes explicitly internal-only until a row-scoped evidence bundle promotes a support-contract field.
4. Avoid copying `selected_q8_path`, route telemetry, or schedule values into `model_compatibility` unless the support contract is intentionally updated with evidence.
5. Keep `TensorStore` and inference route ownership out of frontend terminology; frontend should speak in exact rows, runtime readiness, and row-scoped lanes.

## Retain/Reject Gate

Retain if a unit-only slice proves:

- `/api/capabilities` can include an optimized or experimental `execution_plan` while `model_compatibility` remains unchanged.
- `getChatGateState(...)` keeps `chatUnlocked=false` when runtime readiness is true but exact-row support is false.
- Frontend lane helpers do not treat Q8 route names, schedule values, or generation telemetry as readiness evidence.
- No README, compatibility row, frontend copy, or public support wording broadens from internal route diagnostics.

Reject if the slice makes route names part of exact-row matching, promotes Q8 telemetry into frontend support gates, or treats selected execution-plan diagnostics as parity, throughput, portability, or default-on evidence.

## Next Tracer Bullet

Add a support-boundary regression slice:

1. In Rust API tests, construct `capabilities_response_with_plan(Some(plan))` with an optimized Q8 `ExecutionPlan` and assert the supported rows and `support_contract.current_gate` do not change because the plan is present.
2. In frontend smoke tests or a small unit harness, pass capabilities with an execution plan plus an unsupported exact row and assert `getChatGateState(...).chatUnlocked === false` while runtime readiness is true.
3. Add one fixture name such as `optimizedPlanUnsupportedRow` so future frontend work can reuse the blocked diagnostic state.
4. Run only targeted unit/smoke commands; no model benchmarks and no local server.

Treat the result as internal support-boundary evidence only.
