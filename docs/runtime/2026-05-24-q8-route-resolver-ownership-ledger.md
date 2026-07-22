ARCHITECTURE SCOUT CAMELID: Q8 projection route resolver ownership ledger

File path: `docs/runtime/2026-05-24-q8-route-resolver-ownership-ledger.md`

## Context

Camelid is close to a deep-module boundary for Q8 projection routing, but it is not yet expressed as one reusable contract.

- `src/tensor/mod.rs` owns backend-owned packed runtime storage through `Q8_0RuntimeStorage::PackedRows4` and `q8_0_runtime_packed_rows4_for_tensor(...)`. It also owns tensor-family row reinterpretation in `q8_repack_linear_shape(...)`.
- `src/inference/q8_runtime.rs` owns the resolved runtime plan and now carries `Q8PackedRows4MatmulSchedule` beside Q8 route gates.
- `src/inference.rs` owns route-specific Q8 projection resolvers for attention QKV, FFN gate/up, FFN-down, attention-output, and output projection.
- `src/execution_plan.rs` keeps support claims fail-closed and strips owner-gated passthrough knobs when the owning route is not selected.
- `frontend/src/lib/chatGate.js` and `frontend/src/lib/capabilities.js` keep frontend support gates tied to runtime readiness plus exact support-contract rows, not internal Q8 route or scheduler telemetry.

The remaining gap is not that any single route is obviously wrong. The gap is that route ownership rules are still encoded separately per projection family. That makes future Q8 work easy to blur across four different concerns: permission, packed-storage shape, scheduler policy, and product readiness.

## Deepening Opportunity

Introduce a small private **Q8 projection route ledger** concept before adding more Q8 projection variants.

The ledger should not be a public API. It should be an internal implementation pattern that each resolver satisfies:

```rust
struct Q8ProjectionRoutePolicy {
    family: &'static str,
    route: &'static str,
    row_policy: Q8ProjectionRowPolicy,
    storage_policy: Q8ProjectionStoragePolicy,
    schedule_policy: Q8ProjectionSchedulePolicy,
    telemetry_prefix: &'static str,
}
```

The point is not the exact struct shape. The point is to make every Q8 projection route state the same facts locally:

- Which `ResolvedRuntimePlan` gate enables the route.
- Whether the route is decode-only, prefill-only, or single-owner.
- Whether it requires backend-owned packed runtime storage.
- Which schedule values are already resolved and which late env reads remain.
- Which denial reasons must be emitted when the route fails closed.
- Whether route telemetry is internal-only and must not widen frontend support.

## Current Asymmetry

The route families are converging but still uneven:

- Packed-rows4 matmul routes now receive `runtime_plan.q8_packed_rows4_matmul_schedule`, which is the right direction.
- QKV and FFN gate/up decode group chunking still rely on direct chunk-size helpers in `src/inference.rs`; FFN-down has similar late group-size reads. This matches the existing decode-scheduler-policy note and should be treated as route-policy cleanup, not throughput evidence.
- FFN gate/up and FFN-down route resolvers record explicit denial telemetry for several route failures. Attention QKV currently returns `Ok(None)` for comparable denial states without symmetrical denial telemetry.
- `frontend/src/lib/chatGate.js` remains correctly support-contract-bound: `chatUnlocked` requires both runtime readiness and exact-row support. Route telemetry must stay diagnostic-only unless a separate support-contract evidence bundle promotes a row.

## Recommendations

1. Keep `src/tensor/mod.rs` as the sole owner of packed Q8 runtime storage and tensor-family row reinterpretation.
2. Move remaining decode group schedule values into `ResolvedRuntimePlan` before adding new QKV, FFN, or output projection route variants.
3. Normalize route denial telemetry across QKV, FFN gate/up, FFN-down, attention-output, and output projection so failed-closed behavior is visible without reading every resolver.
4. Keep execution-plan passthrough ownership in `src/execution_plan.rs`; a stale scheduler knob must be removed when the owning route gate is off.
5. Do not surface route-policy success as frontend support, throughput readiness, portability, or default-on evidence. The frontend should continue to unlock chat only from exact-row support plus runtime readiness.

## Next Tracer Bullet

Add a unit-only Q8 route-ledger slice for attention QKV:

1. Add a private policy helper for the attention QKV decode and packed-rows4 matmul routes.
2. Carry decode group chunk sizes through `ResolvedRuntimePlan` for QKV only.
3. Replace the QKV resolver's silent `Ok(None)` branches with the same internal denial telemetry style used by FFN gate/up and FFN-down.
4. Gate with targeted tests proving invalid, absent, or stale chunk values fail closed and do not affect frontend support-contract logic.

Retain only as internal route-policy evidence. Reject if the slice moves packed-storage ownership out of `src/tensor/mod.rs`, adds route telemetry to frontend support gates, or treats the cleanup as performance evidence.
