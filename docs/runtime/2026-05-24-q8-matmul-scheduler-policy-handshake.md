ARCHITECTURE SCOUT CAMELID: Q8 packed-rows4 matmul scheduler policy handshake

Artifact type: tracer-bullet implementation brief
Status: architecture/deepening scout note; no support-contract, frontend readiness, throughput, RSS, parity-envelope, portability, or default-on claim.

## File Path

`docs/runtime/2026-05-24-q8-matmul-scheduler-policy-handshake.md`

## Current Shape

Camelid now has three separate but related control surfaces for packed Q8 matmul scheduling:

- `src/tensor/mod.rs` owns backend-owned `Q8_0RuntimeStorage::PackedRows4` creation, including tensor-family row reinterpretation for `output.weight`, attention projections, and dense FFN projections.
- `src/inference/q8_runtime.rs` resolves route gates into `Q8RuntimeFlags`, but it mostly carries booleans. Scheduler tuning values such as `CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK` are still read by inference helpers rather than carried by the resolved runtime plan.
- `src/inference.rs` consumes packed runtime storage through route-specific helpers and resolvers. The packed-rows4 matmul chunk size is cached outside tests, so it is not the old per-block `getenv` problem, but the selected scheduling policy is still a global helper rather than part of the route decision.
- `src/execution_plan.rs` owns product-facing profile selection and managed experiment env cleanup. It can retain positive integer passthrough knobs only when their owner gate is selected, which is the right direction for preventing stale tuning env from leaking into unrelated profiles.
- `/api/capabilities` and `/v1/health` can expose the selected `ExecutionPlan`; the frontend support gate in `frontend/src/lib/chatGate.js` still unlocks chat only from runtime readiness plus exact support-contract compatibility, not from selected Q8 route telemetry.

The gap is not correctness of one current knob. The deeper issue is that route permission, backend-owned packed storage, scheduler chunk policy, execution-plan visibility, and frontend support boundaries are still coordinated by convention. That makes future matmul scheduler work easy to misread as product readiness or throughput evidence when it is only an internal route-policy experiment.

## Recommendation

Introduce a small internal **Q8 packed matmul scheduler policy** boundary before adding more packed-rows4 scheduler knobs.

Suggested shape:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Q8PackedRows4MatmulSchedule {
    groups_per_chunk: usize,
}
```

Use it narrowly:

- Parse positive scheduler values once while resolving the runtime plan, next to the Q8 route gates that own them.
- Pass the schedule into packed-rows4 matmul helpers or route structs instead of having math helpers reach back into env/global state.
- Keep owner-gate rules in `PlannerEnv`: a passthrough scheduler knob should survive profile application only when the owning route gate is enabled.
- Expose at most diagnostic/path-sanity wording through `ExecutionPlan` or internal telemetry. Do not turn scheduler values into support-contract rows, frontend green states, README claims, or performance promotion.
- Keep `frontend/src/lib/chatGate.js` unchanged: selected schedule can help explain which internal path ran, but chat remains blocked unless runtime readiness and exact support-contract compatibility both pass.

This is a deep-module slice, not a benchmark slice. It should make later evidence cleaner by proving which scheduler policy ran before any same-host timing work starts.

## Retain/Reject Gate

Retain if a focused slice proves:

- `CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK` is captured as a positive integer only under the packed-rows4 matmul owner gate.
- Invalid, zero, absent, or stale scheduler values fall back or clear without changing route defaults.
- Packed runtime storage is still created only by `TensorStore` and consumed only after route-specific packed-shape and interleave checks.
- `/api/capabilities` may include selected execution-plan diagnostics, but support-contract rows and frontend chat gating remain unchanged.
- Unit evidence covers the planner-to-runtime-policy handoff without requiring model benchmarks or a local server.

Reject if the slice adds scheduler state to public support claims, lets a stale env knob affect a safe/auto profile, duplicates tensor-family packing policy outside `src/tensor/mod.rs`, or treats a scheduling knob as throughput evidence without parity plus same-host timing.

## Next Tracer Bullet

Implement only the scheduler-policy handoff:

1. Add a private `Q8PackedRows4MatmulSchedule` to the Q8 runtime planning boundary and carry `groups_per_chunk` through `ResolvedRuntimePlan`.
2. Refactor `q8_packed_rows4_matmul_parallel_chunk_floats(...)` to accept the resolved schedule value instead of reading `CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK` directly.
3. Keep the existing `PlannerEnv` owner-gated passthrough behavior and add one assertion that the passthrough is not restored unless `CAMELID_X86_Q8_PACKED_ROWS4_MATMUL` is enabled.
4. Run:

```sh
cargo test x86_q8_packed_rows4_matmul_chunk_groups_env_override --lib
cargo test planner_env_apply_restores_owned_x86_q8_passthrough_knobs --lib
cargo test capabilities_can_include_selected_execution_plan --lib
```

Treat the result as internal route-policy evidence only. It is not throughput, RSS, profiling, parity-envelope, frontend readiness, portability, default-on, or support-contract promotion evidence.
