ARCHITECTURE SCOUT CAMELID: Q8 decode group scheduler policy boundary

Artifact type: deepening opportunity note
Status: architecture/deepening scout note; no support-contract, frontend readiness, throughput, RSS, parity-envelope, portability, default-on, or broad model-support claim.

## File Path

`docs/runtime/2026-05-24-q8-decode-group-scheduler-policy.md`

## Current Shape

The previous packed-rows4 matmul scheduler slice moved `CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK` into `ResolvedRuntimePlan` through `Q8PackedRows4MatmulSchedule`. That is now the better pattern: route permission and one scheduler value are captured together before inference helpers run.

The decode group-chunking routes have not fully crossed that same boundary yet:

- `src/inference/q8_runtime.rs` captures route gates such as `attention_qkv_decode_group_chunking`, `ffn_gate_up_decode_group_chunking`, and `ffn_down_decode_group_chunking` in `Q8RuntimeFlags`.
- `src/inference.rs` still reads decode scheduler chunk sizes directly through helper functions: `x86_q8_attention_qkv_decode_groups_per_chunk()`, `x86_q8_ffn_gate_up_decode_groups_per_chunk()`, `x86_q8_ffn_down_decode_groups_per_chunk()`, and `mac_q8_ffn_down_decode_groups_per_chunk()`.
- `src/execution_plan.rs` already treats the x86 decode chunk-size env vars as owner-gated passthrough knobs, but the runtime plan does not carry those parsed values to the route consumers.
- `src/tensor/mod.rs` remains the right deep-module owner for backend-owned packed runtime storage and tensor-family layout reinterpretation. The proposed boundary should not move packing rules out of `TensorStore`.
- `frontend/src/lib/chatGate.js` correctly keeps chat unlocked only when runtime readiness and exact support-contract compatibility both pass. Decode scheduler policy can help explain an internal path, but it must not affect frontend support gating.

The architectural gap is locality: decode group chunking is half resolved by plan and half resolved by late env reads. That makes it easier for a future route experiment to blur route ownership, scheduler policy, and product-facing readiness.

## Recommendations

Introduce a private **Q8 decode group scheduler policy** inside the resolved runtime plan, parallel to the packed-rows4 matmul schedule but scoped to single-token decode consumers.

Suggested shape:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Q8DecodeGroupSchedule {
    attention_qkv_groups_per_chunk: usize,
    ffn_gate_up_groups_per_chunk: usize,
    ffn_down_groups_per_chunk: usize,
}
```

Use it narrowly:

- Parse positive decode chunk sizes once in `ResolvedRuntimePlan::from_env()`, next to the route gates that own them.
- Keep defaults at the current helper values so the refactor is behavior-preserving.
- Pass the selected schedule values into QKV, FFN gate/up, and FFN-down decode helpers instead of reading env inside math helpers.
- Preserve `PlannerEnv` owner-gated passthrough for x86 knobs, and either add a managed owner-gated macOS FFN-down chunk-size passthrough or explicitly document why it remains outside execution-plan management.
- Keep `TensorStore` as the only owner of backend-owned packed runtime storage and tensor-family row reinterpretation.
- Keep frontend support gates unchanged. At most, expose decode schedule values as internal diagnostics after route evidence exists.

## Retain/Reject Gate

Retain only if a focused slice proves:

- `ResolvedRuntimePlan` captures decode group chunk sizes once and route helpers consume the captured values.
- Invalid, zero, absent, or stale values fall back to existing defaults without enabling any route.
- Owner-gated execution-plan cleanup still removes chunk-size passthrough values when their owning route gate is not selected.
- Existing decode group-chunking equivalence tests still compare chunked output to unchunked output without model benchmarks.
- No README, compatibility row, frontend chat unlock condition, or public support wording changes.

Reject if the slice duplicates packing policy outside `src/tensor/mod.rs`, adds scheduler values to support-contract logic, leaves mixed plan/env scheduling in the same route family, or treats scheduler-policy evidence as performance evidence.

## Next Tracer Bullet

Implement the plan-owned decode scheduler handoff only:

1. Add `Q8DecodeGroupSchedule` to `src/inference/q8_runtime.rs` and carry it on `ResolvedRuntimePlan`.
2. Replace direct calls to `*_decode_groups_per_chunk()` in QKV, FFN gate/up, and FFN-down decode helpers with schedule values passed from the route plan.
3. Update focused tests to assert plan capture and behavior-preserving chunked-vs-unchunked output for the three decode families.
4. Run:

```sh
cargo test x86_q8_attention_qkv_decode_group_chunking_matches_unchunked_triplet --lib
cargo test q8_ffn_gate_up_decode_group_chunking_matches_unchunked_pair_projection --lib
cargo test x86_q8_ffn_down_decode_group_chunking_is_default_off_and_matches_consumer --lib
cargo test planner_env_apply_restores_owned_x86_q8_passthrough_knobs --lib
```

Treat the result as internal route-policy evidence only.
