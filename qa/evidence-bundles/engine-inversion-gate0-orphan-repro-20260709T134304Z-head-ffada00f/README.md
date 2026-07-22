# Gate 0 receipt — orphan-decode hazard demonstrated on pristine `ffada00f`

Mission: API Engine Inversion (docs/recon/ENGINE_INVERSION_CONDUCTOR.md).
Claim being proven: `AppState::generation_lock`'s guard lifetime is decoupled from
the `spawn_blocking` decode it guards, so handler-future drop (client disconnect,
server generation timeout, SSE hangup) frees the lock while the decode keeps
running, and the next request decodes CONCURRENTLY with the orphan.

## What this bundle contains

- `logs/orphan-repro-tests-FAIL.log` — `cargo test --lib orphan_decode` at source
  head `ffada00f` plus ONLY the Phase 0 test-only instrumentation (a `#[cfg(test)]`
  decode-concurrency probe inside the two blocking decode closures, and the three
  repro tests). No production code path is altered; the probe is compiled out of
  release builds.

## Result: all three trigger paths reproduce the hazard

Each test asserts the DESIRED invariant — the lock is never acquired while a decode
is still live, and two decodes never overlap. All three FAIL on the pin with
`orphans_at_next_acquire = 1`:

| Test | Trigger modeled | Failure observed |
|---|---|---|
| `generation_timeout_must_not_orphan_decode` (P0-T2, deterministic) | server's own `CAMELID_GENERATION_TIMEOUT_MS` 503 (`tokio::time::timeout` drops the JoinHandle, decode detaches) | lock acquired by request B while A's decode still on the blocking pool |
| `client_disconnect_must_not_orphan_decode` (P0-T1) | client drops TCP mid non-streaming request (handler future dropped at its await point — mechanically identical to hyper's disconnect behavior) | same overlap |
| `stream_disconnect_must_not_orphan_decode_step` (P0-T3) | client hangs up mid SSE stream (real `stream_completion` response body dropped between polls) | same overlap (window = one token step) |

Test-sleep hook (`CAMELID_TEST_GENERATION_STEP_SLEEP_MS`, `#[cfg(test)]`-gated,
pre-existing at the pin) stands in for a long decode step; the probe counts live
decode workers on the blocking pool, independent of guard lifetime.

## Environment

- Source head: `ffada00f` (main, 2026-07-08), branch `feat/engine-inversion`,
  working tree modified only by the Phase 0 instrumentation/tests + mission docs.
- rustc 1.95.0 / cargo 1.95.0, Windows 11 x86_64, debug (`test`) profile.
- The tests are CPU-only tiny-fixture decodes; the CUDA banner in the T3 log is
  device enumeration only (no model resident).

## Gate 0 verdict

GO — overlap demonstrated deterministically on T2 and on both disconnect paths.
Phase 1 (cancellation + guard-rides-with-compute) proceeds; these tests must flip
to PASSING at Gate 1 with no other assertion change.
