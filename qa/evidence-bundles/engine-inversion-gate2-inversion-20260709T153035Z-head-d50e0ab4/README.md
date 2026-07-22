# Gate 2 receipt — engine inversion complete: lock deleted, invariants by construction

Mission: API Engine Inversion (docs/recon/ENGINE_INVERSION_CONDUCTOR.md).
Candidate head: `d50e0ab4` (Phases 2a `93dc20c3` + 2b `d50e0ab4` on
`feat/engine-inversion`); baseline: pin `ffada00f`.

## What changed at this gate

One dedicated engine worker thread now executes EVERY decode (non-streaming,
streaming, multi-choice, receipt replay) and every engine-owned-state mutation
(GPU-runnable parity probe, `reset_resident_caches` on unload) behind a bounded
queue (`CAMELID_QUEUE_DEPTH`, default 8). `generation_lock` is deleted; the D5
lock test is superseded by `engine::tests::engine_executes_at_most_one_job_at_a_time`.
Prep (tokenization/rendering) runs outside serialization on every path (D3).
Streaming maps engine events onto unchanged SSE chunk shapes (D4 ping-pong gone).

## Criteria and results

1. **All Phase 0/1 tests green, assertions unchanged** — P0-T1/T2/T3 re-anchored
   to the engine world and passing; full unit suite 677 passed / 0 failed
   (55 ignored as usual); fmt + clippy `--all-targets --all-features -D warnings`
   clean.
2. **New invariant tests green** — (a) engine executes at most one job at a
   time (measured on compute), (b) prep completes while a decode occupies the
   engine (D3), (c) queue-full is a typed 503 (`engine_queue_full`) and depth
   drains, (d) stream timeout event carries the exact pre-inversion payload.
3. **Parity receipts replay clean — full matrix, byte-identical** (`parity/`):
   baseline-vs-candidate canonical outputs identical for TinyLlama-1.1B-Q8_0
   and Llama-3.2-1B-Q8_0 across: greedy chat/completion/SSE-stream, SAMPLED
   (temperature 0.8 + fixed seed) chat/SSE-stream, repeat-request
   (prompt-cache hit on the CPU lane), on BOTH the default resident-CUDA lane
   and the `--deterministic` CPU lane. This banks Gate 3's golden-transcript
   matrix as well (SSE bytes identical modulo timing values — here literally
   identical because canonicalization only strips uuids).
4. **Concurrency smoke** (`soak/soak-phase2b-600s.json`): 10-minute soak,
   6 workers, mixed stream/non-stream with random mid-flight client
   disconnects (3492 aborts — the T1/T3 triggers, continuously):
   **0 garbled responses** (every completed response byte-matched the
   canonical greedy answer), **0 panics**, **0 unexpected 5xx**, RSS stable
   (1386 → 1388 MB). 4129 typed `engine_queue_full` 503s under deliberate
   overload — bounded-queue backpressure working as designed (D2), retryable
   and observable, never an invisible pile of waiters.
5. **TTFT under queued load improved** (D3 removal): stream time-to-first-byte
   while a long decode occupies the engine — baseline median **2892 ms**
   (SSE head stalled behind the lock; `soak/soak-baseline-60s.json`) vs
   candidate median **12.4 ms** (~233x). The role chunk now arrives
   immediately; only content waits its queue turn.

## Behavioral deltas (documented, deliberate)

- Burst beyond queue depth: typed 503 + Retry-After instead of unbounded
  invisible waiting.
- A decode wedged INSIDE a single forward waits instead of "503 + orphan"
  (the lock can never be safely freed while compute might touch KV state).
- Streaming decode may run ahead of a slow client by up to 32 deltas
  (bounded events channel), then parks.
- Streaming completion now marks GAIT safe-boot health on clean finish
  (previously only non-streaming did).

## Verdict

GATE 2: **PASS**. The invariants ledger items 1–4 hold by construction and
are enforced by tests; item 5 (parity) is receipt-proven above.
