# Phase 6.1 — Observatory defect diagnosis (2026-06-12)

Method: scripted real-traffic reproduction (puppeteer, supported 3B row loaded,
instrumented EventSource/rAF), per the Phase 6 gate definition. Phase 6 Telemetry
surfaces (TTFT/tok-s/duration/error tiles, per-model, request log, health history)
re-verified populating from real traffic — no defects found there.

## DEFECT 1 — runs are invisible if they happen while the view is unmounted
**Repro:** open #chat, complete a real generation, then navigate (in-app) to the
Observatory. Expected: the just-finished run's details. Actual: `Status idle ·
Model —` — the waiting/idle state, as if no inference ever happened.
**Root cause:** `useInferenceTelemetry` creates ONE STORE PER MOUNT
(`useMemo(() => createInferenceTelemetryStore(), [])`) and calls
`store.disconnect()` on unmount. SSE events have no buffer; anything emitted while
the user is on any other tab is lost, and accumulated state is destroyed on every
navigation. Phase 7's route-lazy-loading makes the blind window strictly larger.
This is the same failure family as the capture-harness hash-navigation bug:
lifecycle tied to mount when the data is session-scoped.
**Fix:** module-level shared store, connected app-lifetime on first use
(reconnect only when the API base actually changes), never disconnected on
unmount. Guarded by a smoke:ui assertion (no per-mount store creation, no
unmount disconnect).

## DEFECT 2 — connection churn on every visit (same root cause, lesser symptom)
21 mounts produced 41 EventSource open/close cycles (REACT strict double-effect
included). Each reconnect has a gap where mid-run events are missed, so even
visiting DURING a run could show a torn run state. Fixed by the same shared-store
change; balance assertions showed no leak (opened == closed), and rAF activity
after unmount is zero — teardown itself was clean.

## Checked and NOT defective
- rAF loops / listeners after navigation: 0 rAF requests during 1s with the view
  unmounted; EventSource opened == closed across 20 navigate-away/return cycles.
- Backend stream: emits the full event sequence on real runs (verified during the
  stale-binary incident follow-up; the binary fix is recorded in the session log —
  a backend binary predating the Observatory merge serves 404 for
  /api/telemetry/stream and the view correctly sits in its waiting state).
- usage-absent / divide-by-zero on zero-token runs: the Phase 6 store null-guards
  (`Number.isFinite`) and the view renders '—'; no NaN surfaced under a 1-token
  and an aborted run.
- localStorage schema drift: the Observatory persists nothing; N/A.
