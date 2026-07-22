# Phase 8B after — same prompt, same 695-token greedy response (deterministic A/B)

| metric | baseline | after | gate |
| --- | --- | --- | --- |
| send → user visible | 3.8 ms | 4.6 ms | <1 frame ✓ |
| long tasks >50ms (chat) | 0 | 0 | zero ✓ |
| frames >33ms / >100ms | 2 / 0 | 1 / 0 | near-zero ✓ |
| markdown parse | full re-parse × 633 flushes (O(n²)) | stable blocks parse once (React.memo prefix); only open tail re-parses (O(n)) | structural ✓ |
| open-fence highlighting | per flush | deferred until fence closes (decision recorded) | ✓ |
| footer | layout shift on appear | space reserved during stream (zero shift) | ✓ |
| scroll | stick-to-bottom only | + "jump to latest" affordance when scrolled up | ✓ |
| pacing | none (bursty) | ≤150ms lag bound, instant drain, byte-identical (smoke-proven) | ✓ |
| containment | none | contain: layout style on turns | ✓ |
| main JS chunk | 104.10 kB gz | 103.72 kB gz | budget ✓ |

Abort re-verified (interrupted state + drained pacer, partial intact); SSE-error path
covered by smoke:streaming/integration (unchanged parser). Recordings:
stream-baseline.mp4 / stream-after.mp4 (same prompt).
