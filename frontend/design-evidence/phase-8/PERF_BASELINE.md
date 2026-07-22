# Phase 8B baseline — one real 695-token streamed response (code-heavy), M4, dev build

| metric | value |
| --- | --- |
| send → user message visible | 3.8 ms (<1 frame — optimistic send already in place) |
| send → first painted token | 1038.8 ms (TTFT 931ms backend + ~108ms pipeline) |
| long tasks >50ms during stream | 0 (worst: 0ms) |
| frames >33ms / >100ms | 2 / 0 of 2064 |
| content flushes (rAF-coalesced) | 633 — stream batching ALREADY per-rAF (original design) |
| markdown parses | full re-parse per flush: 633 full parses of growing content (O(n²) scaling risk; fine at 695 tokens on M4) |
| footer | appears post-stream with a visible layout shift (no reserved space) |
| scroll | stick-to-bottom present; no "jump to latest" affordance when scrolled up |

Success = hold the zeros while removing the O(n²) parse path, the footer shift, and
adding honest pacing (≤150ms lag bound, byte-identical final text).
