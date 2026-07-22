# Neural Field — Phase 0 recon receipt

Date: 2026-07-02. Branch: `feat/neural-field` off `main` (d60699cc).
Scope check: frontend-only; no backend changes needed for v1. All items below were
verified by reading the current source, not from memory.

## 1. Telemetry event vocabulary consumed

`src/telemetry.rs` (`TELEMETRY_SCHEMA = "camelid.telemetry/v1"`), `Event` enum, lines 72–165.
Serialized with `#[serde(tag = "event", rename_all = "snake_case")]`, wrapped in an
`Envelope { seq, t_ms, request_id?, model_id?, ...event }`.

Copied verbatim (fields the Neural Field consumes are all of them except worker events,
which stay out of scope for this canvas):

```rust
pub enum Event {
    InferenceStarted {
        backend: String,
        quantization: String,
        architecture: String,
        prompt_tokens: usize,
        max_tokens: u32,
        context_length: usize,
        temperature: f64,
        stream: bool,
    },
    InferenceFinished {
        status: &'static str, // "ok" | "error" | "disconnected"
        finish_reason: Option<String>,
        completion_tokens: usize,
        total_ms: u64,
        ttft_ms: Option<u64>,
        decode_tps: Option<f64>,
        prefill_tps: Option<f64>,
        error: Option<String>,
    },
    PrefillStarted {
        prefill_tokens: usize,
        /// "gpu_resident" | "layer_major" | "chunked" | "single_token"
        path: &'static str,
        layers_total: usize,
    },
    PrefillProgress { tokens_done: usize, tokens_total: usize },
    DecodeStarted { context_position: usize },
    LayerStarted { layer: usize, layers_total: usize },
    LayerCompleted { layer: usize, layers_total: usize, duration_us: u64 },
    TokenDecoded {
        token_id: Option<u32>,
        context_position: Option<usize>,
        layers_total: Option<usize>,
    },
    KvCacheUpdated { position: usize, capacity: usize, approx_bytes: Option<u64> },
    SamplerStep {
        chosen_token_id: u32,
        mode: &'static str, // "greedy" | "sampling"
        candidates: Vec<SamplerCandidate>, // { token_id: u32, prob: f32 }
    },
    InferenceError { code: String, message: String },
    ReceiptWritten { receipt_id: String, reproducible: bool, gguf_sha256: Option<String> },
    WorkerNodeActive { node: String, detail: Option<String> },  // not consumed by Neural Field
    WorkerNodeIdle { node: String },                            // not consumed by Neural Field
    WorkerNodeError { node: String, error: String },            // not consumed by Neural Field
}
```

Throttling (relevant to choreography density expectations): layer events min-gap 15ms,
KV 50ms, sampler 80ms, prefill progress 33ms. Lifecycle events always pass. The client
also receives `hello` and `lagged` transport notices (`lagged.skipped` = dropped count).

## 2. Store fields the renderer binds

`frontend/src/lib/inferenceTelemetry.js` — all confirmed present:

| Binding | Field | Confirmed |
| --- | --- | --- |
| Layer count | `run.layersTotal` | line 42 (set from `prefill_started` / layer events / `token_decoded`) |
| Prefill state | `run.prefill.{tokens, done, path, startedAtMs, endedAtMs}` | line 40 |
| Decode pacing | `run.decode.{startedAtMs, tokens, lastTokenAtMs, tokenIntervalMs}` | line 41 |
| KV occupancy | `run.kv.{position, capacity, approxBytes}` | line 45 |
| Sampler | `run.lastSampler` (`{ chosenTokenId, mode, candidates }`) | line 46 |
| Phase | `run.phase` (`idle | running | prefill | decode | finished | error`) | line 39 (note: `'running'` appears transiently between `inference_started` and `prefill_started`, and for mid-run joins — the brief's list omits it; renderer treats it as active-but-unphased) |
| Connection | `connection` via `store.getConnection()` (`connecting | live | unavailable`) | lines 18–22 |
| GPU/CPU lane discrimination | `run.layerEventsSeen` | line 44 |
| Stale-run settle | `store.isRunStale()` / `RUN_STALE_MS = 30000` | lines 16, 294–296 |
| Per-frame raw events | `store.drainEvents()` | lines 310–313 |

`store.drainEvents()` exists exactly as described: drains `pending` (raw envelopes with
`receivedAtMs` stamped), called once per frame by the canvas.

## 3. Data path confirmation

The Neural Field consumes **`useInferenceTelemetry(apiBase)` + `store.drainEvents()`**
— the `InferenceCanvas` pattern (`frontend/src/components/observatory/InferenceCanvas.jsx`,
which drains once per frame and fans events to renderer modules with
`onEvent(evt, frame)` / `draw(ctx, frame)` interfaces).

It does **NOT** use the client-side `lib/telemetryLog` bus that FlowBench consumes.
Rationale: the Neural Field visualizes backend-reported model internals (layers, KV,
sampler candidates); the backend SSE store (`camelid.telemetry/v1`) is the source of
truth for those. The telemetryLog bus is a client-side request-lifecycle instrument and
carries none of the model-internal events.

The hook is a shared app-lifetime store (`useInferenceTelemetry.js` lines 15–26), so a
mode toggle or tab navigation does not lose run state. `ensureInferenceTelemetryConnected`
is called from the app shell.

Note: `InferenceCanvas.jsx` and its modules (`layerVisualizer.js`, `tokenParticles.js`,
`kvCacheTrail.js`, `samplerBloom.js`, `clusterConstellation.js`) are currently **not
mounted anywhere** (the Flow Bench replaced that centerpiece in Phase 6.1); they remain
in-tree as the reference implementation for the rAF/DPR/ResizeObserver skeleton, the
GPU-sweep semantics, and the receipt-burst concept. The Neural Field copies those
patterns rather than importing the modules.

## 4. Frame budget

Target: 60fps (p95 frame ≤ 16.7ms) at 1600×900 CSS px, DPR-aware, on this dev machine
(Windows 11, RTX 3060 Laptop; Canvas2D is CPU-composited so the relevant budget is
single-core JS + raster time). Drawable budget ≤ ~4,000 (nodes + edges) so the per-frame
painter's sort stays cheap: at 22 layers × 18 nodes = 396 nodes + 378 edges ≈ 800
drawables (well inside). Faked glow via wide low-alpha underlay strokes only; no
`ctx.shadowBlur` anywhere. Perf receipt lands in `PERF.md` at Phase 5 with the gate
and the two prescribed reduction steps if missed.

## 5. Discrepancies found (and dispositions)

1. **No `lint` script exists** in `frontend/package.json` — the conductor's validation
   command `npm run lint` cannot run. There is no eslint config in `frontend/` either.
   Disposition: validation = `npm run build` (vite) plus the existing observatory smoke
   scripts where applicable (`npm run smoke:observatory` requires a running backend).
   Not adding a linter — that would be a new dev dependency outside this mission's scope.
2. **No test runner** in `frontend/package.json` scripts (only `node scripts/*.mjs`
   smokes). Per the Phase 1 instruction, `projection.test.js` is skipped and noted here;
   no test framework is added.
3. **`run.phase` includes `'running'`**, a transient value not listed in the brief
   (set between `inference_started` and `prefill_started`, and on mid-run joins).
   Renderer treats `running` as "awake, no phase-specific choreography yet".
4. `MetricsOverlay.jsx` / `ProofOverlay.jsx` exist but are not mounted in the current
   `InferenceObservatoryView` (only `DetailsPanel` is). "MetricsOverlay/DetailsPanel
   continue to work unchanged" is therefore satisfied trivially: DetailsPanel stays
   mounted below the stage in both modes; nothing else is touched.

## 6. Palette bindings (constraint #3)

`src/styles/tokens.css` confirmed: `--color-accent` (#8fb6dc dark / #2b5c84 light),
`--color-text` (#dde5ed / #1b2530), `--color-error` (#e9928a / #b3261e). The
`readPalette()` pattern in `lib/observatory/flowBench.js` (cssColor + parseColor +
blueLean + desaturate) is reused for prompt / generation / error inks; any additional
tones (KV column, idle node base) are derived from these at read time, never hardcoded.

STOP-condition check: none triggered — `store.drainEvents()` and every listed store
field exist as described.
