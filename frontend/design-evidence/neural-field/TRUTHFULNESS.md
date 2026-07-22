# Neural Field — truthfulness audit (Phase 5)

Every animated element in the Neural Field canvas, mapped to the real telemetry event or
store field that drives it. The renderer consumes `useInferenceTelemetry(apiBase)` +
`store.drainEvents()` (backend SSE store, `camelid.telemetry/v1`) — never the client-side
telemetryLog bus, never a synthetic event source. There is no random event generator and
no simulated inference anywhere in `lib/observatory/neuralField/`.

| Animated element | Driven by | Notes |
| --- | --- | --- |
| Network wake (node base alpha 0.07→0.16, 400ms) | `inference_started` | settles over 900ms on `inference_finished` |
| Camera vantage ease (pitch −0.05, ~800ms) | `inference_started` / `inference_finished` (via `state.awake`) | reduced motion: camera fixed |
| Idle camera drift + starfield | none — **explicitly-idle ambient treatment** | ≤0.02 rad/s; declared in header copy; static in reduced motion |
| Inbound prompt motes (≤96) + input-plane carry brightness | `prefill_started` (`prefill_tokens`, cap 96, remainder→brightness) | prompt-blue (`--color-accent`) |
| Prefill path label (verbatim, e.g. `auto`) | `prefill_started.path` | displayed exactly as the backend reported it |
| Input-plane fill fraction | `prefill_progress.tokens_done / tokens_total` | |
| Disc prefill glow up to fill fraction | `prefill_started` + `prefill_progress` (release on `decode_started`) | matches LayerVisualizer.prefillGlow semantics |
| CPU-lane front energy (falloff `(1−|d−front|/4.5)²`) | `layer_started.layer` AND `layer_completed.layer` | both events are real "activity at layer N" reports; the shared 15ms layer-event throttle drops most `layer_started`, so `layer_completed` also moves the front (receipt note below) |
| Disc flash | `layer_completed.layer` | `duration_us` is NOT rendered (no per-layer µs labels anywhere) |
| GPU-lane sweep fronts | `token_decoded` when no layer events seen; each front traverses over `run.decode.tokenIntervalMs` clamped 120–900ms | one front per real decoded token (≤12 concurrent); on-canvas note: "token-paced sweep (per-layer timing not observable on this lane)" |
| Edge firing (brightness ∝ energy, τ≈280ms decay) | derived from the front positions above | edges refuse to fire while `inference_error` is latched |
| Outbound generation motes on the output rail | `token_decoded` (one per token; GPU lanes delayed by the sweep duration so the mote departs when the front arrives) | generation ink |
| KV column fill height | `run.kv.position / run.kv.capacity` (store, from `kv_cache_updated` / `decode_started` / `token_decoded`) | |
| KV top-segment pulse | `kv_cache_updated` | |
| KV label (`MiB`, `position/capacity`) | `run.kv.approxBytes`, `run.kv.position`, `run.kv.capacity` | label layer only |
| Sampler spokes (length ∝ candidate **rank**) | `sampler_step.candidates` | token IDs and token text are never rendered |
| Spoke collapse (greedy instant / sampling 150ms) | `sampler_step.mode`, `chosen_token_id` | |
| Error wash from active disc + edge-fire refusal | `inference_error` | desaturated `--color-error`; released by `inference_finished` |
| Finished exhale pulse (900ms, once) | `inference_finished { status: "ok" }` | `error`/`disconnected` settle **without** the pulse |
| Receipt burst at sampler point | `receipt_written` | generation ink at high brightness — copper/amber never appear in this canvas |
| Unavailable dimming (0.04) + plain-text notice | `store.getConnection() === 'unavailable'` | |
| Reduced-motion discrete steps (`stepEnergy`) | same events (`prefill_started`, layer events, `token_decoded`, `receipt_written`) | no motes, no trails, no orbit; quantized decay |

## Killed-backend settle (RUN_STALE_MS path)

`NeuralField.jsx` passes `runStale: store.isRunStale()` into every frame;
`renderer.draw` clears `state.awake` when the store reports the run stale (no run-scoped
event for `RUN_STALE_MS` = 30s without `inference_finished`). The wake level then decays
to idle over the standard 900ms settle — the field cannot animate an inference forever
after a backend dies mid-run. Additionally, killing the backend drops the EventSource to
`unavailable`, which independently dims the field to 0.04 (captured: `frames/unavailable.png`).

## Evidence provenance

All frames in `frames/` were captured from real TinyLlama 1.1B Q8_0 generations against a
live backend (GPU-resident lane, and a `CUDA_VISIBLE_DEVICES=-1` CPU-lane pass for layer
events). The `error` frame is a REAL `inference_error`: `api_error()` in `src/api/mod.rs`
surfaces any API failure occurring while a generation is live onto the telemetry stream;
the harness fired `/api/models/inspect` with a nonexistent path mid-generation. The
receipt frame is a real sealed receipt (`camelid_receipt: true` request on the supported
lane). No frame was produced through the store's `ingest` test seam.

## Declared abstractions (also stated in the view header copy)

1. 18 nodes per disc is a stylized layer cross-section — attention-head counts are not in
   telemetry, so no per-head claim is made.
2. GPU-resident lanes show a token-paced sweep, not measured per-layer timing (constraint
   #2); the canvas says so while sweeping.
3. Node lane offsets for prompt motes and starfield placement use visual randomness for
   layout only — their existence and count are event-driven; their pixel positions carry
   no telemetry claim.

## Backend observations recorded for follow-up (no backend changes made)

- `prefill_started.path` arrives as `"auto"` on both lanes exercised here, not one of the
  four documented values (`gpu_resident|layer_major|chunked|single_token`) in
  `src/telemetry.rs`'s comment. The canvas displays the value verbatim.
- The shared layer-event throttle means `layer_started` is almost never seen on fast CPU
  decode (one per run in practice); `layer_completed` carries the observable layer cursor.
