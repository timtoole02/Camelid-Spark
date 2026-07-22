/* InferenceTelemetryStore — the single source of truth for the Inference
   Observatory. It subscribes to the backend's live SSE stream
   (`GET /api/telemetry/stream`) and reduces real runtime events into a
   renderable snapshot.

   Truthfulness contract (mirrors src/telemetry.rs on the backend): this store
   never synthesizes, replays, or simulates events. If the stream is not
   connected the state is `unavailable`; if connected but no inference is
   running the state is `idle`. The canvas renders from this store only. */

const MAX_ERRORS = 20
const MAX_RECENT_EVENTS = 4096
/* If no run-scoped event arrives for this long after the last one, and no
   inference_finished was seen (e.g. the server died mid-run), the run is
   marked stale so the UI does not show inference as still happening. */
const RUN_STALE_MS = 30000

export const CONNECTION = {
  CONNECTING: 'connecting',
  LIVE: 'live',
  UNAVAILABLE: 'unavailable',
}

function emptyRun() {
  return {
    active: false,
    requestId: null,
    modelId: null,
    backend: null,
    quantization: null,
    architecture: null,
    promptTokens: 0,
    maxTokens: 0,
    contextLength: 0,
    temperature: null,
    stream: null,
    startedAtMs: null,
    joinedMidRun: false,
    phase: 'idle', // idle | prefill | decode | finished | error
    prefill: { tokens: 0, done: 0, path: null, startedAtMs: null, endedAtMs: null },
    decode: { startedAtMs: null, tokens: 0, lastTokenAtMs: null, tokenIntervalMs: null },
    layersTotal: null,
    activeLayer: null,
    layerEventsSeen: false,
    kv: { position: 0, capacity: 0, approxBytes: 0 },
    lastSampler: null, // { chosenTokenId, mode, candidates }
    generatedTokenIds: [],
    finish: null, // { status, finishReason, completionTokens, totalMs, ttftMs, decodeTps, prefillTps, error }
    receipt: null, // { receiptId, reproducible, ggufSha256, atMs }
    errors: [],
  }
}

export function createInferenceTelemetryStore() {
  let eventSource = null
  let connection = CONNECTION.CONNECTING
  let connectedBase = null
  let lastEventAtMs = null
  let run = emptyRun()
  let lastRun = null // most recent finished run, kept for the details panel
  const workers = new Map() // node -> { status: 'active'|'idle'|'error', detail, error, lastSeenMs }
  const pending = [] // raw events drained by the canvas each frame
  const listeners = new Set()

  function notify() {
    listeners.forEach((listener) => listener())
  }

  function pushPending(event) {
    pending.push(event)
    if (pending.length > MAX_RECENT_EVENTS) pending.splice(0, pending.length - MAX_RECENT_EVENTS)
  }

  function prefillTps(r) {
    if (!r.prefill.startedAtMs || !r.prefill.endedAtMs || !r.prefill.tokens) return null
    const ms = r.prefill.endedAtMs - r.prefill.startedAtMs
    return ms > 0 ? (r.prefill.tokens * 1000) / ms : null
  }

  function decodeTps(r) {
    if (!r.decode.startedAtMs || r.decode.tokens < 2 || !r.decode.lastTokenAtMs) return null
    const ms = r.decode.lastTokenAtMs - r.decode.startedAtMs
    return ms > 0 ? ((r.decode.tokens - 1) * 1000) / ms : null
  }

  const RUN_SCOPED_EVENTS = new Set([
    'prefill_started', 'prefill_progress', 'decode_started', 'layer_started',
    'layer_completed', 'token_decoded', 'kv_cache_updated', 'sampler_step',
  ])

  /* Joining mid-run: if run-scoped events arrive without a preceding
     inference_started (the tab was opened while a generation was already in
     flight), the run is activated from the event's own attribution. These
     events are real inference activity; identity fields not yet observed
     stay null rather than being guessed. */
  function joinRunInProgress(evt, now) {
    if (!RUN_SCOPED_EVENTS.has(evt.event) || run.active) return false
    const sameRun = Boolean(evt.request_id) && run.requestId === evt.request_id
    if (sameRun && run.finish) return false // trailing events of a closed run
    if (!sameRun && run.finish) {
      lastRun = run
      run = emptyRun()
    }
    run.active = true
    run.phase = 'running'
    run.startedAtMs = run.startedAtMs || now
    run.requestId = evt.request_id || run.requestId
    run.modelId = evt.model_id || run.modelId
    // The start was not observed, so durations measured from it (TTFT,
    // wall-clock totals) must not be claimed for this run.
    run.joinedMidRun = true
    return true
  }

  function applyEvent(evt) {
    const now = performance.now()
    lastEventAtMs = now
    const joined = joinRunInProgress(evt, now)
    const changed = (() => {
    switch (evt.event) {
      case 'hello':
        return false
      case 'lagged':
        // The stream dropped events under load; surface it, never paper over it.
        run.errors.push({ code: 'telemetry_lagged', message: `Telemetry stream skipped ${evt.skipped} event(s) under load.`, atMs: now })
        return true
      case 'inference_started':
        run = emptyRun()
        run.active = true
        run.phase = 'running'
        run.requestId = evt.request_id || null
        run.modelId = evt.model_id || null
        run.backend = evt.backend || null
        run.quantization = evt.quantization || null
        run.architecture = evt.architecture || null
        run.promptTokens = evt.prompt_tokens || 0
        run.maxTokens = evt.max_tokens || 0
        run.contextLength = evt.context_length || 0
        run.temperature = typeof evt.temperature === 'number' ? evt.temperature : null
        run.stream = Boolean(evt.stream)
        run.startedAtMs = now
        return true
      case 'prefill_started':
        run.phase = 'prefill'
        run.prefill = { tokens: evt.prefill_tokens || 0, done: 0, path: evt.path || null, startedAtMs: now, endedAtMs: null }
        if (evt.layers_total) run.layersTotal = evt.layers_total
        return true
      case 'prefill_progress':
        run.prefill.done = evt.tokens_done || 0
        if (!run.prefill.tokens) run.prefill.tokens = evt.tokens_total || 0
        return false
      case 'decode_started':
        run.phase = 'decode'
        run.prefill.endedAtMs = run.prefill.endedAtMs || now
        run.decode.startedAtMs = run.decode.startedAtMs || now
        if (typeof evt.context_position === 'number') run.kv.position = evt.context_position
        return true
      case 'layer_started':
      case 'layer_completed':
        run.layerEventsSeen = true
        run.activeLayer = evt.layer
        if (evt.layers_total) run.layersTotal = evt.layers_total
        return false
      case 'token_decoded': {
        run.phase = 'decode'
        run.decode.startedAtMs = run.decode.startedAtMs || now
        if (run.decode.lastTokenAtMs != null) {
          run.decode.tokenIntervalMs = now - run.decode.lastTokenAtMs
        }
        run.decode.lastTokenAtMs = now
        run.decode.tokens += 1
        if (typeof evt.token_id === 'number') run.generatedTokenIds.push(evt.token_id)
        if (typeof evt.context_position === 'number') run.kv.position = evt.context_position
        if (evt.layers_total) run.layersTotal = evt.layers_total
        return false
      }
      case 'kv_cache_updated':
        run.kv = {
          position: evt.position || 0,
          capacity: evt.capacity || 0,
          approxBytes: evt.approx_bytes || 0,
        }
        return false
      case 'sampler_step':
        run.lastSampler = {
          chosenTokenId: evt.chosen_token_id,
          mode: evt.mode,
          candidates: Array.isArray(evt.candidates) ? evt.candidates : [],
        }
        return false
      case 'inference_error':
        run.errors.push({ code: evt.code || 'error', message: evt.message || 'Unknown inference error', atMs: now })
        if (run.errors.length > MAX_ERRORS) run.errors.splice(0, run.errors.length - MAX_ERRORS)
        run.phase = 'error'
        return true
      case 'inference_finished':
        run.active = false
        run.phase = evt.status === 'ok' ? 'finished' : 'error'
        run.finish = {
          status: evt.status,
          finishReason: evt.finish_reason || null,
          completionTokens: evt.completion_tokens || 0,
          totalMs: evt.total_ms || null,
          ttftMs: typeof evt.ttft_ms === 'number' ? evt.ttft_ms : null,
          decodeTps: typeof evt.decode_tps === 'number' ? evt.decode_tps : decodeTps(run),
          prefillTps: typeof evt.prefill_tps === 'number' ? evt.prefill_tps : prefillTps(run),
          error: evt.error || null,
        }
        if (evt.error) {
          run.errors.push({ code: evt.status, message: evt.error, atMs: now })
        }
        lastRun = run
        return true
      case 'receipt_written':
        run.receipt = {
          receiptId: evt.receipt_id,
          reproducible: Boolean(evt.reproducible),
          ggufSha256: evt.gguf_sha256 || null,
          atMs: now,
        }
        if (lastRun && lastRun.requestId === run.requestId) lastRun.receipt = run.receipt
        return true
      case 'worker_node_active':
        workers.set(evt.node, { status: 'active', detail: evt.detail || null, error: null, lastSeenMs: now })
        return true
      case 'worker_node_idle':
        workers.set(evt.node, { ...(workers.get(evt.node) || {}), status: 'idle', error: null, lastSeenMs: now })
        return true
      case 'worker_node_error':
        workers.set(evt.node, { ...(workers.get(evt.node) || {}), status: 'error', error: evt.error || 'unknown error', lastSeenMs: now })
        run.errors.push({ code: 'worker_node_error', message: `${evt.node}: ${evt.error || 'unknown error'}`, atMs: now })
        return true
      default:
        return false
    }
    })()
    return joined || changed
  }

  /* Single ingestion point for backend events: the SSE transport and the
     test harness both land here. Nothing else feeds the store — there is no
     simulation path. */
  function ingest(evt) {
    pushPending({ ...evt, receivedAtMs: performance.now() })
    if (applyEvent(evt)) notify()
  }

  function handleMessage(raw) {
    let evt
    try {
      evt = JSON.parse(raw)
    } catch {
      return
    }
    ingest(evt)
  }

  function connect(apiBase) {
    const base = (apiBase || '').replace(/\/$/, '')
    if (eventSource && connectedBase === base) return
    disconnect()
    connectedBase = base
    connection = CONNECTION.CONNECTING
    notify()
    try {
      eventSource = new EventSource(`${base}/api/telemetry/stream`)
    } catch {
      connection = CONNECTION.UNAVAILABLE
      notify()
      return
    }
    eventSource.onopen = () => {
      connection = CONNECTION.LIVE
      notify()
    }
    eventSource.onerror = () => {
      // EventSource retries on its own; reflect the gap honestly meanwhile.
      connection = eventSource && eventSource.readyState === EventSource.CLOSED
        ? CONNECTION.UNAVAILABLE
        : CONNECTION.CONNECTING
      notify()
    }
    eventSource.addEventListener('telemetry', (event) => handleMessage(event.data))
  }

  function disconnect() {
    if (eventSource) {
      eventSource.close()
      eventSource = null
    }
    connectedBase = null
  }

  function isRunStale() {
    return run.active && lastEventAtMs != null && performance.now() - lastEventAtMs > RUN_STALE_MS
  }

  return {
    connect,
    disconnect,
    /* Test seam only: feeds the exact same reducer the SSE stream uses.
       Production code must never call this — the observatory renders real
       backend events exclusively. */
    ingest,
    subscribe(listener) {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
    /* Drain raw events accumulated since the last call (canvas, once per frame). */
    drainEvents() {
      if (!pending.length) return []
      return pending.splice(0, pending.length)
    },
    getConnection: () => connection,
    getRun: () => run,
    getLastRun: () => lastRun,
    getWorkers: () => workers,
    isRunStale,
    liveDecodeTps: () => decodeTps(run),
    livePrefillTps: () => prefillTps(run),
  }
}
