/* Neural Field choreography — telemetry events → light state.

   This is the state machine behind the renderer: every field here is set by
   a real `camelid.telemetry/v1` event (via onEvent) or decays toward rest
   (via step). There is no random event generator and no simulated inference;
   an idle store leaves this state at rest and the network dark.

   GPU-lane honesty (constraint #2): `layer_started`/`layer_completed` exist
   only on CPU lanes. When `token_decoded` arrives without layer events, the
   sweep front traverses the stack over the token's measured interval
   (`run.decode.tokenIntervalMs`, clamped 120–900ms) — the token really
   crossed every layer; the sweep only distributes it in time. It is never
   presented as measured per-layer timing. */

export const MOTE_CAP = 96
export const SWEEP_HALF_WIDTH = 4.5
export const EDGE_DECAY_TAU_MS = 280
export const EDGE_FIRE_THRESHOLD = 0.3
const WAKE_RAMP_MS = 400
const SETTLE_MS = 900
const SWEEP_FALLBACK_MS = 320
const SAMPLER_COLLAPSE_MS = 150
const RAIL_MOTE_MS = 520
const INBOUND_MOTE_MS = 780

function clamp(v, lo, hi) {
  return Math.min(Math.max(v, lo), hi)
}

/* Falloff-front energy at disc d for a front centered on `front`
   (contract: energy(d) = max(0, 1 − |d − front| / 4.5)²). */
export function discEnergy(front, d) {
  const e = Math.max(0, 1 - Math.abs(d - front) / SWEEP_HALF_WIDTH)
  return e * e
}

export function createChoreography() {
  return {
    // 0..1 wakefulness: ramps up over 400ms on inference_started, settles
    // over 900ms on inference_finished. Drives node base alpha 0.07 → 0.16.
    wake: 0,
    awake: false,
    // Error state: while latched, edges refuse to fire (until finished).
    errorLatched: false,
    errorWash: 0,
    errorDisc: 0,
    // Finished-ok exhale pulse: { startT } or null. Error/disconnected
    // finishes settle without the pulse.
    finishedPulse: null,
    prefill: { glow: 0, fill: 0, path: null, carry: 0 },
    inbound: [], // prompt motes { s, ox, oy, durMs }
    outbound: [], // generation motes { s, delayMs }
    // GPU token sweeps [{ startT, durationMs }] — one front per decoded
    // token, each completing its own 0→N traversal. At 10+ tok/s several
    // fronts overlap and the tunnel pulses continuously (signature motion).
    sweeps: [],
    cpuFront: null, // CPU lanes: disc index of the last layer_started
    flashes: new Map(), // disc -> flash intensity (layer_completed)
    edgeGlow: [], // per disc-gap firing brightness, τ ≈ 280ms
    kvPulse: 0, // brief top-segment brightening on kv_cache_updated
    sampler: null, // { mode, candidates, chosenTokenId, startT }
    receiptBurst: [], // { angle, r, vr, life } spokes at the sampler point
    // Reduced motion: discrete opacity steps instead of continuous motion.
    stepEnergy: 0,
  }
}

/* ctx: { t, run, layersTotal, reducedMotion }. `run` is the telemetry store's
   run snapshot; layersTotal is the scene's disc count. */
export function onEvent(state, evt, ctx) {
  const { t, run, layersTotal, reducedMotion } = ctx
  switch (evt.event) {
    case 'inference_started':
      state.awake = true
      state.errorLatched = false
      state.errorWash = 0
      state.finishedPulse = null
      state.prefill = { glow: 0, fill: 0, path: null, carry: 0 }
      state.sampler = null
      state.cpuFront = null
      state.sweeps = []
      break
    case 'prefill_started': {
      state.prefill.glow = 1
      state.prefill.path = evt.path || null
      state.prefill.fill = 0
      const tokens = evt.prefill_tokens || 0
      if (reducedMotion) {
        state.prefill.carry = clamp(tokens / 256, 0.2, 1)
        state.stepEnergy = 1
      } else {
        const spawn = Math.min(tokens, MOTE_CAP)
        // Cap at 96 motes; the remainder is carried as input-plane brightness.
        state.prefill.carry = clamp((tokens - spawn) / 512, 0, 1)
        for (let i = 0; i < spawn; i += 1) {
          state.inbound.push({
            s: -Math.random() * 0.6, // stagger departures along the approach
            ox: (Math.random() - 0.5) * 110,
            oy: (Math.random() - 0.5) * 70,
            durMs: INBOUND_MOTE_MS * (0.75 + Math.random() * 0.5),
          })
        }
      }
      break
    }
    case 'prefill_progress': {
      const total = evt.tokens_total || run.prefill.tokens || 1
      state.prefill.fill = clamp((evt.tokens_done || 0) / total, 0, 1)
      break
    }
    case 'decode_started':
      state.prefill.glow = 0 // glow releases; decode begins at context_position
      break
    case 'layer_started':
      if (typeof evt.layer === 'number') {
        state.cpuFront = evt.layer
        if (reducedMotion) state.stepEnergy = 1
      }
      break
    case 'layer_completed':
      if (typeof evt.layer === 'number') {
        state.flashes.set(evt.layer, 1)
        // The layer-event throttle (15ms shared across started/completed)
        // drops most starts on fast CPU decode; a completed event is an
        // equally real "activity at layer N" report, so it also moves the
        // front. Without this the tunnel goes dark between rare starts.
        state.cpuFront = evt.layer
        if (reducedMotion) state.stepEnergy = 1
      }
      break
    case 'token_decoded': {
      if (reducedMotion) {
        state.stepEnergy = 1
        break
      }
      if (!run.layerEventsSeen) {
        // GPU sweep paced by the really-measured token interval. Each token
        // gets its own front so fast decode overlaps rather than restarts.
        const durationMs = clamp(run.decode.tokenIntervalMs || SWEEP_FALLBACK_MS, 120, 900)
        if (state.sweeps.length < 12) state.sweeps.push({ startT: t, durationMs })
        // The outbound mote departs when the front reaches the output rail.
        state.outbound.push({ s: 0, delayMs: durationMs })
      } else {
        // Layer events already showed traversal; outbound mote only.
        state.outbound.push({ s: 0, delayMs: 0 })
      }
      break
    }
    case 'kv_cache_updated':
      state.kvPulse = 1
      break
    case 'sampler_step': {
      const candidates = Array.isArray(evt.candidates) ? evt.candidates.slice(0, 8) : []
      candidates.sort((a, b) => b.prob - a.prob)
      state.sampler = {
        mode: evt.mode === 'sampling' ? 'sampling' : 'greedy',
        candidates,
        chosenTokenId: evt.chosen_token_id,
        startT: t,
      }
      break
    }
    case 'inference_error': {
      state.errorLatched = true
      state.errorWash = 1
      const fronts = currentFronts(state, t, layersTotal)
      state.errorDisc = fronts.length ? fronts[fronts.length - 1] : 0
      break
    }
    case 'inference_finished':
      state.awake = false
      state.errorLatched = false
      state.prefill.glow = 0
      if (evt.status === 'ok') state.finishedPulse = { startT: t }
      break
    case 'receipt_written': {
      // Sealed-receipt burst at the sampler point (tokenParticles concept),
      // in generation ink at high brightness — copper/amber never here.
      const spokes = reducedMotion ? 0 : 26
      for (let i = 0; i < spokes; i += 1) {
        state.receiptBurst.push({
          angle: (i / 26) * Math.PI * 2,
          r: 2,
          vr: 26 + Math.random() * 22, // world units / s
          life: 1,
        })
      }
      if (reducedMotion) state.stepEnergy = 1
      break
    }
    default:
      break
  }
}

/* Current sweep-front positions in disc space (possibly several — one per
   in-flight token on GPU lanes; a single one on CPU lanes from the last
   reported layer). Empty array when nothing is traversing. */
export function currentFronts(state, t, layersTotal) {
  if (state.cpuFront != null) return [state.cpuFront]
  const fronts = []
  for (const sweep of state.sweeps) {
    const u = (t - sweep.startT) / sweep.durationMs
    if (u >= 0 && u <= 1) fronts.push(u * (layersTotal - 1))
  }
  return fronts
}

/* Advance decays and motes one frame. Also fires edges from the current
   front (unless the error latch is holding them shut). */
export function step(state, dtMs, t, layersTotal) {
  const dt = clamp(dtMs, 0, 100)

  state.wake = state.awake
    ? Math.min(1, state.wake + dt / WAKE_RAMP_MS)
    : Math.max(0, state.wake - dt / SETTLE_MS)

  for (let i = state.sweeps.length - 1; i >= 0; i -= 1) {
    if (t - state.sweeps[i].startT > state.sweeps[i].durationMs) state.sweeps.splice(i, 1)
  }
  if (state.finishedPulse && t - state.finishedPulse.startT > SETTLE_MS) state.finishedPulse = null

  // Edge firing: any disc-gap whose source disc holds energy > 0.3 fires at
  // brightness ∝ energy, then decays with τ ≈ 280ms.
  if (state.edgeGlow.length !== Math.max(layersTotal - 1, 0)) {
    state.edgeGlow = new Array(Math.max(layersTotal - 1, 0)).fill(0)
  }
  const fronts = currentFronts(state, t, layersTotal)
  const decay = Math.exp(-dt / EDGE_DECAY_TAU_MS)
  for (let d = 0; d < state.edgeGlow.length; d += 1) {
    let glow = state.edgeGlow[d] * decay
    if (!state.errorLatched) {
      for (const front of fronts) {
        const energy = discEnergy(front, d)
        if (energy > EDGE_FIRE_THRESHOLD) glow = Math.max(glow, energy)
      }
    }
    state.edgeGlow[d] = glow
  }

  for (const [disc, intensity] of state.flashes) {
    const next = intensity - dt / 600
    if (next <= 0) state.flashes.delete(disc)
    else state.flashes.set(disc, next)
  }

  for (let i = state.inbound.length - 1; i >= 0; i -= 1) {
    const mote = state.inbound[i]
    mote.s += dt / mote.durMs
    if (mote.s >= 1) state.inbound.splice(i, 1)
  }
  for (let i = state.outbound.length - 1; i >= 0; i -= 1) {
    const mote = state.outbound[i]
    if (mote.delayMs > 0) {
      mote.delayMs -= dt
    } else {
      mote.s += dt / RAIL_MOTE_MS
      if (mote.s >= 1) state.outbound.splice(i, 1)
    }
  }
  for (let i = state.receiptBurst.length - 1; i >= 0; i -= 1) {
    const p = state.receiptBurst[i]
    p.r += p.vr * (dt / 1000)
    p.life -= dt / 950
    if (p.life <= 0) state.receiptBurst.splice(i, 1)
  }

  state.kvPulse = Math.max(0, state.kvPulse - dt / 700)
  state.prefill.glow = Math.max(0, state.prefill.glow - dt / 4000)
  if (!state.errorLatched) state.errorWash = Math.max(0, state.errorWash - dt / 1200)
  // Reduced motion: quantized decay so the change reads as discrete steps.
  if (state.stepEnergy > 0) {
    state.stepEnergy = Math.max(0, Math.round((state.stepEnergy - dt / 900) * 4) / 4)
  }
}

/* Sampler collapse progress at time t: 0 = spokes fully spread, 1 = absorbed
   into the chosen spoke. Greedy collapses instantly; sampling over 150ms. */
export function samplerCollapse(sampler, t) {
  if (!sampler) return 1
  if (sampler.mode === 'greedy') return 1
  return clamp((t - sampler.startT) / SAMPLER_COLLAPSE_MS, 0, 1)
}
