/* Neural Field renderer — draws the model-as-geometry scene lit by the
   choreography state. Standard module interface (`onEvent(evt, frame)`,
   `draw(ctx, frame)`) so it composes like the InferenceCanvas modules.

   Truthfulness: every lit element traces to a telemetry event or a store
   field (see design-evidence/neural-field/TRUTHFULNESS.md). Idle = the
   network at rest; the only standing motion is the explicitly-idle camera
   drift. Palette (constraint #3): all colors derive from tokens.css custom
   properties via readPalette(); copper/amber never appear here.

   Rendering technique: "glow" is a wide low-alpha underlay stroke/fill under
   the crisp element. ctx.shadowBlur is never used (frame-budget contract). */

import { makeCamera, project, depthAlpha, orbitStep, sortByDepth } from './projection'
import { buildScene, inboundPoint, railPoint } from './scene'
import {
  createChoreography,
  onEvent as choreographyEvent,
  step as choreographyStep,
  currentFronts,
  discEnergy,
  samplerCollapse,
} from './choreography'
import { readPalette } from '../flowBench'

const BASE_ALPHA_UNAVAILABLE = 0.04
const BASE_ALPHA_IDLE = 0.07
const BASE_ALPHA_AWAKE = 0.16

/* Precompute `rgba(r,g,b,` prefixes so per-frame color strings are one
   concat, not four number formats. */
function inkPrefix([r, g, b]) {
  return `rgba(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)},`
}

function mix(a, b, t) {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

export function createNeuralFieldRenderer({ reducedMotion = false } = {}) {
  const camera = makeCamera({})
  const state = createChoreography()
  let scene = buildScene(null)
  let palette = readPalette()
  let ink = null
  let reduced = reducedMotion

  function refreshPalette() {
    palette = readPalette()
    ink = {
      gen: inkPrefix(palette.generation),
      prompt: inkPrefix(palette.prompt),
      error: inkPrefix(palette.error),
      // Derived tones (never hardcoded hex): KV column leans prompt-blue,
      // wash mixes generation ink toward the desaturated error tone.
      kv: inkPrefix(mix(palette.prompt, palette.generation, 0.35)),
    }
  }
  refreshPalette()

  function ensureScene(layersTotal) {
    const wanted = Number.isInteger(layersTotal) && layersTotal > 1 ? layersTotal : null
    if (wanted && (scene.placeholder || scene.layersTotal !== wanted)) {
      scene = buildScene(wanted)
    }
  }

  function onEvent(evt, frame) {
    ensureScene(frame.run.layersTotal)
    choreographyEvent(state, evt, {
      t: frame.t,
      run: frame.run,
      layersTotal: scene.layersTotal,
      reducedMotion: reduced,
    })
  }

  function discTint(d, fronts, pulse) {
    // Per-disc energy from the traversal fronts, layer flashes, prefill glow.
    let energy = 0
    for (const front of fronts) energy = Math.max(energy, discEnergy(front, d))
    const flash = state.flashes.get(d) || 0
    energy = Math.max(energy, flash)
    if (state.prefill.glow > 0 && state.prefill.fill > 0) {
      const frac = scene.layersTotal > 1 ? d / (scene.layersTotal - 1) : 0
      if (frac <= state.prefill.fill) energy = Math.max(energy, 0.5 * state.prefill.glow)
    }
    if (reduced) energy = Math.max(energy, state.stepEnergy)
    // Error wash radiates outward from the disc that was active at failure.
    const wash = state.errorWash * Math.max(0, 1 - Math.abs(d - state.errorDisc) / 6)
    return { energy: Math.min(energy + pulse, 1), wash }
  }

  function draw(ctx, frame) {
    const { w, h, t, dt, run, connection } = frame
    ensureScene(run.layersTotal)

    // A run whose events stopped without inference_finished (killed backend)
    // goes stale via the store's RUN_STALE_MS path; the field settles.
    if (frame.runStale && state.awake) state.awake = false

    choreographyStep(state, dt, t, scene.layersTotal)
    orbitStep(camera, dt, state.awake, reduced)

    const unavailable = connection === 'unavailable'
    const baseAlpha = unavailable
      ? BASE_ALPHA_UNAVAILABLE
      : BASE_ALPHA_IDLE + (BASE_ALPHA_AWAKE - BASE_ALPHA_IDLE) * state.wake

    const fronts = currentFronts(state, t, scene.layersTotal)
    let pulse = 0
    if (state.finishedPulse) {
      const u = Math.min((t - state.finishedPulse.startT) / 900, 1)
      pulse = Math.sin(Math.PI * u) * 0.2 // one coherent exhale, then rest
    }

    const tints = scene.discs.map((disc) => discTint(disc.index, fronts, pulse))

    // ---- project + depth-sort the drawables (painter's algorithm) ----
    const projected = scene.nodes.map((node) => project(node, camera, w, h))
    const drawables = []
    for (let i = 0; i < scene.nodes.length; i += 1) {
      const p = projected[i]
      if (p) drawables.push({ kind: 'node', node: scene.nodes[i], p, depth: p.depth })
    }
    for (const edge of scene.edges) {
      const a = projected[edge.a.disc * scene.discs[0].nodes.length + edge.a.k]
      const b = projected[edge.b.disc * scene.discs[0].nodes.length + edge.b.k]
      if (a && b) drawables.push({ kind: 'edge', edge, a, b, depth: (a.depth + b.depth) / 2 })
    }
    sortByDepth(drawables)

    ctx.save()
    ctx.lineCap = 'round'
    for (const item of drawables) {
      if (item.kind === 'edge') {
        const d = item.edge.disc
        const glow = unavailable ? 0 : state.edgeGlow[d] || 0
        const tint = tints[d]
        const fade = depthAlpha(item.a)
        const structural = baseAlpha * 0.55 * fade
        const color = tint.wash > 0.05 ? ink.error : ink.gen
        if (glow > 0.03 && !reduced) {
          // Faux glow: wide low-alpha underlay beneath the crisp stroke.
          ctx.strokeStyle = color + (glow * 0.28 * fade).toFixed(3) + ')'
          ctx.lineWidth = 5.5 * item.a.scale
          ctx.beginPath()
          ctx.moveTo(item.a.x, item.a.y)
          ctx.lineTo(item.b.x, item.b.y)
          ctx.stroke()
        }
        ctx.strokeStyle = color + Math.min(structural + glow * 0.7 * fade, 0.9).toFixed(3) + ')'
        ctx.lineWidth = Math.max(0.5, 0.7 * item.a.scale)
        ctx.beginPath()
        ctx.moveTo(item.a.x, item.a.y)
        ctx.lineTo(item.b.x, item.b.y)
        ctx.stroke()
      } else {
        const { node, p } = item
        const tint = tints[node.disc]
        // Front-half nodes of each disc render brighter (depth cue contract).
        const facing = node.frontness > 0 ? 1 : 0.65
        const alpha = Math.min((baseAlpha + tint.energy * 0.8) * depthAlpha(p) * facing, 1)
        const color = tint.wash > 0.05 ? ink.error : ink.gen
        const radius = (1.15 + tint.energy * 1.7) * p.scale
        if (tint.energy > 0.25 && !reduced) {
          ctx.fillStyle = color + (tint.energy * 0.14 * depthAlpha(p)).toFixed(3) + ')'
          ctx.beginPath()
          ctx.arc(p.x, p.y, radius * 3.2, 0, Math.PI * 2)
          ctx.fill()
        }
        ctx.fillStyle = color + alpha.toFixed(3) + ')'
        ctx.beginPath()
        ctx.arc(p.x, p.y, radius, 0, Math.PI * 2)
        ctx.fill()
      }
    }
    ctx.restore()

    drawInputPlane(ctx, frame, baseAlpha)
    drawOutputRail(ctx, frame, baseAlpha)
    drawSampler(ctx, frame, baseAlpha)
    drawKvColumn(ctx, frame, baseAlpha, run)
    drawOverlayText(ctx, frame, unavailable)
  }

  function drawInputPlane(ctx, frame, baseAlpha) {
    const { w, h } = frame
    const plane = scene.inputPlane
    const c = project(plane, camera, w, h)
    if (!c) return
    const hw = plane.halfW * c.scale
    const hh = plane.halfH * c.scale
    const fade = depthAlpha(c)
    // Staging area frame.
    ctx.strokeStyle = ink.prompt + (baseAlpha * 2.2 * fade).toFixed(3) + ')'
    ctx.lineWidth = 1
    ctx.strokeRect(c.x - hw, c.y - hh, hw * 2, hh * 2)
    // Fill fraction = tokens_done / tokens_total, plus the over-cap carry
    // brightness for prompts larger than the 96-mote spawn budget.
    const fill = Math.min(state.prefill.fill + state.prefill.carry * 0.3, 1)
    if (fill > 0 && state.prefill.glow > 0) {
      ctx.fillStyle = ink.prompt + (0.2 * state.prefill.glow * fade).toFixed(3) + ')'
      ctx.fillRect(c.x - hw, c.y + hh - hh * 2 * fill, hw * 2, hh * 2 * fill)
    }
    // Inbound prompt motes (real prompt tokens in flight, capped upstream).
    for (const mote of state.inbound) {
      if (mote.s < 0) continue
      const p = project(inboundPoint(scene, mote.s, mote.ox, mote.oy), camera, frame.w, frame.h)
      if (!p) continue
      const a = 0.55 * Math.sin(Math.PI * Math.min(mote.s, 1)) * depthAlpha(p)
      ctx.fillStyle = ink.prompt + a.toFixed(3) + ')'
      ctx.beginPath()
      ctx.arc(p.x, p.y, 1.6 * p.scale, 0, Math.PI * 2)
      ctx.fill()
    }
  }

  function drawOutputRail(ctx, frame, baseAlpha) {
    const { w, h } = frame
    const a = project(scene.outputRail.from, camera, w, h)
    const b = project(scene.outputRail.to, camera, w, h)
    if (!a || !b) return
    ctx.strokeStyle = ink.gen + (baseAlpha * 1.6 * depthAlpha(b)).toFixed(3) + ')'
    ctx.lineWidth = 1
    ctx.setLineDash([3, 5])
    ctx.beginPath()
    ctx.moveTo(a.x, a.y)
    ctx.lineTo(b.x, b.y)
    ctx.stroke()
    ctx.setLineDash([])
    // Outbound generation motes: one per really-decoded token.
    for (const mote of state.outbound) {
      if (mote.delayMs > 0) continue
      const p = project(railPoint(scene, mote.s), camera, w, h)
      if (!p) continue
      const alpha = 0.85 * (1 - mote.s * 0.4) * depthAlpha(p)
      ctx.fillStyle = ink.gen + (alpha * 0.18).toFixed(3) + ')'
      ctx.beginPath()
      ctx.arc(p.x, p.y, 5.5 * p.scale, 0, Math.PI * 2)
      ctx.fill()
      ctx.fillStyle = ink.gen + alpha.toFixed(3) + ')'
      ctx.beginPath()
      ctx.arc(p.x, p.y, 2 * p.scale, 0, Math.PI * 2)
      ctx.fill()
    }
  }

  function drawSampler(ctx, frame, baseAlpha) {
    const { w, h, t } = frame
    const c = project(scene.samplerPoint, camera, w, h)
    if (!c) return
    const fade = depthAlpha(c)
    // The bloom point itself.
    ctx.fillStyle = ink.gen + (baseAlpha * 2.4 * fade).toFixed(3) + ')'
    ctx.beginPath()
    ctx.arc(c.x, c.y, 2.4 * c.scale, 0, Math.PI * 2)
    ctx.fill()

    const sampler = state.sampler
    if (sampler && sampler.candidates.length && state.wake > 0.01) {
      const collapse = samplerCollapse(sampler, t)
      const n = sampler.candidates.length
      for (let i = 0; i < n; i += 1) {
        const cand = sampler.candidates[i]
        const chosen = cand.token_id === sampler.chosenTokenId
        // Length ∝ candidate weight ORDERING (rank), not raw probability.
        const rankLen = (34 - (28 * i) / Math.max(n - 1, 1)) * c.scale
        // Chosen spoke absorbs the others as the collapse completes.
        const len = chosen ? rankLen * (1 + collapse * 0.35) : rankLen * (1 - collapse)
        if (len < 0.5) continue
        const angle = (i / n) * Math.PI * 2 - Math.PI / 2
        const alpha = (chosen ? 0.9 : 0.5) * fade * Math.max(state.wake, 0.35)
        ctx.strokeStyle = ink.gen + alpha.toFixed(3) + ')'
        ctx.lineWidth = chosen ? 1.8 : 1
        ctx.beginPath()
        ctx.moveTo(c.x, c.y)
        ctx.lineTo(c.x + Math.cos(angle) * len, c.y + Math.sin(angle) * len)
        ctx.stroke()
      }
    }

    // Sealed-receipt burst: generation ink at high brightness (never amber).
    for (const p of state.receiptBurst) {
      const px = c.x + Math.cos(p.angle) * p.r * c.scale
      const py = c.y + Math.sin(p.angle) * p.r * c.scale
      ctx.fillStyle = ink.gen + (0.9 * p.life * fade).toFixed(3) + ')'
      ctx.beginPath()
      ctx.arc(px, py, 1.6 * c.scale, 0, Math.PI * 2)
      ctx.fill()
    }
  }

  function drawKvColumn(ctx, frame, baseAlpha, run) {
    const { w, h } = frame
    const col = scene.kvColumn
    const bottom = project({ x: col.x, y: col.yBottom, z: col.zBase }, camera, w, h)
    const top = project({ x: col.x, y: col.yBottom + col.height, z: col.zBase }, camera, w, h)
    if (!bottom || !top) return
    const fade = depthAlpha(bottom)
    const halfW = col.halfW * bottom.scale
    ctx.strokeStyle = ink.kv + (baseAlpha * 2 * fade).toFixed(3) + ')'
    ctx.lineWidth = 1
    ctx.strokeRect(Math.min(bottom.x, top.x) - halfW, top.y, halfW * 2, bottom.y - top.y)
    // Filled height = kv.position / kv.capacity — real store values only.
    const frac = run.kv.capacity > 0 ? Math.min(run.kv.position / run.kv.capacity, 1) : 0
    if (frac > 0) {
      const fillTop = bottom.y - (bottom.y - top.y) * frac
      ctx.fillStyle = ink.kv + (0.28 * fade).toFixed(3) + ')'
      ctx.fillRect(Math.min(bottom.x, top.x) - halfW + 1, fillTop, halfW * 2 - 2, bottom.y - fillTop)
      if (state.kvPulse > 0) {
        // Brief single-segment brightening at the new top.
        ctx.fillStyle = ink.kv + (0.6 * state.kvPulse * fade).toFixed(3) + ')'
        ctx.fillRect(Math.min(bottom.x, top.x) - halfW + 1, fillTop, halfW * 2 - 2, 3)
      }
      if (run.kv.approxBytes > 0) {
        ctx.fillStyle = ink.kv + (0.5 * fade).toFixed(3) + ')'
        ctx.font = '10px "IBM Plex Mono", monospace'
        ctx.textAlign = 'center'
        ctx.fillText(`${(run.kv.approxBytes / 1048576).toFixed(1)} MiB`, bottom.x, bottom.y + 14)
        ctx.fillText(`${run.kv.position}/${run.kv.capacity}`, bottom.x, bottom.y + 26)
      }
    }
  }

  function drawOverlayText(ctx, frame, unavailable) {
    ctx.font = '11px "IBM Plex Mono", monospace'
    ctx.textAlign = 'left'
    if (unavailable) {
      ctx.fillStyle = ink.gen + '0.4)'
      ctx.fillText('telemetry stream unavailable — nothing to show', 14, 20)
      return
    }
    if (state.prefill.path && state.wake > 0.01) {
      ctx.fillStyle = ink.prompt + (0.55 * state.wake).toFixed(3) + ')'
      ctx.fillText(`prefill path: ${state.prefill.path}`, 14, 20)
    }
    if (frame.run.active && !frame.run.layerEventsSeen && state.sweeps.length) {
      // GPU-lane honesty: the sweep is token-paced, never per-layer timing.
      ctx.fillStyle = ink.gen + '0.4)'
      ctx.fillText('token-paced sweep (per-layer timing not observable on this lane)', 14, 36)
    }
  }

  return {
    onEvent,
    draw,
    refreshPalette,
    setReducedMotion(value) {
      reduced = Boolean(value)
    },
    // Test/debug seam: current choreography state (read-only use).
    _state: state,
  }
}
