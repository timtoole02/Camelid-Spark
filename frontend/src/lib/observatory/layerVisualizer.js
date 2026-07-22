/* LayerVisualizer — the model core: concentric glass rings, one per
   transformer layer. Rings energize from real layer events when the engine
   reports them (CPU lanes), or as a sweep paced by each really-decoded
   token's measured interval on GPU-resident lanes (where per-layer timing is
   not observable — the token genuinely traversed every layer, the sweep just
   distributes it across the rings). Idle = rings at rest. */

const SWEEP_FALLBACK_MS = 320

export class LayerVisualizer {
  constructor() {
    this.energies = []
    this.sweep = null // { startT, durationMs }
    this.prefillGlow = 0
    this.errorFlash = 0
  }

  ensureLayers(count) {
    while (this.energies.length < count) this.energies.push(0)
    if (this.energies.length > count) this.energies.length = count
  }

  onEvent(evt, frame) {
    const total = frame.run.layersTotal || this.energies.length || 0
    if (total) this.ensureLayers(total)
    if (evt.event === 'layer_started' || evt.event === 'layer_completed') {
      if (typeof evt.layer === 'number' && evt.layer < this.energies.length) {
        this.energies[evt.layer] = 1
      }
    }
    if (evt.event === 'token_decoded' && !frame.run.layerEventsSeen && this.energies.length) {
      const interval = frame.run.decode.tokenIntervalMs
      this.sweep = {
        startT: frame.t,
        durationMs: Math.min(Math.max(interval || SWEEP_FALLBACK_MS, 120), 900),
      }
    }
    if (evt.event === 'prefill_started') this.prefillGlow = 1
    if (evt.event === 'decode_started') this.prefillGlow = 0
    if (evt.event === 'inference_error' || evt.event === 'worker_node_error') this.errorFlash = 1
  }

  draw(ctx, frame) {
    const { cx, cy, R, t, dt, run, connection } = frame
    const total = this.energies.length
    const live = connection === 'live'
    const idleAlpha = live ? 0.16 : 0.07

    // Sweep wave position (GPU-resident lanes): 0..1 through the ring stack.
    let sweepPos = -1
    if (this.sweep) {
      const elapsed = t - this.sweep.startT
      if (elapsed <= this.sweep.durationMs) sweepPos = elapsed / this.sweep.durationMs
      else this.sweep = null
    }

    const ringCount = total || 12 // resting geometry before a model reports its depth
    const inner = R * 0.55
    const outer = R * 1.35
    ctx.save()
    for (let i = 0; i < ringCount; i += 1) {
      const frac = ringCount > 1 ? i / (ringCount - 1) : 0
      const radius = inner + (outer - inner) * frac
      let energy = total ? this.energies[i] : 0
      if (sweepPos >= 0) {
        const d = Math.abs(frac - sweepPos)
        energy = Math.max(energy, Math.max(0, 1 - d * 6))
      }
      if (this.prefillGlow > 0 && run.prefill.tokens > 0) {
        const progress = run.prefill.done / Math.max(run.prefill.tokens, 1)
        if (frac <= progress) energy = Math.max(energy, 0.5 * this.prefillGlow)
      }
      const alpha = idleAlpha + energy * 0.65
      const hue = this.errorFlash > 0.3 ? '255, 107, 107' : '126, 231, 255'
      ctx.strokeStyle = `rgba(${hue}, ${alpha})`
      ctx.lineWidth = 1 + energy * 1.8
      ctx.beginPath()
      ctx.arc(cx, cy, radius, 0, Math.PI * 2)
      ctx.stroke()
      if (total) this.energies[i] = Math.max(0, this.energies[i] - dt * 0.0035)
    }

    // Core: a soft glass orb that is lit only while a run is active.
    const coreLit = run.active ? 1 : 0
    const breathe = 0.5 + 0.5 * Math.sin(t * 0.0006)
    const coreAlpha = live
      ? 0.05 + 0.03 * breathe + coreLit * 0.3
      : 0.03
    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, inner)
    grad.addColorStop(0, `rgba(154, 123, 255, ${coreAlpha})`)
    grad.addColorStop(0.7, `rgba(126, 231, 255, ${coreAlpha * 0.5})`)
    grad.addColorStop(1, 'rgba(126, 231, 255, 0)')
    ctx.fillStyle = grad
    ctx.beginPath()
    ctx.arc(cx, cy, inner, 0, Math.PI * 2)
    ctx.fill()

    this.prefillGlow = Math.max(0, this.prefillGlow - dt * 0.0004)
    this.errorFlash = Math.max(0, this.errorFlash - dt * 0.0008)
    ctx.restore()
  }
}
