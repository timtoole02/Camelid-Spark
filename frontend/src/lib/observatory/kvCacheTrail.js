/* KVCacheTrail — the memory ring: an arc around the core whose filled extent
   tracks the run's real KV cache position. The scale is the run's own
   working window (prompt + max_tokens), so growth is visible at chat scale
   while the absolute numbers stay truthful in the overlays. */

export class KVCacheTrail {
  constructor() {
    this.displayFill = 0
    this.headGlow = 0
  }

  onEvent(evt) {
    if (evt.event === 'kv_cache_updated' || evt.event === 'token_decoded') this.headGlow = 1
    if (evt.event === 'inference_started') {
      this.displayFill = 0
      this.headGlow = 0
    }
  }

  draw(ctx, frame) {
    const { cx, cy, R, dt, run, connection } = frame
    const radius = R * 1.62
    const start = -Math.PI / 2

    const windowTokens = Math.max(run.promptTokens + run.maxTokens, 64)
    const targetFill = run.kv.position > 0 ? Math.min(run.kv.position / windowTokens, 1) : 0
    this.displayFill += (targetFill - this.displayFill) * Math.min(dt * 0.01, 1)

    ctx.save()
    // Track (always visible, faint).
    ctx.strokeStyle = connection === 'live' ? 'rgba(126, 231, 255, 0.08)' : 'rgba(126, 231, 255, 0.04)'
    ctx.lineWidth = 3
    ctx.beginPath()
    ctx.arc(cx, cy, radius, 0, Math.PI * 2)
    ctx.stroke()

    if (this.displayFill > 0.002) {
      const end = start + this.displayFill * Math.PI * 2
      ctx.strokeStyle = `rgba(154, 123, 255, ${0.4 + this.headGlow * 0.25})`
      ctx.lineWidth = 3
      ctx.lineCap = 'round'
      ctx.beginPath()
      ctx.arc(cx, cy, radius, start, end)
      ctx.stroke()
      // Glowing head where memory is being appended right now.
      const hx = cx + Math.cos(end) * radius
      const hy = cy + Math.sin(end) * radius
      ctx.globalCompositeOperation = 'lighter'
      ctx.fillStyle = `rgba(213, 196, 255, ${0.25 + this.headGlow * 0.6})`
      ctx.beginPath()
      ctx.arc(hx, hy, 3 + this.headGlow * 3, 0, Math.PI * 2)
      ctx.fill()
    }
    this.headGlow = Math.max(0, this.headGlow - dt * 0.002)
    ctx.restore()
  }
}
