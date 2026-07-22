/* TokenParticleSystem — prompt tokens flowing into the core during prefill,
   one bright outbound particle per really-decoded token, and a golden burst
   when a receipt is sealed. Spawning is driven exclusively by telemetry
   events; with no events the field is empty. */

const MAX_PARTICLES = 480

function rand(min, max) {
  return min + Math.random() * (max - min)
}

export class TokenParticleSystem {
  constructor() {
    this.particles = []
    this.prefillCarry = 0
  }

  spawnPromptWave(frame, count) {
    const n = Math.min(count, 24)
    for (let i = 0; i < n; i += 1) {
      if (this.particles.length >= MAX_PARTICLES) break
      const edgeY = rand(frame.h * 0.18, frame.h * 0.82)
      this.particles.push({
        kind: 'prompt',
        x: -12,
        y: edgeY,
        tx: frame.cx,
        ty: frame.cy + rand(-frame.R * 0.4, frame.R * 0.4),
        life: 1,
        speed: rand(0.55, 0.95),
        size: rand(1.1, 2.4),
        wobble: rand(0, Math.PI * 2),
      })
    }
  }

  spawnDecodedToken(frame) {
    if (this.particles.length >= MAX_PARTICLES) return
    this.particles.push({
      kind: 'token',
      x: frame.cx + frame.R * 0.2,
      y: frame.cy + rand(-frame.R * 0.18, frame.R * 0.18),
      vx: rand(0.16, 0.24),
      vy: rand(-0.025, 0.025),
      life: 1,
      size: rand(2.2, 3.4),
      trail: [],
    })
  }

  spawnReceiptBurst(frame) {
    for (let i = 0; i < 26; i += 1) {
      if (this.particles.length >= MAX_PARTICLES) break
      const angle = (i / 26) * Math.PI * 2
      this.particles.push({
        kind: 'seal',
        x: frame.cx,
        y: frame.cy,
        vx: Math.cos(angle) * rand(0.05, 0.12),
        vy: Math.sin(angle) * rand(0.05, 0.12),
        life: 1,
        size: rand(1.2, 2.0),
      })
    }
  }

  onEvent(evt, frame) {
    if (evt.event === 'token_decoded') this.spawnDecodedToken(frame)
    if (evt.event === 'receipt_written') this.spawnReceiptBurst(frame)
    if (evt.event === 'prefill_progress') {
      // One visible mote per ~2 really-prefilled tokens, capped per event.
      const delta = Math.max(0, (evt.tokens_done || 0) - this.lastPrefillDone || 0)
      this.lastPrefillDone = evt.tokens_done || 0
      this.prefillCarry += delta / 2
      const toSpawn = Math.floor(this.prefillCarry)
      if (toSpawn > 0) {
        this.prefillCarry -= toSpawn
        this.spawnPromptWave(frame, toSpawn)
      }
    }
    if (evt.event === 'prefill_started') {
      this.lastPrefillDone = 0
      this.spawnPromptWave(frame, Math.min(16, Math.max(6, Math.floor((evt.prefill_tokens || 8) / 4))))
    }
  }

  draw(ctx, frame) {
    const { dt } = frame
    ctx.save()
    ctx.globalCompositeOperation = 'lighter'
    for (let i = this.particles.length - 1; i >= 0; i -= 1) {
      const p = this.particles[i]
      if (p.kind === 'prompt') {
        p.wobble += dt * 0.004
        const dx = p.tx - p.x
        const dy = p.ty - p.y
        const dist = Math.hypot(dx, dy)
        if (dist < frame.R * 0.5) {
          p.life -= dt * 0.005
        }
        p.x += (dx / Math.max(dist, 1)) * p.speed * dt * 0.22 + Math.sin(p.wobble) * 0.3
        p.y += (dy / Math.max(dist, 1)) * p.speed * dt * 0.22 + Math.cos(p.wobble * 0.8) * 0.3
        ctx.fillStyle = `rgba(126, 231, 255, ${0.55 * p.life})`
        ctx.beginPath()
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2)
        ctx.fill()
      } else if (p.kind === 'token') {
        p.x += p.vx * dt
        p.y += p.vy * dt
        p.life -= dt * 0.00038
        p.trail.push({ x: p.x, y: p.y })
        if (p.trail.length > 14) p.trail.shift()
        for (let t = 0; t < p.trail.length; t += 1) {
          const seg = p.trail[t]
          const a = (t / p.trail.length) * 0.35 * p.life
          ctx.fillStyle = `rgba(154, 123, 255, ${a})`
          ctx.beginPath()
          ctx.arc(seg.x, seg.y, p.size * (t / p.trail.length), 0, Math.PI * 2)
          ctx.fill()
        }
        ctx.fillStyle = `rgba(236, 244, 255, ${0.9 * p.life})`
        ctx.beginPath()
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2)
        ctx.fill()
      } else if (p.kind === 'seal') {
        p.x += p.vx * dt
        p.y += p.vy * dt
        p.life -= dt * 0.0012
        ctx.fillStyle = `rgba(255, 211, 123, ${0.8 * p.life})`
        ctx.beginPath()
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2)
        ctx.fill()
      }
      if (p.life <= 0 || p.x > frame.w + 24 || p.y < -24 || p.y > frame.h + 24) {
        this.particles.splice(i, 1)
      }
    }
    ctx.restore()
  }
}
