/* SamplerBloom — a probability bloom for the candidates the sampler really
   considered at a decode step. Petals fan out above the core, length and
   brightness proportional to each candidate's true post-softmax probability;
   the chosen token's petal is highlighted. Fades unless refreshed by real
   sampler_step events. */

const BLOOM_TTL_MS = 1100

export class SamplerBloom {
  constructor() {
    this.bloom = null // { candidates, chosen, bornT }
  }

  onEvent(evt, frame) {
    if (evt.event === 'sampler_step' && Array.isArray(evt.candidates) && evt.candidates.length) {
      this.bloom = {
        candidates: evt.candidates.slice(0, 8),
        chosen: evt.chosen_token_id,
        bornT: frame.t,
      }
    }
    if (evt.event === 'inference_finished') this.bloom = null
  }

  draw(ctx, frame) {
    if (!this.bloom) return
    const { cx, cy, R, t } = frame
    const age = t - this.bloom.bornT
    if (age > BLOOM_TTL_MS) {
      this.bloom = null
      return
    }
    const fade = 1 - age / BLOOM_TTL_MS
    const { candidates, chosen } = this.bloom
    const spread = Math.PI * 0.7
    const baseAngle = -Math.PI / 2 - spread / 2
    const innerR = R * 1.42

    ctx.save()
    ctx.globalCompositeOperation = 'lighter'
    candidates.forEach((cand, i) => {
      const angle = baseAngle + (candidates.length > 1 ? (i / (candidates.length - 1)) * spread : spread / 2)
      const len = R * 0.18 + R * 0.5 * Math.sqrt(Math.max(cand.prob, 0))
      const isChosen = cand.token_id === chosen
      const x0 = cx + Math.cos(angle) * innerR
      const y0 = cy + Math.sin(angle) * innerR
      const x1 = cx + Math.cos(angle) * (innerR + len)
      const y1 = cy + Math.sin(angle) * (innerR + len)
      const alpha = fade * (isChosen ? 0.85 : 0.18 + cand.prob * 0.5)
      ctx.strokeStyle = isChosen ? `rgba(236, 244, 255, ${alpha})` : `rgba(126, 231, 255, ${alpha})`
      ctx.lineWidth = isChosen ? 2.2 : 1.2
      ctx.lineCap = 'round'
      ctx.beginPath()
      ctx.moveTo(x0, y0)
      ctx.lineTo(x1, y1)
      ctx.stroke()
      ctx.fillStyle = ctx.strokeStyle
      ctx.beginPath()
      ctx.arc(x1, y1, isChosen ? 2.6 : 1.6, 0, Math.PI * 2)
      ctx.fill()
    })
    ctx.restore()
  }
}
