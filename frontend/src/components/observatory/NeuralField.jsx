/* NeuralField — canvas host for the Neural Field renderer: the loaded
   model's real structure (layers, attention flow, KV growth, sampling) lit
   by live telemetry. Same rAF/DPR/ResizeObserver/cleanup skeleton as
   InferenceCanvas; data path is the backend SSE store
   (useInferenceTelemetry + store.drainEvents), NOT the client-side
   telemetryLog bus — the backend is the source of truth for model internals.

   Idle = the network at rest; the only standing motion is the explicitly-
   idle ambient treatment (slow camera drift + starfield), never an
   inference signal. Reduced motion: no drift, no motes; discrete steps. */

import { useEffect, useRef } from 'react'
import { useInferenceTelemetry } from '../../hooks/useInferenceTelemetry'
import { createNeuralFieldRenderer } from '../../lib/observatory/neuralField/renderer'
import { readPalette } from '../../lib/observatory/flowBench'

const DPR_CAP = 2
const STAR_COUNT = 70

function makeStars() {
  return Array.from({ length: STAR_COUNT }, () => ({
    x: Math.random(),
    y: Math.random(),
    r: 0.4 + Math.random() * 1.0,
    a: 0.04 + Math.random() * 0.08,
    drift: 0.000004 + Math.random() * 0.00001,
  }))
}

export default function NeuralField({ apiBase, reducedMotion = false }) {
  const canvasRef = useRef(null)
  const store = useInferenceTelemetry(apiBase)
  const storeRef = useRef(store)
  storeRef.current = store

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return undefined
    const ctx = canvas.getContext('2d')
    const renderer = createNeuralFieldRenderer({ reducedMotion })
    const stars = makeStars()
    let starInk = null
    const refreshInks = () => {
      renderer.refreshPalette()
      const [r, g, b] = readPalette().generation
      starInk = `rgba(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)},`
    }
    refreshInks()
    const themeObserver = new MutationObserver(refreshInks)
    themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })

    let raf = 0
    let lastT = performance.now()
    let dpr = Math.min(window.devicePixelRatio || 1, DPR_CAP)

    const resize = () => {
      dpr = Math.min(window.devicePixelRatio || 1, DPR_CAP)
      const rect = canvas.getBoundingClientRect()
      canvas.width = Math.max(1, Math.round(rect.width * dpr))
      canvas.height = Math.max(1, Math.round(rect.height * dpr))
    }
    resize()
    const observer = new ResizeObserver(resize)
    observer.observe(canvas)

    const tick = (t) => {
      raf = window.requestAnimationFrame(tick)
      const dt = Math.min(t - lastT, 50)
      lastT = t
      const s = storeRef.current
      const w = canvas.width / dpr
      const h = canvas.height / dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, w, h)

      const frame = {
        w,
        h,
        t,
        dt,
        run: s.getRun(),
        connection: s.getConnection(),
        runStale: s.isRunStale(),
      }

      // Ambient starfield (explicitly-idle treatment; static under reduced motion).
      for (const star of stars) {
        if (!reducedMotion) star.x = (star.x + star.drift * dt) % 1
        ctx.fillStyle = starInk + star.a.toFixed(3) + ')'
        ctx.beginPath()
        ctx.arc(star.x * w, star.y * h, star.r, 0, Math.PI * 2)
        ctx.fill()
      }

      // Real events drained once per frame — the only animation source.
      const t0 = import.meta.env.DEV ? performance.now() : 0
      for (const evt of s.drainEvents()) renderer.onEvent(evt, frame)
      renderer.draw(ctx, frame)
      if (import.meta.env.DEV) {
        // Perf-evidence seam (dev builds only): JS cost of event fan-out +
        // draw, excluding compositing. Read by neural-field-evidence.mjs.
        const rec = (window.__nfDrawCost = window.__nfDrawCost || { samples: [], on: false })
        if (rec.on) rec.samples.push(performance.now() - t0)
      }
    }
    raf = window.requestAnimationFrame(tick)

    const onVisibility = () => {
      if (document.hidden) {
        window.cancelAnimationFrame(raf)
        raf = 0
      } else if (!raf) {
        lastT = performance.now()
        raf = window.requestAnimationFrame(tick)
      }
    }
    document.addEventListener('visibilitychange', onVisibility)

    return () => {
      if (raf) window.cancelAnimationFrame(raf)
      document.removeEventListener('visibilitychange', onVisibility)
      observer.disconnect()
      themeObserver.disconnect()
    }
  }, [reducedMotion])

  return (
    <div className="flowbench neuralfield">
      <canvas
        ref={canvasRef}
        className="flowbench__canvas"
        aria-label="Neural Field — live model structure lit by real inference telemetry"
      />
    </div>
  )
}
