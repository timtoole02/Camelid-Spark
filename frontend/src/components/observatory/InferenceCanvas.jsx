/* InferenceCanvas — the rendering surface of the Inference Observatory.
   A single requestAnimationFrame loop draws the renderer modules over a dark
   field. Every animated element is driven by the telemetry store (real
   backend events); the only standing motion is the explicitly-idle ambient
   starfield behind the resting geometry. */

import { useEffect, useRef } from 'react'
import { TokenParticleSystem } from '../../lib/observatory/tokenParticles'
import { LayerVisualizer } from '../../lib/observatory/layerVisualizer'
import { KVCacheTrail } from '../../lib/observatory/kvCacheTrail'
import { SamplerBloom } from '../../lib/observatory/samplerBloom'
import { ClusterConstellation } from '../../lib/observatory/clusterConstellation'

const STAR_COUNT = 90

function makeStars() {
  return Array.from({ length: STAR_COUNT }, () => ({
    x: Math.random(),
    y: Math.random(),
    r: 0.4 + Math.random() * 1.1,
    a: 0.04 + Math.random() * 0.1,
    drift: 0.000004 + Math.random() * 0.000012,
  }))
}

export default function InferenceCanvas({ store, showLabels = false }) {
  const canvasRef = useRef(null)
  const storeRef = useRef(store)
  const labelsRef = useRef(showLabels)
  storeRef.current = store
  labelsRef.current = showLabels

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return undefined
    const ctx = canvas.getContext('2d')
    const modules = [
      new KVCacheTrail(),
      new LayerVisualizer(),
      new SamplerBloom(),
      new TokenParticleSystem(),
      new ClusterConstellation(),
    ]
    const stars = makeStars()
    let raf = 0
    let lastT = performance.now()
    let dpr = window.devicePixelRatio || 1

    const resize = () => {
      dpr = window.devicePixelRatio || 1
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

      const frame = {
        w,
        h,
        t,
        dt,
        cx: w * 0.5,
        cy: h * 0.5,
        R: Math.min(w, h) * 0.17,
        run: s.getRun(),
        workers: s.getWorkers(),
        connection: s.getConnection(),
        showLabels: labelsRef.current,
      }

      // Background: deep field with a soft vignette.
      ctx.fillStyle = '#04060c'
      ctx.fillRect(0, 0, w, h)
      const vignette = ctx.createRadialGradient(frame.cx, frame.cy, 0, frame.cx, frame.cy, Math.max(w, h) * 0.75)
      vignette.addColorStop(0, 'rgba(13, 20, 38, 0.55)')
      vignette.addColorStop(1, 'rgba(2, 3, 7, 0)')
      ctx.fillStyle = vignette
      ctx.fillRect(0, 0, w, h)

      // Idle ambient: slow starfield (present in every state; it is the sky,
      // not an inference signal).
      stars.forEach((star) => {
        star.x = (star.x + star.drift * dt) % 1
        ctx.fillStyle = `rgba(170, 196, 230, ${star.a})`
        ctx.beginPath()
        ctx.arc(star.x * w, star.y * h, star.r, 0, Math.PI * 2)
        ctx.fill()
      })

      // Real events drained once per frame, fanned to every module.
      const events = s.drainEvents()
      for (const evt of events) {
        for (const mod of modules) {
          if (mod.onEvent) mod.onEvent(evt, frame)
        }
      }
      for (const mod of modules) mod.draw(ctx, frame)
    }
    raf = window.requestAnimationFrame(tick)
    return () => {
      window.cancelAnimationFrame(raf)
      observer.disconnect()
    }
  }, [])

  return <canvas ref={canvasRef} className="observatory-canvas" aria-label="Live inference visualization" />
}
