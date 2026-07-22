#!/usr/bin/env node
/* Neural Field Phase 5 evidence capture (design-evidence/neural-field/).
   Drives ONE real TinyLlama generation per capture pass against a live
   backend (127.0.0.1:8181) through the dev server (127.0.0.1:4175) and
   screenshots the Neural Field canvas in each contract state. Every captured
   state is produced by real telemetry events — the only manufactured input
   is a deliberately-invalid API request to trigger a REAL api_error while a
   generation is live (that is the backend's true inference_error path).

   Usage:
     node scripts/neural-field-evidence.mjs full            # idle→run→finished(+receipt)→error→reduced + perf @DPR1/@DPR2
     node scripts/neural-field-evidence.mjs decode <label>  # one run, decode frames named decode-<label>
     node scripts/neural-field-evidence.mjs unavailable     # backend must be DOWN; captures the unavailable state
*/
import { mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import puppeteer from 'puppeteer-core'

const MODE = process.argv[2] || 'full'
const LABEL = process.argv[3] || 'default'
const OUT = join(import.meta.dirname, '..', 'design-evidence', 'neural-field', 'frames')
mkdirSync(OUT, { recursive: true })

const API = 'http://127.0.0.1:8181'
const APP = 'http://127.0.0.1:4175/#observatory'
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe'
const PROMPT = 'Explain, in plain language and complete sentences, how a transformer language model turns a prompt into a reply: tokenization, embeddings, the stack of layers with attention and feed-forward blocks, the key-value cache that grows one entry per token, and the final sampling step that picks each next token. Then list five everyday analogies for attention, one per line, and keep going with more detail about each analogy until you run out of room. '.repeat(2)

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

async function launch(dpr = 1) {
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new', args: ['--enable-gpu', '--ignore-gpu-blocklist'] })
  const page = await browser.newPage()
  await page.setViewport({ width: 1600, height: 900, deviceScaleFactor: dpr })
  await page.evaluateOnNewDocument(() => {
    window.localStorage.setItem('camelid.observatory.renderer', 'neuralfield')
    window.localStorage.setItem('camelid-theme', 'dark')
    // Independent SSE tap: records which REAL events arrived (evidence ledger).
    window.__nfEvents = []
    const es = new EventSource('http://127.0.0.1:8181/api/telemetry/stream')
    es.addEventListener('telemetry', (e) => {
      try { window.__nfEvents.push(JSON.parse(e.data)) } catch { /* noop */ }
    })
    // Frame-time recorder for PERF.md: measures real rAF cadence while active.
    window.__nfFrames = { deltas: [], on: false }
    let last = null
    const loop = (t) => {
      if (window.__nfFrames.on && last != null) window.__nfFrames.deltas.push(t - last)
      last = t
      window.requestAnimationFrame(loop)
    }
    window.requestAnimationFrame(loop)
  })
  await page.goto(APP, { waitUntil: 'domcontentloaded' })
  await page.waitForSelector('.neuralfield .flowbench__canvas', { timeout: 20000 })
  return { browser, page }
}

async function shoot(page, name) {
  const canvas = await page.$('.neuralfield')
  await canvas.screenshot({ path: join(OUT, `${name}.png`) })
  console.log(`captured ${name}.png`)
}

async function modelId(page) {
  return page.evaluate(async (api) => {
    const res = await fetch(`${api}/v1/models`)
    const body = await res.json()
    return body.data?.[0]?.id || null
  }, API)
}

/* One REAL streamed generation via the OpenAI-compatible endpoint. */
function startGeneration(page, model, maxTokens) {
  return page.evaluate(async (api, model, maxTokens, prompt) => {
    const res = await fetch(`${api}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, max_tokens: maxTokens, stream: true, messages: [{ role: 'user', content: prompt }] }),
    })
    const reader = res.body.getReader()
    for (;;) {
      const { done } = await reader.read()
      if (done) break
    }
    return res.status
  }, API, model, maxTokens, PROMPT)
}

function perfStats(deltas) {
  if (!deltas.length) return null
  const sorted = [...deltas].sort((a, b) => a - b)
  const avg = deltas.reduce((s, v) => s + v, 0) / deltas.length
  const p95 = sorted[Math.min(Math.floor(sorted.length * 0.95), sorted.length - 1)]
  return { frames: deltas.length, avgMs: +avg.toFixed(2), p95Ms: +p95.toFixed(2) }
}

async function perfRun(page, model, dprLabel) {
  await page.evaluate(() => {
    window.__nfFrames.deltas = []
    window.__nfFrames.on = true
    window.__nfDrawCost = { samples: [], on: true }
  })
  const status = await startGeneration(page, model, 50)
  await page.evaluate(() => { window.__nfFrames.on = false; window.__nfDrawCost.on = false })
  const cadence = await page.evaluate(() => window.__nfFrames.deltas)
  const draw = await page.evaluate(() => window.__nfDrawCost.samples)
  const out = { dpr: dprLabel, httpStatus: status, cadence: perfStats(cadence), drawCost: perfStats(draw) }
  console.log(`perf ${dprLabel}:`, JSON.stringify(out))
  return out
}

async function main() {
  if (MODE === 'unavailable') {
    const { browser, page } = await launch(1)
    // Backend is down: EventSource cannot connect; wait for the store to
    // observe CLOSED and the canvas to drop to the unavailable treatment.
    await sleep(12000)
    await shoot(page, 'unavailable')
    await browser.close()
    return
  }

  const { browser, page } = await launch(1)
  const model = await modelId(page)
  if (!model) throw new Error('no model loaded on the backend')
  console.log(`model: ${model}`)
  await sleep(2500)

  if (MODE === 'decode') {
    await shoot(page, `idle-${LABEL}`)
    const run = startGeneration(page, model, 50)
    await sleep(600)
    await shoot(page, `prefill-${LABEL}`)
    // Wait for the REAL decode_started event, then shoot mid-decode.
    await page.waitForFunction(() => window.__nfEvents.some((e) => e.event === 'decode_started'), { timeout: 120000 })
    await sleep(800)
    await shoot(page, `decode-${LABEL}`)
    await sleep(800)
    await shoot(page, `sampler-kv-${LABEL}`)
    await run
    await sleep(400)
    await shoot(page, `finished-${LABEL}`)
    const events = await page.evaluate(() => window.__nfEvents.map((e) => e.event))
    console.log('events seen:', [...new Set(events)].join(', '))
    await browser.close()
    return
  }

  // ---- full pass ----
  await shoot(page, 'idle-live')

  // Long visual run (TinyLlama decodes ~45 tok/s here; 400 tokens ≈ 9s) so
  // every mid-run screenshot lands while decode is genuinely in flight.
  const run = startGeneration(page, model, 400)
  await sleep(350)
  await shoot(page, 'prefill')
  await sleep(1600)
  await shoot(page, 'decode')
  await sleep(900)
  await shoot(page, 'sampler-bloom')
  await run
  await sleep(350)
  await shoot(page, 'finished-ok')

  // PERF @ DPR 1 on a fresh 50-token run.
  const perf1 = await perfRun(page, model, 'dpr1')

  // REAL inference_error: api_error() surfaces any API failure that happens
  // while a generation is live onto the telemetry stream (src/api/mod.rs).
  // /api/models/inspect with a nonexistent path routes through api_error, so
  // firing it mid-generation produces a genuine inference_error event.
  const longRun = startGeneration(page, model, 300)
  await sleep(1500)
  const inspectStatus = await page.evaluate(async (api) => {
    const res = await fetch(`${api}/api/models/inspect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: 'C:/does/not/exist.gguf' }),
    })
    return res.status
  }, API)
  console.log(`inspect (error trigger) status: ${inspectStatus}`)
  await sleep(400)
  await shoot(page, 'error')
  await longRun
  await sleep(1200)

  // receipt_written is opt-in per request (camelid_receipt: true) on the
  // supported TinyLlama lane; a real sealed receipt emits the event.
  await page.evaluate(async (api, model) => {
    await fetch(`${api}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, max_tokens: 24, camelid_receipt: true, messages: [{ role: 'user', content: 'Name three colors.' }] }),
    })
  }, API, model)
  await sleep(450)
  await shoot(page, 'receipt-burst')

  // Reduced motion: the rail's real Motion toggle re-mounts the renderer.
  await page.evaluate(() => {
    const btn = [...document.querySelectorAll('.flowbench-rail__foot button')].find((b) => b.textContent.includes('Motion'))
    btn?.click()
  })
  await sleep(600)
  const rmRun = startGeneration(page, model, 40)
  await sleep(1500)
  await shoot(page, 'reduced-motion')
  await rmRun

  const events = await page.evaluate(() => window.__nfEvents.map((e) => e.event))
  const seen = [...new Set(events)]
  console.log('events seen:', seen.join(', '))
  await browser.close()

  // PERF @ DPR 2 in a fresh browser context.
  const ctx2 = await launch(2)
  const model2 = await modelId(ctx2.page)
  await sleep(2000)
  const perf2 = await perfRun(ctx2.page, model2, 'dpr2')
  await ctx2.browser.close()

  writeFileSync(join(OUT, '..', 'capture-summary.json'), JSON.stringify({ capturedAt: new Date().toISOString(), model, eventsSeen: seen, perf: [perf1, perf2] }, null, 2))
  console.log('full pass complete')
}

await main()
