#!/usr/bin/env node
/* smoke:flow (Phase 6.2): objective visual floors for the Flow Bench, run
   against ONE scripted REAL request per theme. Floors against regression-to-
   invisible, not aesthetic targets. Needs backend + dev server + a loaded
   generation-ready model. */
import assert from 'node:assert/strict'
import puppeteer from 'puppeteer-core'

const IDLE_WAIT_MS = Number(process.env.FLOW_SMOKE_IDLE_MS || 60000)
const lum = ([r, g, b]) => {
  const lin = (c) => { const s = c / 255; return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4 }
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
}
const contrast = (a, b) => { const [hi, lo] = lum(a) >= lum(b) ? [lum(a), lum(b)] : [lum(b), lum(a)]; return (hi + 0.05) / (lo + 0.05) }

const browser = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: 'new', args: ['--enable-gpu', '--use-angle=metal', '--ignore-gpu-blocklist'] })

async function runTheme(theme, withIdleCheck) {
  const page = await browser.newPage()
  await page.setViewport({ width: 1440, height: 900 })
  await page.evaluateOnNewDocument((t) => window.localStorage.setItem('camelid-theme', t), theme)
  await page.goto('http://127.0.0.1:4175/#observatory', { waitUntil: 'domcontentloaded' })
  await new Promise(r => setTimeout(r, 3000))
  const grab = () => page.evaluate(() => {
    const c = document.querySelector('.flowbench__canvas')
    const probe = document.createElement('canvas'); probe.width = c.width; probe.height = c.height
    const ctx = probe.getContext('2d'); ctx.drawImage(c, 0, 0)
    const d = ctx.getImageData(0, 0, c.width, c.height).data
    const px = []
    for (let i = 0; i < d.length; i += 16) px.push([d[i], d[i + 1], d[i + 2], d[i + 3]])
    return px
  })
  const bg = await page.evaluate(() => {
    const v = getComputedStyle(document.documentElement).getPropertyValue('--color-bg').trim()
    const n = parseInt(v.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
  })
  const idle = await grab()
  const litPct = (px) => px.filter(p => p[3] > 12).length / px.length * 100
  assert.ok(litPct(idle) < 1, `${theme}: idle canvas should be near-empty`)

  await page.evaluate(() => document.querySelector('.rail__new-chat')?.click())
  await page.waitForSelector('.cxcomposer__input', { timeout: 10000 })
  await page.type('.cxcomposer__input', 'Count from 1 to 150, one number per line, no other text.')
  await page.keyboard.press('Enter')
  await page.waitForFunction(() => (window.__camelidFlowBenchLog || []).some(e => e.type === 'first_content'), { timeout: 120000 })
  await page.evaluate(() => [...document.querySelectorAll('.rail__nav-item')].find(b => b.textContent.includes('Observatory'))?.click())
  await new Promise(r => setTimeout(r, 250))
  const atTtft = await grab()
  assert.ok(litPct(atTtft) >= 1, `${theme}: >=1% of pixels must depart idle within 250ms of first token (got ${litPct(atTtft).toFixed(2)}%)`)

  await new Promise(r => setTimeout(r, 3000))
  const mid = await grab()
  const inkMid = mid.filter(p => p[3] > 12).sort((a, b) => b[3] - a[3])
  const top = inkMid.slice(0, Math.max(1, Math.floor(inkMid.length / 10)))
  const meanTop = top.reduce((acc, p) => [acc[0] + p[0], acc[1] + p[1], acc[2] + p[2]], [0, 0, 0]).map(v => v / top.length)
  const ratio = contrast(meanTop, bg)
  assert.ok(ratio >= 3, `${theme}: top-decile ink contrast vs background must be >=3:1 (got ${ratio.toFixed(2)})`)

  await page.waitForFunction(() => document.querySelectorAll('.flowbench-rail__row').length >= 1, { timeout: 180000 })
  const done = await grab()
  assert.ok(litPct(done) >= 15, `${theme}: >=15% of pixels must have departed idle by completion (got ${litPct(done).toFixed(2)}%)`)
  console.log(`${theme}: ttft=${litPct(atTtft).toFixed(1)}% complete=${litPct(done).toFixed(1)}% contrast=${ratio.toFixed(2)}:1`)

  if (withIdleCheck) {
    for (let i = 0; i < 3; i += 1) {
      await new Promise(r => setTimeout(r, IDLE_WAIT_MS / 3))
      const probe = await grab()
      const litNow = probe.filter(p => p[3] > 12).length / probe.length * 100
      const maxA = Math.max(...probe.map(p => p[3]))
      console.log(`  idle+${Math.round(IDLE_WAIT_MS / 3 * (i + 1) / 1000)}s: lit=${litNow.toFixed(1)}% maxA=${maxA}`)
    }
    const a = await grab()
    await new Promise(r => setTimeout(r, 1200))
    const b = await grab()
    let changed = 0
    for (let i = 0; i < a.length; i += 1) if (Math.abs(a[i][3] - b[i][3]) > 10) changed += 1
    const changedPct = changed / a.length * 100
    assert.ok(changedPct < 0.5, `${theme}: after ${IDLE_WAIT_MS / 1000}s idle, frame-to-frame change must be near-still (got ${changedPct.toFixed(2)}%)`)
    console.log(`${theme}: idle stillness after ${IDLE_WAIT_MS / 1000}s = ${changedPct.toFixed(3)}% frame delta`)
  }
  await page.close()
}

try {
  await runTheme('dark', true)
  await runTheme('light', false)
  console.log('flow visual smoke passed')
} finally {
  await browser.close()
}
