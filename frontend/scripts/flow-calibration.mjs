#!/usr/bin/env node
/* Flow Bench calibration & demo driver (Phase 6.2).
   Sends REAL prompts to the loaded supported model through the actual UI
   (varied lengths, one aborted stream, one operator-forced transport error via
   a dead API base) — no synthetic telemetry, just operator-initiated real
   traffic. Saves frames + coverage stats for tuning iterations.
   Usage: node scripts/flow-calibration.mjs --out design-evidence/phase-6.2/iterations/iter1 [--theme dark] [--quick] */
import { mkdirSync } from 'node:fs'
import puppeteer from 'puppeteer-core'

const args = new Map()
for (let i = 2; i < process.argv.length; i += 1) {
  const a = process.argv[i]
  if (a.startsWith('--')) args.set(a.slice(2), process.argv[i + 1]?.startsWith('--') || process.argv[i + 1] === undefined ? '1' : process.argv[++i])
}
const out = args.get('out') || 'design-evidence/phase-6.2/iterations/scratch'
const theme = args.get('theme') || 'dark'
const quick = args.has('quick')
mkdirSync(out, { recursive: true })

const browser = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: 'new', args: ['--enable-gpu', '--use-angle=metal', '--ignore-gpu-blocklist'] })
const page = await browser.newPage()
await page.setViewport({ width: 1440, height: 900 })
await page.evaluateOnNewDocument((t) => window.localStorage.setItem('camelid-theme', t), theme)
await page.goto('http://127.0.0.1:4175/#observatory', { waitUntil: 'domcontentloaded' })
await new Promise(r => setTimeout(r, 3000))

export const stats = async () => page.evaluate(() => {
  const c = document.querySelector('.flowbench__canvas')
  const probe = document.createElement('canvas'); probe.width = c.width; probe.height = c.height
  const ctx = probe.getContext('2d'); ctx.drawImage(c, 0, 0)
  const d = ctx.getImageData(0, 0, c.width, c.height).data
  let lit = 0, n = 0
  const samples = []
  for (let i = 0; i < d.length; i += 16) {
    n += 1
    if (d[i + 3] > 12) { lit += 1; samples.push([d[i], d[i + 1], d[i + 2], d[i + 3]]) }
  }
  samples.sort((a, b) => b[3] - a[3])
  const top = samples.slice(0, Math.max(1, Math.floor(samples.length / 10)))
  const mean = top.length ? top.reduce((acc, s) => [acc[0] + s[0], acc[1] + s[1], acc[2] + s[2]], [0, 0, 0]).map(v => Math.round(v / top.length)) : null
  return { litPct: +(lit / n * 100).toFixed(2), topDecileInk: mean }
})

const snap = async (name) => {
  await page.screenshot({ path: `${out}/${name}.png` })
  const s = await stats()
  console.log(`${name}: lit=${s.litPct}% topInk=${JSON.stringify(s.topDecileInk)}`)
  return s
}

const sendChat = async (prompt) => {
  await page.evaluate(() => document.querySelector('.rail__new-chat')?.click())
  await page.waitForSelector('.cxcomposer__input', { timeout: 10000 })
  await page.type('.cxcomposer__input', prompt)
  await page.keyboard.press('Enter')
  await new Promise(r => setTimeout(r, 250))
  await page.evaluate(() => [...document.querySelectorAll('.rail__nav-item')].find(b => b.textContent.includes('Observatory'))?.click())
}

await snap('00-idle')
await sendChat('Count from 1 to 150, one number per line, no other text.')
await new Promise(r => setTimeout(r, 900)); await snap('01-ttft')
await new Promise(r => setTimeout(r, 3000)); await snap('02-midstream')
await page.waitForFunction(() => document.querySelectorAll('.flowbench-rail__row').length >= 1, { timeout: 180000 })
await new Promise(r => setTimeout(r, 1200)); await snap('03-complete')

if (!quick) {
  // aborted stream (real interrupt)
  await sendChat('Count from 1 to 300, one number per line.')
  await new Promise(r => setTimeout(r, 2500))
  await page.evaluate(() => document.querySelector('.rail__new-chat')?.click())
  await new Promise(r => setTimeout(r, 400))
  await page.keyboard.press('Escape')
  await page.evaluate(() => [...document.querySelectorAll('.rail__nav-item')].find(b => b.textContent.includes('Observatory'))?.click())
  await new Promise(r => setTimeout(r, 1200)); await snap('04-aborted')
  // forced transport error: real request against a dead base (operator-initiated)
  await page.evaluate(() => { window.localStorage.setItem('camelid.apiBase', 'http://127.0.0.1:9999') })
  await page.evaluate(() => document.querySelector('.rail__new-chat')?.click())
  await new Promise(r => setTimeout(r, 2500))
  await page.evaluate(() => window.localStorage.setItem('camelid.apiBase', 'http://127.0.0.1:8181'))
  await page.evaluate(() => [...document.querySelectorAll('.rail__nav-item')].find(b => b.textContent.includes('Observatory'))?.click())
  await new Promise(r => setTimeout(r, 1500)); await snap('05-after-error-window')
}
await browser.close()
