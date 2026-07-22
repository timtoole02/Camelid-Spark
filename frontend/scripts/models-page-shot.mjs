/* One-off full-page screenshot of the Models (#library) view for before/after
   evidence. Windows-friendly (finds Chrome/Edge). Usage:
     node scripts/models-page-shot.mjs --out qa/models-before.png [--url http://127.0.0.1:4175] [--width 1280] */
import { mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'
import { existsSync } from 'node:fs'
import puppeteer from 'puppeteer-core'

const args = new Map()
for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i].replace(/^--/, ''), process.argv[i + 1])
const url = args.get('url') || 'http://127.0.0.1:4175'
const out = args.get('out') || 'models-page.png'
const width = Number(args.get('width') || 1280)

const CHROME_PATHS = [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
]
const executablePath = CHROME_PATHS.find((p) => existsSync(p))
if (!executablePath) throw new Error('No local Chrome/Edge found')

await mkdir(dirname(out), { recursive: true }).catch(() => {})
const browser = await puppeteer.launch({ executablePath, headless: 'new' })
try {
  const page = await browser.newPage()
  await page.setViewport({ width, height: 900 })
  await page.goto(`${url}/#library`, { waitUntil: 'networkidle2', timeout: 30000 })
  await new Promise((r) => setTimeout(r, 2500))
  await page.screenshot({ path: out, fullPage: true })
  console.log(`saved ${out}`)
} finally {
  await browser.close()
}
