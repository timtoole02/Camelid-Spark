/* WCAG AA contrast audit for the Camelid token palettes (Phase 1 gate).
   Parses src/styles/tokens.css, resolves the dark (:root) and light
   ([data-theme="light"]) palettes, alpha-composites rgba() values over the
   theme background, and asserts AA for every text-on-surface and
   status-color-on-surface pair the UI actually renders.

   Normal text needs 4.5:1. Tokens used only at large/bold sizes or as
   non-text indicators (dots, borders) are checked at 3:1 and marked so. */

import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const css = readFileSync(join(here, '..', 'src', 'styles', 'tokens.css'), 'utf8')

function extractBlock(source, opener) {
  const start = source.indexOf(opener)
  if (start === -1) throw new Error(`block not found: ${opener}`)
  let depth = 0
  let i = source.indexOf('{', start)
  const open = i
  for (; i < source.length; i += 1) {
    if (source[i] === '{') depth += 1
    if (source[i] === '}') {
      depth -= 1
      if (depth === 0) break
    }
  }
  return source.slice(open + 1, i)
}

function parseVars(block) {
  const vars = {}
  for (const match of block.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)) {
    vars[match[1]] = match[2].trim()
  }
  return vars
}

function parseColor(value) {
  let m = value.match(/^#([0-9a-f]{6})$/i)
  if (m) {
    const n = parseInt(m[1], 16)
    return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255, a: 1 }
  }
  m = value.match(/^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+)\s*)?\)$/i)
  if (m) return { r: +m[1], g: +m[2], b: +m[3], a: m[4] === undefined ? 1 : +m[4] }
  return null
}

function composite(fg, bg) {
  const a = fg.a
  return {
    r: fg.r * a + bg.r * (1 - a),
    g: fg.g * a + bg.g * (1 - a),
    b: fg.b * a + bg.b * (1 - a),
    a: 1,
  }
}

function luminance({ r, g, b }) {
  const lin = (c) => {
    const s = c / 255
    return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4
  }
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
}

function contrast(fg, bg) {
  const l1 = luminance(fg)
  const l2 = luminance(bg)
  const [hi, lo] = l1 >= l2 ? [l1, l2] : [l2, l1]
  return (hi + 0.05) / (lo + 0.05)
}

const darkVars = parseVars(extractBlock(css, ':root {'))
const lightVars = parseVars(extractBlock(css, ":root[data-theme='light']"))

/* Text + status tokens, checked against every surface they can sit on.
   level: 'AA-normal' (4.5) or 'AA-large' (3.0, also non-text indicators). */
const CHECKS = [
  { fg: '--color-text', level: 'AA-normal' },
  { fg: '--color-text-muted', level: 'AA-normal' },
  { fg: '--color-text-faint', level: 'AA-normal' },
  { fg: '--color-accent-text', level: 'AA-normal' },
  { fg: '--color-verified', level: 'AA-normal' },
  { fg: '--color-evidence', level: 'AA-normal' },
  { fg: '--color-planned', level: 'AA-normal' },
  { fg: '--color-unsupported', level: 'AA-normal' },
  { fg: '--color-ready', level: 'AA-normal' },
  { fg: '--color-warning', level: 'AA-normal' },
  { fg: '--color-error', level: 'AA-normal' },
  { fg: '--color-info', level: 'AA-normal' },
]
const SURFACES = ['--color-bg', '--color-bg-elevated', '--color-surface', '--color-surface-subtle', '--color-surface-strong']

/* Fill pairs: text drawn on a colored fill. */
const FILL_PAIRS = [
  { fg: '--color-accent-ink', bg: '--color-accent', level: 'AA-normal' },
  { fg: '--color-verified-ink', bg: '--color-verified', level: 'AA-normal' },
]

const MIN = { 'AA-normal': 4.5, 'AA-large': 3.0 }

let failures = 0
let checks = 0

for (const [themeName, vars] of [['dark', darkVars], ['light', lightVars]]) {
  const resolve = (name) => {
    const raw = vars[name] ?? darkVars[name]
    if (!raw) throw new Error(`missing token ${name} in ${themeName}`)
    const color = parseColor(raw)
    if (!color) throw new Error(`unparseable color ${name}: ${raw} (${themeName})`)
    return color
  }
  const themeBg = resolve('--color-bg')
  const surface = (name) => composite(resolve(name), themeBg)

  for (const check of CHECKS) {
    for (const surfaceName of SURFACES) {
      const fg = composite(resolve(check.fg), surface(surfaceName))
      const ratio = contrast(fg, surface(surfaceName))
      checks += 1
      const min = MIN[check.level]
      const ok = ratio >= min
      if (!ok) {
        failures += 1
        console.error(`FAIL [${themeName}] ${check.fg} on ${surfaceName}: ${ratio.toFixed(2)} < ${min} (${check.level})`)
      }
    }
  }
  for (const pair of FILL_PAIRS) {
    const bg = surface('--color-bg')
    const fill = composite(resolve(pair.bg), bg)
    const fg = composite(resolve(pair.fg), fill)
    const ratio = contrast(fg, fill)
    checks += 1
    if (ratio < MIN[pair.level]) {
      failures += 1
      console.error(`FAIL [${themeName}] ${pair.fg} on ${pair.bg}: ${ratio.toFixed(2)} < ${MIN[pair.level]}`)
    }
  }
}

if (failures > 0) {
  console.error(`\ncontrast-check: ${failures}/${checks} pairs below WCAG AA`)
  process.exit(1)
}
console.log(`contrast-check: all ${checks} token pairs pass WCAG AA in both themes`)
