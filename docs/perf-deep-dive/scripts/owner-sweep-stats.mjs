#!/usr/bin/env node
// Paired stats for `camelid bench-owner-sweep` JSONL. Configs are measured INTERLEAVED per round
// (one model load, env switched per generation), so per-round ratios cancel slow thermal/clock
// drift. Reports per-config median prefill tok/s and, for each pair, the median per-round ratio +
// a bootstrap 95% CI + sign-test => "significant" only if the CI excludes 1.0 (the effect survives
// the box's noise). This is the harness that resolves what the cross-invocation A/B could not.
//
// Usage: node owner-sweep-stats.mjs <sweep.jsonl> [label]
import { readFileSync } from 'node:fs'

const recs = readFileSync(process.argv[2], 'utf8').split('\n').filter(l => l.trim().startsWith('{')).map(l => JSON.parse(l))
const label = process.argv[3] || (recs[0] && recs[0].model) || 'model'
const med = (a) => { const v = a.filter(Number.isFinite).slice().sort((x, y) => x - y); if (!v.length) return null; const m = v.length >> 1; return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2 }
const r3 = (x) => Number.isFinite(x) ? Math.round(x * 1000) / 1000 : null

const byRound = new Map()
for (const r of recs) {
  if (!byRound.has(r.round)) byRound.set(r.round, {})
  byRound.get(r.round)[r.config] = r.prefill_tok_s
}
const rounds = [...byRound.values()]
const configs = [...new Set(recs.map(r => r.config))]
const perConfigMedian = Object.fromEntries(configs.map(c => [c, r3(med(rounds.map(x => x[c]).filter(Number.isFinite)))]))

function bootstrapCI(ratios, B = 4000) {
  const n = ratios.length
  const meds = []
  for (let b = 0; b < B; b++) {
    const s = []
    for (let i = 0; i < n; i++) s.push(ratios[(Math.random() * n) | 0])
    meds.push(med(s))
  }
  meds.sort((a, b) => a - b)
  return [meds[Math.floor(B * 0.025)], meds[Math.floor(B * 0.975)]]
}
function compare(base, cand) {
  const ratios = rounds.map(x => (Number.isFinite(x[base]) && Number.isFinite(x[cand]) && x[base] > 0) ? x[cand] / x[base] : null).filter(Number.isFinite)
  if (ratios.length < 3) return { base, cand, n: ratios.length, note: 'insufficient' }
  const m = med(ratios)
  const [lo, hi] = bootstrapCI(ratios)
  const wins = ratios.filter(r => r > 1).length
  const significant = lo > 1.0 || hi < 1.0
  return { base, cand, n: ratios.length, median_ratio: r3(m), ci95: [r3(lo), r3(hi)], rounds_cand_faster: `${wins}/${ratios.length}`, significant, verdict: significant ? `SIGNIFICANT ${m > 1 ? '+' : ''}${r3((m - 1) * 100)}%` : 'within noise' }
}

const out = {
  schema: 'camelid.bench-owner-sweep-stats/v1',
  model: label, rounds: rounds.length,
  per_config_median_prefill_tok_s: perConfigMedian,
  paired: [
    compare('off', 'owner_avx2'),
    compare('off', 'owner_vnni4x4'),
    compare('off', 'owner_vnni4x8'),
    compare('owner_vnni4x4', 'owner_vnni4x8'),
  ],
}
console.log(JSON.stringify(out, null, 2))
