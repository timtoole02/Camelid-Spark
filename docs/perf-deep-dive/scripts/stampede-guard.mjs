#!/usr/bin/env node
// STAMPEDE P0.3 regression guard: compare a candidate cpu-baseline-medN.mjs receipt
// against a pinned baseline receipt. A phase lands ONLY with a passing guard run attached.
//
// Usage: node stampede-guard.mjs <baseline.json> <candidate.json> [--tolerance 0.03]
//
// FAIL conditions:
//   - candidate camelid decode or prefill median regresses > tolerance vs baseline
//   - candidate camelid-vs-camelid parity: baseline and candidate greedy decode TEXT differ
//     (same prompt, temp=0 — byte-identical contract; llama text is informational only)
// Exit 0 = PASS, 1 = FAIL, 2 = usage/parse error.
import { readFile } from 'node:fs/promises'

const args = process.argv.slice(2)
const tolIx = args.indexOf('--tolerance')
const TOL = tolIx >= 0 ? parseFloat(args[tolIx + 1]) : 0.03
const files = args.filter((a, i) => !a.startsWith('--') && (tolIx < 0 || (i !== tolIx && i !== tolIx + 1)))
if (files.length !== 2 || !Number.isFinite(TOL)) {
  console.error('usage: stampede-guard.mjs <baseline.json> <candidate.json> [--tolerance 0.03]')
  process.exit(2)
}
const [base, cand] = await Promise.all(files.map(async f => JSON.parse(await readFile(f, 'utf8'))))

const checks = []
function checkMetric(name, b, c) {
  if (!Number.isFinite(b) || !Number.isFinite(c)) {
    checks.push({ name, status: 'SKIP', note: `missing (${b} vs ${c})` })
    return
  }
  const delta = (c - b) / b
  const pass = delta >= -TOL
  checks.push({ name, baseline: b, candidate: c, delta_pct: Math.round(delta * 1000) / 10, status: pass ? 'PASS' : 'FAIL' })
}
checkMetric('camelid_decode_tok_s', base.median?.camelid_decode_tok_s, cand.median?.camelid_decode_tok_s)
checkMetric('camelid_prefill_tok_s', base.median?.camelid_prefill_tok_s, cand.median?.camelid_prefill_tok_s)

const bTxt = base.parity?.camelid_text
const cTxt = cand.parity?.camelid_text
if (typeof bTxt === 'string' && typeof cTxt === 'string') {
  checks.push({
    name: 'camelid_greedy_text_identical', status: bTxt === cTxt ? 'PASS' : 'FAIL',
    note: bTxt === cTxt ? 'byte-identical' : `diverges at char ${[...bTxt].findIndex((ch, i) => ch !== cTxt[i])}`,
  })
} else {
  checks.push({ name: 'camelid_greedy_text_identical', status: 'SKIP', note: 'parity text missing in a receipt' })
}

const failed = checks.some(c => c.status === 'FAIL')
// Receipts are privacy-scrubbed before commit (<home> placeholder), so compare
// the model FILENAME, not the raw path (either separator style).
const modelName = p => (p ?? '').split(/[\\/]/).pop()
const report = {
  schema: 'camelid.stampede-guard/v1',
  baseline: files[0], candidate: files[1], tolerance: TOL,
  baseline_model: base.model, candidate_model: cand.model,
  model_match: modelName(base.model) === modelName(cand.model),
  checks, verdict: failed ? 'FAIL' : 'PASS',
}
if (!report.model_match) { report.verdict = 'FAIL'; report.note = 'baseline and candidate measured different models' }
console.log(JSON.stringify(report, null, 2))
process.exit(report.verdict === 'PASS' ? 0 : 1)
