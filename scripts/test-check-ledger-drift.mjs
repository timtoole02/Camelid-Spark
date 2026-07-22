#!/usr/bin/env node
// Self-test for check-ledger-drift.mjs (run in CI by the validation-scripts
// test-*.mjs glob). Exercises the pure helpers that back the two drift checks.
import assert from 'node:assert/strict'
import { firstDiff, canon, norm, fillerKey, parseTable, isSupported, shaFindings } from './check-ledger-drift.mjs'

// --- firstDiff / canon (freshness engine) ---
assert.equal(firstDiff({ a: 1, b: [1, 2] }, { b: [1, 2], a: 1 }), null, 'key order must not matter after canon-free deep compare')
assert.match(firstDiff({ s: 'supported_exact_row_smoke' }, { s: 'active_validation_unsupported' }), /\.s \(/, 'a status change must be located')
assert.match(firstDiff({ rows: [{ id: 'a' }] }, { rows: [{ id: 'a' }, { id: 'b' }] }), /rows\.length/, 'a row-count change must be located')
// canon makes key order irrelevant
assert.equal(JSON.stringify(canon({ b: 1, a: 2 })), JSON.stringify({ a: 2, b: 1 }))

// --- isSupported (the support predicate) ---
assert.equal(isSupported('supported_exact_row_smoke'), true)
assert.equal(isSupported('supported'), true)
assert.equal(isSupported('active_validation_partial_runtime'), false)
assert.equal(isSupported('planned_exact_row_candidate'), false)
assert.equal(isSupported('unsupported'), false)

// --- label -> id mapping (the non-contradiction key) ---
// display labels that omit filler tokens must still map to the underscore ids
assert.equal(fillerKey('Qwen3 1.7B Q8_0'), fillerKey('qwen3_1_7b_instruct_q8_0'), 'Qwen label maps to its instruct id')
assert.equal(fillerKey('Gemma 4 26B-A4B-It QAT Q4_0'), fillerKey('gemma4_26b_a4b_it_q4_0'), 'QAT/It fillers + underscores align')
assert.equal(norm('Mistral-7B-Instruct-v0.3.Q8_0.gguf'.replace(/\.gguf$/, '')), norm('mistral_7b_instruct_v0_3_q8_0'), 'filename (minus .gguf) normalizes to the id')
// distinct rows must NOT collide
assert.notEqual(fillerKey('Qwen3 4B Q8_0'), fillerKey('qwen3_4b_q4_k_m'), 'different quants stay distinct')

// --- parseTable (surface table extraction) ---
const md = [
  'intro',
  '| Model row | Quant | Serve lane | Evidence |',
  '| --- | --- | --- | --- |',
  '| TinyLlama 1.1B Chat | Q8_0 | single-node | gate |',
  '| Llama 3.2 3B Instruct | Q8_0 | single-node | smoke |',
  '',
  'after',
].join('\n')
const rows = parseTable(md, /^\|\s*Model row\s*\|\s*Quant\s*\|/)
assert.equal(rows.length, 2, 'parseTable stops at the blank line')
assert.equal(rows[0][0], 'TinyLlama 1.1B Chat')
assert.equal(rows[1][1], 'Q8_0')
assert.equal(parseTable(md, /^\|\s*Nope\s*\|/), null, 'missing header returns null')

// --- shaFindings (Phase 4 sha256 cross-surface agreement) ---
const SHA_A = '6c1a2b411610'.padEnd(64, '0')
const SHA_B = 'abcd'.padEnd(64, '0')
const canonMap = new Map([['Model-A.gguf', { sha: SHA_A, id: 'model_a' }]])
const known = new Set([SHA_A])
// correct sha next to the file -> verified, no finding
{
  const r = shaFindings('X', `the Model-A.gguf row (sha256 ${SHA_A})`, canonMap, known)
  assert.equal(r.verified, 1); assert.equal(r.findings.length, 0)
}
// a wrong/unknown sha next to the file -> one finding
{
  const r = shaFindings('DOC', `Model-A.gguf sha256 ${'dead'.padEnd(64, '0')}`, canonMap, known)
  assert.equal(r.findings.length, 1, 'a stale/wrong sha for an anchored file must be caught')
  assert.match(r.findings[0], /Model-A\.gguf/)
}
// another known file's sha co-occurring on the same line -> NOT flagged (no false positive)
{
  const two = new Map([...canonMap, ['Other.gguf', { sha: SHA_B, id: 'other' }]])
  const r = shaFindings('Y', `Model-A.gguf mentioned near Other.gguf sha256 ${SHA_B}`, two, new Set([SHA_A, SHA_B]))
  assert.equal(r.findings.length, 0, "another file's sha on the same line must not be flagged for the wrong file")
}

console.log('test-check-ledger-drift: all checks passed')
