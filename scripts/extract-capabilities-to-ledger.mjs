#!/usr/bin/env node
// extract-capabilities-to-ledger.mjs — CAIRN Phase 2, ONE-TIME bootstrap.
//
// Parses the static `CapabilitiesResponse { ... }` literal that
// capabilities_response_with_plan() builds in src/api/mod.rs and emits
// ledger/camelid-ledger.json (camelid.ledger/v1). The capability structs derive
// plain serde `Serialize` with no renames, and every field value is a
// single-line literal with no escaped quotes, so the parsed field names + values
// are byte-identical to what /api/capabilities serves — no build, no server, no
// model load (this box is memory-constrained; see the bench-safety rules).
//
// This is a bootstrap tool: once the Phase 3 generator exists it runs the OTHER
// way (ledger -> capabilities), and this script is retired. The output is
// validated by scripts/check-ledger-schema.mjs.
import { readFile, writeFile, mkdir, access } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { execSync } from 'node:child_process'

const ROOT = resolve('.')
const OUT = join(ROOT, 'ledger', 'camelid-ledger.json')

// --- extract the balanced CapabilitiesResponse { ... } block ---------------
function balancedBlock(src, openIdx, open = '{', close = '}') {
  let depth = 0, inStr = false
  for (let i = openIdx; i < src.length; i++) {
    const c = src[i]
    if (inStr) { if (c === '"') inStr = false; continue }
    if (c === '"') { inStr = true; continue }
    if (c === open) depth++
    else if (c === close) { depth--; if (depth === 0) return src.slice(openIdx, i + 1) }
  }
  throw new Error('unbalanced block')
}

// --- tokenizer for the Rust-literal subset ---------------------------------
function tokenize(src) {
  const toks = []
  let i = 0
  while (i < src.length) {
    const c = src[i]
    if (c === ' ' || c === '\n' || c === '\r' || c === '\t') { i++; continue }
    if (c === '/' && src[i + 1] === '/') { while (i < src.length && src[i] !== '\n') i++; continue }
    if (c === '"') { // no escaped quotes in this block (verified)
      let j = i + 1, s = ''
      while (j < src.length && src[j] !== '"') { s += src[j]; j++ }
      toks.push({ t: 'str', v: s }); i = j + 1; continue
    }
    if (c >= '0' && c <= '9') { let j = i; while (j < src.length && src[j] >= '0' && src[j] <= '9') j++; toks.push({ t: 'num', v: Number(src.slice(i, j)) }); i = j; continue }
    if (/[A-Za-z_]/.test(c)) { let j = i; while (j < src.length && /[A-Za-z0-9_]/.test(src[j])) j++; toks.push({ t: 'ident', v: src.slice(i, j) }); i = j; continue }
    if ('{}[](),:!&'.includes(c)) { toks.push({ t: c }); i++; continue }
    i++ // skip anything else (`.`, etc.)
  }
  return toks
}

// --- recursive-descent parser ----------------------------------------------
function parse(toks) {
  let p = 0
  const peek = () => toks[p]
  function value() {
    const t = toks[p]
    if (t.t === 'str') { p++; return t.v }
    if (t.t === 'num') { p++; return t.v }
    if (t.t === '&') { p++; return value() }
    if (t.t === '[') return array()
    if (t.t === '{') return object()
    if (t.t === 'ident') {
      if (t.v === 'true' || t.v === 'false') { p++; return t.v === 'true' }
      if (t.v === 'vec') { p++; if (peek()?.t === '!') p++; return array() }
      if (toks[p + 1]?.t === '{') return object() // TypeName { ... }
      p++; return { __ident: t.v } // bare identifier (field-init shorthand target)
    }
    p++; return null
  }
  function array() {
    if (peek().t !== '[') throw new Error('expected [')
    p++; const arr = []
    while (peek() && peek().t !== ']') { arr.push(value()); if (peek()?.t === ',') p++ }
    p++; return arr
  }
  function object() {
    if (peek()?.t === 'ident') p++ // skip TypeName prefix
    if (peek().t !== '{') throw new Error('expected {')
    p++; const obj = {}
    while (peek() && peek().t !== '}') {
      const key = peek().v; p++
      if (peek().t === ',' || peek().t === '}') { obj[key] = { __shorthand: key }; if (peek().t === ',') p++; continue }
      if (peek().t === ':') p++
      obj[key] = value()
      if (peek()?.t === ',') p++
    }
    p++; return obj
  }
  return value()
}

// --- helpers ---------------------------------------------------------------
const GGUF_RE = /([A-Za-z0-9][A-Za-z0-9._-]*\.gguf)/
const SHA_RE = /\b([a-f0-9]{64})\b/
const RECEIPT_RE = /qa\/evidence-bundles\/[A-Za-z0-9._/-]+?\/(?:manifest\.json|SHA256SUMS)/g

async function exists(p) { try { await access(p); return true } catch { return false } }

export async function buildLedger(root = ROOT) {
  const src = await readFile(join(root, 'src', 'api', 'mod.rs'), 'utf8')
  const marker = 'CapabilitiesResponse {'
  const fnPos = src.indexOf('fn capabilities_response_with_plan')
  // The fn signature ends `-> CapabilitiesResponse {` (the fn body brace); the
  // struct literal `CapabilitiesResponse {` is the NEXT occurrence (the return
  // expression) — that is the one we parse.
  const sigIdx = src.indexOf(marker, fnPos)
  const idx = src.indexOf(marker, sigIdx + marker.length)
  if (idx < 0) throw new Error('CapabilitiesResponse struct literal not found')
  const block = balancedBlock(src, src.indexOf('{', idx))
  const cr = parse(tokenize(block))

  // execution_plan is the function param (None) -> null in the static contract
  cr.execution_plan = null

  const rows = cr.model_compatibility
  if (!Array.isArray(rows)) throw new Error('model_compatibility did not parse to an array')

  const receiptWarnings = []
  const model_rows = []
  for (const contract of rows) {
    const prose = [contract.evidence, contract.frontend_readiness_gate, contract.full_support_blockers, contract.tested_context].join(' ')
    const identity = { id: contract.id, family: contract.family, quantization: contract.quantization }
    const gguf = GGUF_RE.exec(prose)
    if (gguf) identity.gguf_filename = gguf[1]
    const sha = SHA_RE.exec(prose)
    if (sha) identity.sha256 = sha[1]

    const receipts = []
    const seen = new Set()
    for (const m of prose.matchAll(RECEIPT_RE)) {
      const path = m[0]
      if (seen.has(path)) continue
      seen.add(path)
      if (await exists(join(root, path))) receipts.push({ path })
      else receiptWarnings.push(`${contract.id}: receipt ${path} does not resolve on disk (omitted)`)
    }
    const row = { identity, contract }
    if (receipts.length) row.receipts = receipts
    model_rows.push(row)
  }

  const capabilities = {
    engine: cr.engine,
    gguf_metadata: cr.gguf_metadata,
    tensor_loading: cr.tensor_loading,
    tokenization: cr.tokenization,
    inference: cr.inference,
    streaming: cr.streaming,
    model_downloads: cr.model_downloads,
    hf_catalog_install: cr.hf_catalog_install,
    execution_plan: null,
    support_contract: cr.support_contract,
    supported_quantization: cr.supported_quantization,
    planned_quantization: cr.planned_quantization,
    supported_model_families: cr.supported_model_families,
    planned_model_families: cr.planned_model_families,
    api_features: cr.api_features,
    notes: cr.notes,
  }

  let head = 'unknown'
  try { head = execSync('git rev-parse --short HEAD', { cwd: root }).toString().trim() } catch {}

  const ledger = {
    ledger_version: 'camelid.ledger/v1',
    provenance: {
      source_head: head,
      note: 'Derived from the static CapabilitiesResponse literal in src/api/mod.rs by scripts/extract-capabilities-to-ledger.mjs. Contract fields are byte-faithful (plain serde Serialize, no renames). Per CAIRN Amendment 1, the CODE is the contract source of truth; this ledger is its derived canonical form, and scripts/check-ledger-drift.mjs re-derives from code and fails CI if code and ledger disagree (provenance excluded).',
    },
    capabilities,
    model_rows,
  }

  return { ledger, receiptWarnings, fieldCount: Object.keys(rows[0]).length }
}

async function main() {
  const { ledger, receiptWarnings, fieldCount } = await buildLedger(ROOT)
  await mkdir(join(ROOT, 'ledger'), { recursive: true })
  await writeFile(OUT, JSON.stringify(ledger, null, 2) + '\n')
  const c = ledger.capabilities
  console.log(`extracted ${ledger.model_rows.length} model row(s); each contract has ${fieldCount} fields`)
  console.log(`envelope: ${c.supported_quantization.length} supported_quant, ${c.planned_quantization.length} planned_quant, ${c.supported_model_families.length} supported_fam, ${c.planned_model_families.length} planned_fam, ${c.api_features.length} api_features, ${c.notes.length} notes`)
  console.log(`receipts attached: ${ledger.model_rows.reduce((n, r) => n + (r.receipts?.length || 0), 0)} (resolved on disk)`)
  if (receiptWarnings.length) { console.log('receipt notes:'); receiptWarnings.forEach((w) => console.log('  - ' + w)) }
  console.log(`wrote ${OUT}`)
}

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  main().catch((e) => { console.error(e); process.exit(1) })
}
