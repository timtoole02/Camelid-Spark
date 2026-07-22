#!/usr/bin/env node
// check-ledger-schema.mjs — CAIRN Phase 1.
// Validates camelid.ledger/v1 documents against ledger/camelid-ledger.schema.json
// with a zero-dependency JSON Schema subset validator (the repo has no root
// package.json; check-*.mjs scripts are pure node), plus two ledger-specific
// invariants:
//   1. code-enum coverage — the schema's status / support_scope /
//      full_support_status / latest_checked_result enums must be a SUPERSET of
//      the literal values in src/api/mod.rs, so nothing the contract emits is
//      unexpressible (CAIRN Phase 1 "zero loss / anything it cannot express is
//      reported"). This also front-runs Phase 4 drift: a new code enum fails CI.
//   2. identity.id === contract.id on every model row.
//
// Usage: node scripts/check-ledger-schema.mjs [ledger.json ...]
// With no args: validates ledger/examples/*.json and, if present,
// ledger/camelid-ledger.json.
import { readFile, readdir, access } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

const ROOT = resolve(process.argv[2] === '--root' ? process.argv[3] : '.')
const SCHEMA_PATH = join(ROOT, 'ledger', 'camelid-ledger.schema.json')
const MODRS_PATH = join(ROOT, 'src', 'api', 'mod.rs')
const failures = []
const fail = (msg) => failures.push(msg)

// ---------------------------------------------------------------------------
// zero-dependency JSON Schema subset validator
// supports: $ref(#/definitions/*), type (incl. array-of-types + integer/null),
// const, enum, required, properties, additionalProperties:false, items,
// pattern, minimum.
// ---------------------------------------------------------------------------
function jsonType(v) {
  if (v === null) return 'null'
  if (Array.isArray(v)) return 'array'
  return typeof v // 'string' | 'number' | 'boolean' | 'object'
}
function typeMatches(v, t) {
  if (t === 'integer') return typeof v === 'number' && Number.isInteger(v)
  if (t === 'number') return typeof v === 'number'
  if (t === 'object') return jsonType(v) === 'object'
  return jsonType(v) === t
}
function deref(schema, root) {
  let s = schema
  const seen = new Set()
  while (s && typeof s.$ref === 'string') {
    if (seen.has(s.$ref)) throw new Error(`circular $ref ${s.$ref}`)
    seen.add(s.$ref)
    const m = /^#\/definitions\/(.+)$/.exec(s.$ref)
    if (!m) throw new Error(`unsupported $ref ${s.$ref}`)
    s = root.definitions?.[m[1]]
    if (!s) throw new Error(`missing definition ${m[1]}`)
  }
  return s
}
function validate(value, schema, root, path, errors) {
  schema = deref(schema, root)
  if ('const' in schema && JSON.stringify(value) !== JSON.stringify(schema.const)) {
    errors.push(`${path}: expected const ${JSON.stringify(schema.const)}, got ${JSON.stringify(value)}`)
    return
  }
  if (schema.type) {
    const types = Array.isArray(schema.type) ? schema.type : [schema.type]
    if (!types.some((t) => typeMatches(value, t))) {
      errors.push(`${path}: expected type ${types.join('|')}, got ${jsonType(value)}`)
      return
    }
  }
  if (schema.enum && !schema.enum.some((e) => JSON.stringify(e) === JSON.stringify(value))) {
    errors.push(`${path}: value ${JSON.stringify(value)} not in enum`)
  }
  if (typeof value === 'string' && schema.pattern && !new RegExp(schema.pattern).test(value)) {
    errors.push(`${path}: string does not match /${schema.pattern}/`)
  }
  if (typeof value === 'number' && typeof schema.minimum === 'number' && value < schema.minimum) {
    errors.push(`${path}: ${value} < minimum ${schema.minimum}`)
  }
  if (jsonType(value) === 'object') {
    for (const req of schema.required || []) {
      if (!(req in value)) errors.push(`${path}: missing required property "${req}"`)
    }
    const props = schema.properties || {}
    if (schema.additionalProperties === false) {
      for (const key of Object.keys(value)) {
        if (!(key in props)) errors.push(`${path}: unexpected property "${key}" (additionalProperties:false)`)
      }
    }
    for (const [key, sub] of Object.entries(props)) {
      if (key in value) validate(value[key], sub, root, `${path}.${key}`, errors)
    }
  }
  if (jsonType(value) === 'array' && schema.items) {
    value.forEach((el, i) => validate(el, schema.items, root, `${path}[${i}]`, errors))
  }
}

function validateLedger(data, schema, label) {
  const errors = []
  validate(data, schema, schema, label, errors)
  // invariant 2: identity.id === contract.id
  for (const [i, row] of (data.model_rows || []).entries()) {
    if (row?.identity?.id && row?.contract?.id && row.identity.id !== row.contract.id) {
      errors.push(`${label}.model_rows[${i}]: identity.id "${row.identity.id}" !== contract.id "${row.contract.id}"`)
    }
  }
  return errors
}

// ---------------------------------------------------------------------------
// invariant 1: code-enum coverage (schema enum ⊇ live src/api/mod.rs values)
// ---------------------------------------------------------------------------
function extractCodeEnum(src, field) {
  const re = new RegExp(`\\b${field}:\\s*"([a-z0-9_]+)"`, 'g')
  const out = new Set()
  let m
  while ((m = re.exec(src))) out.add(m[1])
  return out
}
function checkCoverage(schema, src) {
  const pairs = [
    ['status', 'statusVocabulary'],
    ['support_scope', 'supportScopeVocabulary'],
    ['full_support_status', 'fullSupportStatusVocabulary'],
    ['latest_checked_result', 'latestCheckedResultVocabulary'],
  ]
  for (const [field, defName] of pairs) {
    const schemaEnum = new Set(schema.definitions?.[defName]?.enum || [])
    if (!schemaEnum.size) { fail(`schema definition ${defName} has no enum`); continue }
    const codeValues = extractCodeEnum(src, field)
    if (!codeValues.size) { fail(`no \`${field}:\` literals found in src/api/mod.rs (coverage check cannot run)`); continue }
    const missing = [...codeValues].filter((v) => !schemaEnum.has(v)).sort()
    if (missing.length) {
      fail(`schema ${defName} is not a superset of code \`${field}:\` values — missing: ${missing.join(', ')}`)
    } else {
      console.log(`  coverage ok: ${defName} ⊇ ${codeValues.size} live \`${field}:\` value(s) from src/api/mod.rs`)
    }
  }
}

// ---------------------------------------------------------------------------
async function exists(p) { try { await access(p); return true } catch { return false } }

async function main() {
  const schema = JSON.parse(await readFile(SCHEMA_PATH, 'utf8'))
  if (schema.$id !== 'camelid.ledger/v1') fail(`schema $id is "${schema.$id}", expected "camelid.ledger/v1"`)

  // coverage against the code (source of truth)
  if (await exists(MODRS_PATH)) {
    checkCoverage(schema, await readFile(MODRS_PATH, 'utf8'))
  } else {
    fail(`src/api/mod.rs not found at ${MODRS_PATH}; cannot run code-enum coverage`)
  }

  // pick the ledgers to validate
  let targets = process.argv.slice(2).filter((a) => a !== '--root' && a !== ROOT && a.endsWith('.json'))
  if (!targets.length) {
    const exDir = join(ROOT, 'ledger', 'examples')
    if (await exists(exDir)) {
      for (const f of await readdir(exDir)) if (f.endsWith('.json')) targets.push(join(exDir, f))
    }
    const authoritative = join(ROOT, 'ledger', 'camelid-ledger.json')
    if (await exists(authoritative)) targets.push(authoritative)
  }
  if (!targets.length) fail('no ledger documents found to validate (expected ledger/examples/*.json or ledger/camelid-ledger.json)')

  for (const t of targets) {
    let data
    try { data = JSON.parse(await readFile(t, 'utf8')) }
    catch (e) { fail(`${t}: not valid JSON — ${e.message}`); continue }
    const errors = validateLedger(data, schema, t)
    if (errors.length) errors.forEach((e) => fail(e))
    else console.log(`  valid: ${t} (${(data.model_rows || []).length} model row(s))`)
  }

  if (failures.length) {
    console.error(`\nledger schema check FAILED (${failures.length}):`)
    for (const f of failures) console.error(`  - ${f}`)
    process.exit(1)
  }
  console.log(`\nledger schema check passed: ${targets.length} document(s), coverage ⊇ code enums`)
}

export { validate, validateLedger, extractCodeEnum, deref }

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  main().catch((e) => { console.error(e); process.exit(1) })
}
