#!/usr/bin/env node
// Self-test for check-ledger-schema.mjs (run in CI by the validation-scripts
// job's test-*.mjs glob). Exercises the zero-dep validator against the real
// example ledger plus a battery of corruptions.
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { resolve, dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { validateLedger, extractCodeEnum } from './check-ledger-schema.mjs'

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const schema = JSON.parse(await readFile(join(repoRoot, 'ledger', 'camelid-ledger.schema.json'), 'utf8'))
const example = JSON.parse(await readFile(join(repoRoot, 'ledger', 'examples', 'example-ledger.json'), 'utf8'))

const clone = (o) => structuredClone(o)
const errsFor = (mutate) => { const d = clone(example); mutate(d); return validateLedger(d, schema, 'test') }

// baseline: the committed example is valid
assert.deepEqual(validateLedger(example, schema, 'example'), [], 'the committed example ledger must validate with zero errors')

// bad status enum value
assert.ok(
  errsFor((d) => { d.model_rows[0].contract.status = 'totally_bogus_status' }).some((e) => /status.*not in enum/.test(e)),
  'an out-of-vocabulary status must be rejected',
)

// missing required contract field
assert.ok(
  errsFor((d) => { delete d.model_rows[0].contract.evidence }).some((e) => /missing required property "evidence"/.test(e)),
  'a missing required contract field must be reported',
)

// identity.id / contract.id mismatch
assert.ok(
  errsFor((d) => { d.model_rows[0].identity.id = 'not_the_contract_id' }).some((e) => /identity\.id.*!==.*contract\.id/.test(e)),
  'identity.id must equal contract.id',
)

// additionalProperties:false on contract
assert.ok(
  errsFor((d) => { d.model_rows[0].contract.surprise = 'x' }).some((e) => /unexpected property "surprise"/.test(e)),
  'an unknown contract field must be rejected (additionalProperties:false)',
)

// wrong type: window must be integer
assert.ok(
  errsFor((d) => { d.model_rows[0].contract.bounded_context_window = '512' }).some((e) => /expected type integer/.test(e)),
  'a stringified integer window must be rejected',
)

// sha256 pattern
assert.ok(
  errsFor((d) => { d.model_rows[0].identity.sha256 = 'not-a-real-hash' }).some((e) => /does not match/.test(e)),
  'a malformed sha256 must be rejected by pattern',
)

// bad top-level const
assert.ok(
  errsFor((d) => { d.ledger_version = 'camelid.ledger/v2' }).some((e) => /expected const/.test(e)),
  'the ledger_version const must be enforced',
)

// extractCodeEnum pulls literal values from a source string
const enums = extractCodeEnum('status: "supported_exact_row_smoke", x, status: "planned"', 'status')
assert.deepEqual([...enums].sort(), ['planned', 'supported_exact_row_smoke'], 'extractCodeEnum must find all literal values')

console.log('test-check-ledger-schema: all checks passed')
