#!/usr/bin/env node
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { join } from 'node:path'

const repoRoot = new URL('..', import.meta.url)
const readme = readFileSync(new URL('../README.md', import.meta.url), 'utf8')
const expectedAsset = 'docs/assets/camelid-readme-chat-surface-dark.png'
const retiredLightAsset = 'docs/assets/ui-screenshot-v2.png'
const expectedSha256 = '2576d003da76fa5a4e32462f9922555daec2cd1b5a88ccb392e125931baba418'

assert.match(
  readme,
  new RegExp(`!\\[Camelid WebUI chat surface\\]\\(${expectedAsset.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\)`),
  'README must use the approved dark collapsed-rail Camelid chat screenshot',
)
assert.doesNotMatch(
  readme,
  new RegExp(retiredLightAsset.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')),
  'README must not point at the retired light WebUI screenshot',
)
assert.match(
  readme,
  /dark, collapsed-rail chat surface/i,
  'README caption must preserve the intended dark collapsed-rail screenshot contract',
)

const assetBytes = readFileSync(join(fileURLToPath(repoRoot), expectedAsset))
const actualSha256 = createHash('sha256').update(assetBytes).digest('hex')
assert.equal(
  actualSha256,
  expectedSha256,
  'approved README screenshot bytes changed; update this guard only with explicit product approval',
)

console.log('README screenshot guard passed')
