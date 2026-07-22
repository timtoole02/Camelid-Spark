#!/usr/bin/env node
/* Assemble a Gemma 4 exact-row evidence bundle from raw run logs.
   Usage:
     node scripts/build-gemma4-evidence-bundle.mjs <row_id> <model_path> \
       --log name=path [--log name=path ...] \
       [--field key=value ...]

   Writes qa/evidence-bundles/gemma4-<row>-<UTC>-head-<sha>/{manifest.json,
   SHA256SUMS, logs/...}. The manifest records the exact model file, SHA256,
   oracle build, prompt pack + oracle artifact paths, git head, and every raw
   log it was assembled from. Claims live in the logs; the manifest only
   indexes them. */

import { createHash } from 'node:crypto'
import { execSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

const [rowId, modelPath, ...rest] = process.argv.slice(2)
if (!rowId || !modelPath) {
  console.error('usage: build-gemma4-evidence-bundle.mjs <row_id> <model_path> --log name=path ... [--field k=v ...]')
  process.exit(2)
}

const logs = []
const fields = {}
for (let i = 0; i < rest.length; i += 2) {
  const [flag, kv] = [rest[i], rest[i + 1]]
  const eq = kv.indexOf('=')
  const key = kv.slice(0, eq)
  const value = kv.slice(eq + 1)
  if (flag === '--log') logs.push({ name: key, path: value })
  else if (flag === '--field') fields[key] = value
}

const repo = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..')
const head = execSync('git rev-parse --short=12 HEAD', { cwd: repo }).toString().trim()
const dirty = execSync('git status --porcelain', { cwd: repo }).toString().trim().length > 0
const utc = new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, 'Z')
const bundleName = `gemma4-${rowId.replace(/_/g, '-')}-${utc}-head-${head}`
const bundleDir = path.join(repo, 'qa', 'evidence-bundles', bundleName)
fs.mkdirSync(path.join(bundleDir, 'logs'), { recursive: true })

function sha256(file) {
  // Stream in 64 MiB chunks — model files exceed Node's 2 GiB readFileSync cap.
  const h = createHash('sha256')
  const fd = fs.openSync(file, 'r')
  const buf = Buffer.alloc(64 * 1024 * 1024)
  let read
  while ((read = fs.readSync(fd, buf, 0, buf.length, null)) > 0) {
    h.update(buf.subarray(0, read))
  }
  fs.closeSync(fd)
  return h.digest('hex')
}

console.error(`hashing model ${modelPath} ...`)
const modelSha = sha256(modelPath)
const modelBytes = fs.statSync(modelPath).size

const logEntries = []
for (const { name, path: src } of logs) {
  const dst = path.join(bundleDir, 'logs', `${name}.txt`)
  fs.copyFileSync(src, dst)
  logEntries.push({ name, file: `logs/${name}.txt`, sha256: sha256(dst) })
}

const manifest = {
  schema: 'camelid.gemma4_exact_row_public_evidence.v1',
  generated_utc: new Date().toISOString(),
  git_head: head,
  checkout_clean: !dirty,
  row_id: rowId,
  model: {
    filename: path.basename(modelPath),
    size_bytes: modelBytes,
    sha256: modelSha,
    quantization: 'Q8_0',
    architecture: 'gemma4',
  },
  oracle: {
    build: 'llama.cpp 5d56eff (llama-server, CPU, temperature=0/top_k=1, cache_prompt=false)',
    prompt_pack: 'qa/gemma4/prompt_packs/basic_v1.json',
    oracle_artifact: `qa/gemma4/oracle/${path.basename(modelPath).replace(/\.gguf$/, '')}.basic_v1.json`,
  },
  logs: logEntries,
  fields,
  claim_boundary:
    'Exact-row text-token generation within the checked basic_v1 prompt pack and API smoke envelope only. ' +
    'No bounded-context promotion, no performance claim beyond the raw logs, no multimodal input ' +
    '(fails closed with a typed error), no neighboring-row or Gemma-family support, no shared-memory claim ' +
    '(distributed mode is layer sharding).',
}

fs.writeFileSync(path.join(bundleDir, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n')

const sums = []
for (const f of ['manifest.json', ...logEntries.map((l) => l.file)]) {
  sums.push(`${sha256(path.join(bundleDir, f))}  ${f}`)
}
fs.writeFileSync(path.join(bundleDir, 'SHA256SUMS'), sums.join('\n') + '\n')
console.log(bundleDir)
