#!/usr/bin/env node
/* Assemble a Gemma 4 bounded-context (512..8192) evidence bundle from the raw
   `gemma4_generation_parity` run logs.

   Each run is `cargo test --release --test gemma4_generation_parity
   gemma4_greedy_generation_matches_llama_cpp_oracle` with
   CAMELID_GEMMA4_GGUF=<model> and CAMELID_GEMMA4_PACK=context_<W>_v1. That test
   asserts, per prompt, that camelid's greedy generation matches the committed
   pinned-llama.cpp oracle on ALL THREE of: prompt token ids, generated token
   ids, and generated text — so a PASS ("test result: ok. 1 passed") is proof of
   full three-way parity for that (row, context window).

   Usage:
     node scripts/build-gemma4-context-bundle.mjs \
       --logs-dir <dir with {E2B,E4B}.context_<W>_v1.log> \
       [--model-dir <dir with the GGUFs, default $CAMELID_MODEL_DIR or ./models>]

   Writes qa/evidence-bundles/gemma4-e2b-e4b-context-512-8192-<UTC>-head-<sha>/
   {manifest.json, SHA256SUMS, logs/...}. Claims live in the logs + committed
   oracle; the manifest indexes them and records the per-(row,window) pass.

   NOTE: the repo .gitignore has `*.log`, so commit the bundle logs with
   `git add -f <bundle>/logs/*.log` (same convention as existing bundles). */

import { createHash } from 'node:crypto'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import fs from 'node:fs'
import path from 'node:path'

const args = process.argv.slice(2)
function optArg(flag, fallback) {
  const i = args.indexOf(flag)
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback
}
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const logsDir = optArg('--logs-dir', null)
if (!logsDir) {
  console.error('usage: build-gemma4-context-bundle.mjs --logs-dir <dir> [--model-dir <dir>]')
  process.exit(2)
}
const modelDir = optArg('--model-dir', process.env.CAMELID_MODEL_DIR || path.join(repo, 'models'))

const WINDOWS = [512, 1024, 2048, 4096, 8192]
const ROWS = [
  { row_id: 'gemma4_e2b_it_q8_0', display_name: 'Gemma 4 E2B-it Q8_0', log_key: 'E2B', model_file: 'gemma-4-E2B-it-Q8_0.gguf' },
  { row_id: 'gemma4_e4b_it_q8_0', display_name: 'Gemma 4 E4B-it Q8_0', log_key: 'E4B', model_file: 'gemma-4-E4B-it-Q8_0.gguf' },
]

function sha256(file) {
  const h = createHash('sha256')
  const fd = fs.openSync(file, 'r')
  const buf = Buffer.alloc(64 * 1024 * 1024)
  let read
  while ((read = fs.readSync(fd, buf, 0, buf.length, null)) > 0) h.update(buf.subarray(0, read))
  fs.closeSync(fd)
  return h.digest('hex')
}

const head = execFileSync('git', ['rev-parse', '--short=12', 'HEAD'], { cwd: repo }).toString().trim()
const dirty = execFileSync('git', ['status', '--porcelain'], { cwd: repo }).toString().trim().length > 0
const utc = new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, 'Z')
const bundleName = `gemma4-e2b-e4b-context-512-8192-${utc}-head-${head}`
const bundleDir = path.join(repo, 'qa', 'evidence-bundles', bundleName)
fs.mkdirSync(path.join(bundleDir, 'logs'), { recursive: true })

const modelSha = {}
const modelBytes = {}
for (const row of ROWS) {
  const mp = path.join(modelDir, row.model_file)
  console.error(`hashing ${mp} ...`)
  modelSha[row.row_id] = sha256(mp)
  modelBytes[row.row_id] = fs.statSync(mp).size
}

const rows = []
let allPass = true
for (const row of ROWS) {
  for (const w of WINDOWS) {
    const packStem = `context_${w}_v1`
    const logSrc = path.join(logsDir, `${row.log_key}.${packStem}.log`)
    const log = fs.existsSync(logSrc) ? fs.readFileSync(logSrc, 'utf8') : ''
    const passed = /test result: ok\. 1 passed/.test(log) && new RegExp(`recall-${w} OK`).test(log)
    const genMatch = log.match(new RegExp(`recall-${w} OK \\((\\d+) tokens\\): "([\\s\\S]*?)"\\n`))
    const pack = JSON.parse(
      fs.readFileSync(path.join(repo, 'qa', 'gemma4', 'prompt_packs', `${packStem}.json`), 'utf8'),
    )
    const logName = `${row.log_key}.${packStem}.log`
    // Public-bundle privacy: redact operator home/user paths from raw logs (the
    // first run captures a `Compiling ... (<home>\repo)` line). Evidence lines
    // (recall-* / test result) carry no paths and are untouched.
    if (log) {
      const scrubbed = log
        .replace(/[A-Za-z]:[\\/]Users[\\/][^\\/\s)]+/g, '<home>')
        .replace(/\/(?:c\/)?Users\/[^/\s)]+/g, '<home>')
        .replace(/\/home\/[^/\s)]+/g, '<home>')
      fs.writeFileSync(path.join(bundleDir, 'logs', logName), scrubbed)
    }
    if (!passed) allPass = false
    rows.push({
      row_id: row.row_id,
      display_name: row.display_name,
      model_file: row.model_file,
      model_sha256: modelSha[row.row_id],
      context_window: w,
      pack_id: pack.pack_id,
      expected_code: pack.expected_code,
      reference_prompt_token_count: pack.reference_prompt_token_count,
      // The Rust harness asserts all three internally; a PASS proves all three.
      prompt_tokens_match: passed,
      generated_tokens_match: passed,
      generated_text_match: passed,
      generated_token_count: genMatch ? Number(genMatch[1]) : null,
      generated_text: genMatch ? genMatch[2] : null,
      oracle_artifact: `qa/gemma4/oracle/${row.model_file.replace(/\.gguf$/, '')}.${packStem}.json`,
      raw_artifact: log ? `logs/${logName}` : null,
    })
  }
}

const manifest = {
  schema: 'camelid.gemma4_context_512_8192_public_evidence.v1',
  generated_utc: new Date().toISOString(),
  git_head: head,
  checkout_clean: !dirty,
  scope:
    'Durable on-current-head bounded-context (512/1024/2048/4096/8192) greedy parity for the two ' +
    'dense Gemma 4 exact rows (E2B-it / E4B-it Q8_0). Each row proves camelid == pinned llama.cpp ' +
    '5d56eff oracle on prompt tokens, generated tokens, AND generated text via the in-tree ' +
    'gemma4_generation_parity harness.',
  harness: 'cargo test --release --test gemma4_generation_parity gemma4_greedy_generation_matches_llama_cpp_oracle (CAMELID_GEMMA4_GGUF + CAMELID_GEMMA4_PACK=context_<W>_v1)',
  oracle_build: 'llama.cpp 5d56eff llama-server (CPU, -ngl 0 --no-repack -fa off -ctk f32 -ctv f32 -ub 1; greedy temperature=0/top_k=1)',
  models: ROWS.map((r) => ({ row_id: r.row_id, filename: r.model_file, size_bytes: modelBytes[r.row_id], sha256: modelSha[r.row_id] })),
  rows,
  passed: allPass,
  claim_boundary:
    'Closes the bounded 512-8192 context packs for the exact E2B-it / E4B-it Q8_0 rows ONLY, on ' +
    'this current head. It does NOT promote model-native/larger context, other Gemma 4 sizes/quants, ' +
    'multimodal input, production throughput, portability, or broad Gemma-family support.',
}

fs.writeFileSync(path.join(bundleDir, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n')
const sums = []
for (const f of ['manifest.json', ...rows.filter((r) => r.raw_artifact).map((r) => r.raw_artifact)]) {
  sums.push(`${sha256(path.join(bundleDir, f))}  ${f}`)
}
fs.writeFileSync(path.join(bundleDir, 'SHA256SUMS'), sums.join('\n') + '\n')
console.log(`passed=${allPass} bundle=${bundleDir}`)
