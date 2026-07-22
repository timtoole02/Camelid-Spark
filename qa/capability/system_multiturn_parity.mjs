#!/usr/bin/env node
// Capability harness for `chat.system_multiturn` (oracle class D) on Windows CPU.
//
// Validates two complementary things for a system + multi-turn conversation:
//   1. GENERATION PARITY (the class-D core): Camelid emits a server-sealed
//      camelid.parity-receipt/v1 (camelid_receipt:true); `camelid verify-receipt`
//      then feeds Camelid's EXACT prompt token ids to llama.cpp /completion and
//      asserts the continuation matches bit-exact. Pinning the prompt isolates
//      decode parity from chat-template rendering.
//   2. TEMPLATE FIDELITY: compares Camelid's own /apply-template rendering + prompt
//      token ids against llama.cpp's for the same messages, and characterizes any
//      divergence (notably the Llama-3.x "Cutting Knowledge Date / Today Date"
//      system preamble, which carries a LIVE date and is therefore not byte-parity-able
//      across engines — see src/receipt/verify.rs:662).
//
// Exit 0 iff generation parity verifies. Template divergence is REPORTED, not failed,
// because the live-date preamble is an intended, deterministic-design difference.
//
// Usage (run from the repo root, release camelid + a Windows llama.cpp build):
//   node qa/capability/system_multiturn_parity.mjs \
//     --gguf models/Llama-3.2-3B-Instruct-Q8_0.gguf \
//     --row llama-3.2-3b-instruct-q8_0 --label "Llama 3.2 3B Instruct Q8_0" \
//     --camelid-exe target/release/camelid.exe \
//     --llama-server <path-to>/llama-server.exe \
//     --camelid-port 8231 --llama-port 8233 --out qa/capability/sm_out/<row>
// --gguf / --llama-server accept any path; defaults assume llama-server is on PATH.
import { execFile, spawn } from 'node:child_process'
import { mkdir, writeFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)
const args = new Map()
for (let i = 2; i < process.argv.length; i += 1) {
  const a = process.argv[i]
  if (!a.startsWith('--')) continue
  const [k, inline] = a.slice(2).split('=', 2)
  const v = inline ?? (process.argv[i + 1]?.startsWith('--') ? 'true' : process.argv[++i] ?? 'true')
  args.set(k, v)
}
const gguf = resolve(args.get('gguf') || (() => { throw new Error('--gguf required') })())
const row = args.get('row') || (() => { throw new Error('--row required') })()
const label = args.get('label') || row
const camelidExe = args.get('camelid-exe') || 'target/release/camelid.exe'
const llamaServer = args.get('llama-server') || 'llama-server'
const camelidPort = args.get('camelid-port') || '8231'
const llamaPort = args.get('llama-port') || '8233'
const maxTokens = Number.parseInt(args.get('max-tokens') || '24', 10)
const out = resolve(args.get('out') || `qa/capability/sm_out/${row}`)
const camelidBase = `http://127.0.0.1:${camelidPort}`
const llamaBase = `http://127.0.0.1:${llamaPort}`
const env = { ...process.env, CUDA_VISIBLE_DEVICES: '-1' }

// A system role + multi-turn (2 user turns, 1 prior assistant turn) conversation.
const messages = [
  { role: 'system', content: 'You are a terse assistant. Reply with a single short sentence.' },
  { role: 'user', content: 'What is the capital of France?' },
  { role: 'assistant', content: 'Paris.' },
  { role: 'user', content: 'And the capital of Japan?' },
]

async function fetchJson(url, body) {
  const res = await fetch(url, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) })
  const t = await res.text()
  if (!res.ok) throw new Error(`${url}: ${res.status}: ${t}`)
  return t ? JSON.parse(t) : null
}
async function waitUrl(url, label, ms = 90000) {
  const deadline = Date.now() + ms
  let last
  while (Date.now() < deadline) {
    try { const r = await fetch(url); if (r.ok || r.status === 404) return } catch (e) { last = e }
    await new Promise(r => setTimeout(r, 500))
  }
  throw new Error(`${label} not reachable at ${url}: ${last?.message}`)
}
function eqArr(a, b) { return a.length === b.length && a.every((x, i) => x === b[i]) }
function firstDiff(a, b) { const n = Math.max(a.length, b.length); for (let i = 0; i < n; i++) if (a[i] !== b[i]) return i; return -1 }

await mkdir(out, { recursive: true })
const summary = { row, label, gguf, checks: {} }

// ---- Phase 1: Camelid emits the sealed parity receipt + its own rendering ----
let camelid = spawn(camelidExe, ['serve', '--addr', `127.0.0.1:${camelidPort}`, '--model', gguf], { stdio: 'ignore', env })
let parityReceipt, camelidPrompt, camelidPromptTokens, camelidText
try {
  await waitUrl(`${camelidBase}/api/models/current`, 'camelid serve')
  const resp = await fetchJson(`${camelidBase}/v1/chat/completions`, {
    messages, temperature: 0, max_tokens: maxTokens, stream: false, camelid_receipt: true,
  })
  parityReceipt = resp.camelid_receipt
  if (!parityReceipt) throw new Error(`no camelid_receipt in response: ${JSON.stringify(resp.error || resp).slice(0, 300)}`)
  camelidPromptTokens = resp.camelid?.prompt_token_ids || []
  camelidText = resp.choices?.[0]?.message?.content ?? ''
  const tmpl = await fetchJson(`${camelidBase}/apply-template`, { messages })
  camelidPrompt = tmpl.prompt
} finally {
  camelid.kill('SIGTERM')
  await new Promise(r => setTimeout(r, 800))
}
const receiptPath = resolve(out, 'parity-receipt.json')
await writeFile(receiptPath, JSON.stringify(parityReceipt, null, 2) + '\n')
summary.checks.camelid_emit = { generated_text: camelidText, prompt_tokens: camelidPromptTokens.length, receipt_id: parityReceipt.receipt_id }

// ---- Phase 2: class-D generation parity via verify-receipt (spawns its own llama-server) ----
let verifyOut = ''
let verifyPass = false
try {
  const { stdout } = await execFileAsync(camelidExe, [
    'verify-receipt', receiptPath,
    '--gguf', gguf,
    '--llama-server', llamaServer,
    '--llama-ctx', '2048',
    '--llama-port', String(Number(llamaPort) + 10),
  ], { env, maxBuffer: 16 * 1024 * 1024 })
  verifyOut = stdout
} catch (e) { verifyOut = (e.stdout || '') + (e.stderr || '') + String(e) }
verifyPass = /RECEIPT VERIFIED/.test(verifyOut) && /PASS reference-rerun/.test(verifyOut)
await writeFile(resolve(out, 'verify.log'), verifyOut)
summary.checks.generation_parity = { pass: verifyPass, evidence: (verifyOut.match(/PASS reference-rerun:.*/)?.[0] || '').trim() }

// ---- Phase 3: template fidelity vs llama.cpp ----
let llama = spawn(llamaServer, ['--host', '127.0.0.1', '--port', llamaPort, '-m', gguf, '-ngl', '0', '-c', '2048', '--no-warmup'], { stdio: 'ignore', env })
try {
  await waitUrl(`${llamaBase}/health`, 'llama-server')
  const llamaTmpl = await fetchJson(`${llamaBase}/apply-template`, { messages })
  const llamaPrompt = llamaTmpl.prompt
  const llamaTok = await fetchJson(`${llamaBase}/tokenize`, { content: llamaPrompt, add_special: true })
  const llamaTokens = llamaTok.tokens || []
  const stringMatch = camelidPrompt === llamaPrompt
  const tokenMatch = eqArr(camelidPromptTokens, llamaTokens)
  // Characterize the divergence: does llama's rendering carry the live-date preamble?
  const preamble = /Cutting Knowledge Date|Today Date/.test(llamaPrompt) && !/Cutting Knowledge Date|Today Date/.test(camelidPrompt)
  summary.checks.template_fidelity = {
    prompt_string_match: stringMatch,
    prompt_tokens_match: tokenMatch,
    first_token_diff_index: firstDiff(camelidPromptTokens, llamaTokens),
    camelid_prompt_tokens: camelidPromptTokens.length,
    llama_prompt_tokens: llamaTokens.length,
    divergence: stringMatch ? 'none' : (preamble ? 'llama_injects_live_date_system_preamble' : 'other'),
    camelid_prompt: camelidPrompt,
    llama_prompt: llamaPrompt,
  }
} finally {
  llama.kill('SIGTERM')
}

await writeFile(resolve(out, 'summary.json'), JSON.stringify(summary, null, 2) + '\n')
console.log(`row=${row}`)
console.log(`generated_text=${JSON.stringify(camelidText)}`)
console.log(`generation_parity_pass=${summary.checks.generation_parity.pass}  (${summary.checks.generation_parity.evidence})`)
console.log(`template_string_match=${summary.checks.template_fidelity.prompt_string_match}  tokens_match=${summary.checks.template_fidelity.prompt_tokens_match}  divergence=${summary.checks.template_fidelity.divergence}`)
console.log(`receipt=${receiptPath}`)
console.log(`SUMMARY=${resolve(out, 'summary.json')}`)
if (!verifyPass) { console.log('FAIL: generation parity did not verify'); process.exitCode = 1 }
else console.log('OK')
