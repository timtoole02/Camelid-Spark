#!/usr/bin/env node
// Capability harness for `context.full_length` / `context.rope_scaling` (oracle class D)
// on Windows CPU. Drives a LONG prompt through Camelid, emits a server-sealed
// camelid.parity-receipt/v1, and asserts bit-exact generation parity vs llama.cpp with
// the prompt PINNED (verify-receipt) — so a match proves Camelid's KV cache AND rope
// (incl. the llama3 baked rope_freqs scaling, for prompts whose positions exceed the
// original 8192 context) are correct across the whole context, token-for-token.
//
// SAFETY (conductor §9 predict-and-abort): the f32 CPU KV cache is the dominant,
// otherwise-uncapped memory term. Camelid now guards it in-engine (`ensure_position_capacity`
// in src/inference/kv_cache.rs projects the KV growth bytes and refuses over-budget with a
// typed error before allocating), so an oversized context fails closed instead of OOMing
// mid-generation. This harness independently projects KV bytes =
// target_tokens * kv_bytes_per_token and ABORTS before requesting an unsafe context, so the
// run never even reaches the engine guard — the host is never discovered by crashing it.
//
// Usage (repo root; release camelid + Windows llama.cpp build):
//   node qa/capability/context_parity.mjs --gguf models/Llama-3.2-1B-Instruct-Q8_0.gguf \
//     --row llama-3.2-1b-instruct-q8_0 --label "Llama 3.2 1B Instruct Q8_0" \
//     --target-tokens 16000 --kv-bytes-per-token 65536 --llama-ctx 16384 \
//     --camelid-exe target/release/camelid.exe --llama-server <path>/llama-server.exe \
//     --camelid-port 8231 --llama-port 8233 --out qa/capability/ctx_out/<row>
import { execFile, spawn } from 'node:child_process'
import { mkdir, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'
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
const need = k => args.get(k) ?? (() => { throw new Error(`--${k} required`) })()
const gguf = resolve(need('gguf'))
const row = need('row')
const label = args.get('label') || row
const targetTokens = Number.parseInt(need('target-tokens'), 10)
const kvBytesPerToken = Number.parseInt(need('kv-bytes-per-token'), 10)
const llamaCtx = Number.parseInt(args.get('llama-ctx') || String(targetTokens + 64), 10)
const maxGen = Number.parseInt(args.get('max-gen') || '8', 10)
const camelidExe = args.get('camelid-exe') || 'target/release/camelid.exe'
const llamaServer = args.get('llama-server') || 'llama-server'
const camelidPort = args.get('camelid-port') || '8231'
const llamaPort = args.get('llama-port') || '8233'
const out = resolve(args.get('out') || `qa/capability/ctx_out/${row}`)
// 'full' = camelid self-replay + llama.cpp reference (both); 'reference-only' skips the
// in-process self-replay (a 2nd slow CPU prefill) — the class-D parity vs llama.cpp is
// still fully asserted from the receipt's pinned prompt. Use reference-only on big rows.
const verifyMode = args.get('verify-mode') || 'full'
const safeBudget = Number.parseInt(args.get('safe-kv-budget-bytes') || String(3.5 * 1024 ** 3), 10)
const camelidBase = `http://127.0.0.1:${camelidPort}`
const env = { ...process.env, CUDA_VISIBLE_DEVICES: '-1' }

// ---- predict-and-abort: refuse a context whose projected KV exceeds the safe budget ----
const projectedKv = targetTokens * kvBytesPerToken
const gib = n => (n / 1024 ** 3).toFixed(2) + ' GiB'
console.log(`row=${row} target_tokens=${targetTokens} kv/token=${kvBytesPerToken}B projected_KV=${gib(projectedKv)} safe_budget=${gib(safeBudget)}`)
if (projectedKv > safeBudget) {
  console.log(`ABORT (predict-and-abort): projected KV ${gib(projectedKv)} exceeds safe budget ${gib(safeBudget)} on this host — not requested.`)
  process.exitCode = 2
  process.exit()
}

import { writeFile as wf, mkdtemp } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
// Heavy long-prompt POST goes through curl: a multi-thousand-token CPU prefill is
// O(n^2) in attention and routinely outruns node/undici's 5-min headers timeout.
async function curlJson(url, body) {
  const dir = await mkdtemp(join(tmpdir(), 'ctxreq-'))
  const reqPath = join(dir, 'req.json')
  await wf(reqPath, JSON.stringify(body))
  const { stdout } = await execFileAsync('curl', ['-s', '--max-time', '1800', '-H', 'content-type: application/json', '-d', `@${reqPath}`, url], { maxBuffer: 64 * 1024 * 1024 })
  return stdout ? JSON.parse(stdout) : null
}
async function waitUrl(url, ms = 120000) {
  const deadline = Date.now() + ms
  let last
  while (Date.now() < deadline) {
    try { const r = await fetch(url); if (r.ok || r.status === 404) return } catch (e) { last = e }
    await new Promise(r => setTimeout(r, 500))
  }
  throw new Error(`not reachable at ${url}: ${last?.message}`)
}

// Build a long, non-degenerate prompt by repeating a varied paragraph to ~target tokens
// (~3.6 chars/token heuristic; actual length is read back from the receipt).
const para = 'The river wound past the old stone mill, where farmers once ground wheat into flour for the village bakeries. ' +
  'Each autumn the orchards filled with ripe apples, and children carried baskets along the dusty lane toward the market square. ' +
  'A blacksmith hammered iron near the well while merchants argued over the price of wool and salt. '
const reps = Math.ceil((targetTokens * 3.6) / para.length)
const prompt = (para.repeat(reps)).slice(0, Math.floor(targetTokens * 3.6))

await mkdir(out, { recursive: true })
let camelid = spawn(camelidExe, ['serve', '--addr', `127.0.0.1:${camelidPort}`, '--model', gguf], { stdio: 'ignore', env })
let parityReceipt, promptTokens, genText
try {
  await waitUrl(`${camelidBase}/api/models/current`)
  const resp = await curlJson(`${camelidBase}/v1/completions`, {
    prompt, max_tokens: maxGen, temperature: 0, stream: false, camelid_receipt: true,
  })
  parityReceipt = resp.camelid_receipt
  if (!parityReceipt) throw new Error(`no camelid_receipt: ${JSON.stringify(resp.error || resp).slice(0, 300)}`)
  promptTokens = (resp.camelid?.prompt_token_ids || parityReceipt.result?.prompt_token_ids || []).length
  genText = resp.choices?.[0]?.text ?? ''
} finally {
  camelid.kill('SIGTERM')
  await new Promise(r => setTimeout(r, 800))
}
const receiptPath = resolve(out, 'parity-receipt.json')
await writeFile(receiptPath, JSON.stringify(parityReceipt, null, 2) + '\n')
const ctx = Math.max(llamaCtx, promptTokens + maxGen + 8)
console.log(`actual_prompt_tokens=${promptTokens} positions_exceed_8192=${promptTokens > 8192} llama_ctx=${ctx}`)

let verifyOut = ''
const verifyArgs = [
  'verify-receipt', receiptPath, '--gguf', gguf, '--llama-server', llamaServer,
  '--llama-ctx', String(ctx), '--llama-port', String(Number(llamaPort) + 10),
]
if (verifyMode === 'reference-only') verifyArgs.push('--reference-only')
try {
  const { stdout } = await execFileAsync(camelidExe, verifyArgs, { env, maxBuffer: 32 * 1024 * 1024 })
  verifyOut = stdout
} catch (e) { verifyOut = (e.stdout || '') + (e.stderr || '') + String(e) }
const pass = verifyMode === 'reference-only'
  ? /PASS reference-rerun/.test(verifyOut)
  : (/RECEIPT VERIFIED/.test(verifyOut) && /PASS reference-rerun/.test(verifyOut))
await writeFile(resolve(out, 'verify.log'), verifyOut)
const summary = { row, label, gguf, target_tokens: targetTokens, actual_prompt_tokens: promptTokens, positions_exceed_8192: promptTokens > 8192, projected_kv_bytes: projectedKv, generation_parity_pass: pass, evidence: (verifyOut.match(/PASS reference-rerun:.*/)?.[0] || '').trim(), generated_text: genText }
await writeFile(resolve(out, 'summary.json'), JSON.stringify(summary, null, 2) + '\n')
console.log(`generation_parity_pass=${pass}  (${summary.evidence})`)
console.log(`receipt=${receiptPath}`)
if (!pass) { console.log('FAIL: generation parity did not verify'); process.exitCode = 1 } else console.log('OK')
