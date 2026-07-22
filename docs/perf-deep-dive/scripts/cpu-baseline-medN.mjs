#!/usr/bin/env node
// Treaty-clean CPU baseline: llama.cpp (CPU) vs Camelid (CPU, default config A) — ONE model, median-of-N.
// Derivative of cpu-prefill-matrix.mjs (drops the known-dead config B; adds repeats + medians + camelid-vs-llama parity).
// Greedy/temp=0. CUDA MUST be hidden by the caller (CUDA_VISIBLE_DEVICES=-1) so -ngl 0 is a TRUE CPU run.
// Env required: CAMELID_BIN, LLAMA_SERVER_BIN, MODEL_GGUF.  Optional: MODEL_ID, OUT_JSON, REPEATS(5), FACTS(24),
//   DECODE_TOKENS(64), LLAMA_CTX(1536), THREADS(8).
import { spawn } from 'node:child_process'
import { writeFile, mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'

const CAMELID_BIN = process.env.CAMELID_BIN
const LLAMA_SERVER_BIN = process.env.LLAMA_SERVER_BIN
const MODEL = process.env.MODEL_GGUF
const MODEL_ID = process.env.MODEL_ID || 'probe-model'
const OUT = process.env.OUT_JSON
const REPEATS = parseInt(process.env.REPEATS || '5', 10)
const FACTS = parseInt(process.env.FACTS || '24', 10)
const DECODE_TOKENS = parseInt(process.env.DECODE_TOKENS || '64', 10)
const LLAMA_CTX = process.env.LLAMA_CTX || '1536'
const THREADS = process.env.THREADS || '8'
if (!CAMELID_BIN || !LLAMA_SERVER_BIN || !MODEL) { console.error('need CAMELID_BIN, LLAMA_SERVER_BIN, MODEL_GGUF'); process.exit(2) }

const longBlock = Array.from({ length: FACTS }, (_, i) =>
  `Fact ${i + 1}: the quick brown fox jumps over the lazy dog near the riverbank at dawn.`).join(' ')
let nonceCtr = 0
const mkPrefill = (nonce) => ([
  { role: 'system', content: `You are a concise assistant. [run ${nonce}]` },
  { role: 'user', content: `Read the following text and answer with the single word OK.\n\n${longBlock}` },
])
const decodeMessages = [
  { role: 'system', content: 'You are a concise assistant.' },
  { role: 'user', content: 'Count from 1 to 80, separated by commas. Output only the numbers.' },
]
const round = (x) => Number.isFinite(x) ? Math.round(x * 100) / 100 : null
const sleep = (ms) => new Promise(r => setTimeout(r, ms))
const median = (a) => { const v = a.filter(Number.isFinite).slice().sort((x, y) => x - y); if (!v.length) return null; const m = v.length >> 1; return round(v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2) }
const firstDivergentIndex = (a, b) => { const n = Math.min(a.length, b.length); for (let i = 0; i < n; i++) if (a[i] !== b[i]) return i; return a.length === b.length ? -1 : n }

async function waitHealth(url, ms = 600000) {
  const start = Date.now()
  for (;;) {
    try { const r = await fetch(url); if (r.ok) return } catch {}
    if (Date.now() - start > ms) throw new Error(`health timeout ${url}`)
    await sleep(400)
  }
}
async function chat(base, messages, maxTokens) {
  const isLlama = base.includes('8183')
  const body = { model: MODEL_ID, messages, max_tokens: maxTokens, temperature: 0, stream: false }
  if (isLlama) body.cache_prompt = false // defeat llama prompt cache; Camelid rejects this param + uses a unique nonce
  const t0 = performance.now() // BEFORE fetch: fetch() resolves only once the server has computed the (non-streamed) response
  const r = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${(await r.text()).slice(0, 300)}`)
  const j = await r.json()
  const t1 = performance.now()
  return { ms: t1 - t0, text: j?.choices?.[0]?.message?.content ?? '', usage: j?.usage ?? null, timings: j?.timings ?? null }
}
// One measured probe: cold prefill (unique nonce) + decode window (tN - t1).
async function probeOnce(base) {
  const pm = await chat(base, mkPrefill(`measure-${++nonceCtr}-${Date.now()}`), 1)
  const promptTokens = pm.usage?.prompt_tokens ?? pm.timings?.prompt_n ?? null
  const d1 = await chat(base, decodeMessages, 1)
  const dN = await chat(base, decodeMessages, DECODE_TOKENS)
  const nOut = dN.usage?.completion_tokens ?? null
  const win = dN.ms - d1.ms
  return {
    prefill_tok_s: promptTokens ? round(promptTokens / (pm.ms / 1000)) : null,
    prompt_tokens: promptTokens,
    decode_tok_s: (nOut && nOut > 1 && win > 0) ? round((nOut - 1) / (win / 1000)) : null,
    decode_text: dN.text,
  }
}

function startProc(bin, argv, env) {
  const child = spawn(bin, argv, { stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, ...env } })
  const tag = bin.includes('camelid') ? 'camelid' : 'llama'
  child.stdout.on('data', c => process.stderr.write(`[${tag}] ${c}`))
  child.stderr.on('data', c => process.stderr.write(`[${tag}] ${c}`))
  return child
}

const report = {
  generated_utc: new Date().toISOString(), model: MODEL, model_id: MODEL_ID,
  facts: FACTS, decode_tokens: DECODE_TOKENS, repeats: REPEATS, threads: THREADS, llama_ctx: LLAMA_CTX,
  cuda_visible_devices: process.env.CUDA_VISIBLE_DEVICES ?? '(unset!)',
  host: process.env.BENCH_HOST || 'win i7-11800H', camelid_head: process.env.CAMELID_HEAD || null,
  llama_pin: process.env.LLAMA_PIN || 'acd79d6',
  // Flag provenance: which owner lanes the camelid child inherited (the whole
  // A/B claim of a receipt hangs on these — record them, don't infer from
  // filenames).
  flags_env: {
    CAMELID_X86_Q8_MATMUL_OWNER: process.env.CAMELID_X86_Q8_MATMUL_OWNER ?? '(unset: platform default)',
    CAMELID_X86_KQUANT_MATMUL_OWNER: process.env.CAMELID_X86_KQUANT_MATMUL_OWNER ?? '(unset: off)',
    CAMELID_X86_KQUANT_MATMUL_OWNER_VNNI: process.env.CAMELID_X86_KQUANT_MATMUL_OWNER_VNNI ?? '(unset: on-when-owner-on)',
    CAMELID_WIN_PIN: process.env.CAMELID_WIN_PIN ?? '(unset: off)',
  },
  repeats_detail: [], parity: {},
}
let llama, camelid
try {
  // --cache-ram 0 disables llama's cross-request slot/prompt cache (LCP reuse) so each prefill is truly cold,
  // on top of per-request cache_prompt:false. Without it llama re-uses a near-identical cached prompt (sim~1.0).
  llama = startProc(LLAMA_SERVER_BIN, ['--host', '127.0.0.1', '--port', '8183', '-m', MODEL, '-ngl', '0', '-c', LLAMA_CTX, '-t', THREADS, '--no-warmup', '--cache-ram', '0'], {})
  await waitHealth('http://127.0.0.1:8183/health')
  camelid = startProc(CAMELID_BIN, ['serve', '--addr', '127.0.0.1:8181'], { CAMELID_CUDA_RESIDENT_DECODE: '0', CAMELID_CUDA_RESIDENT_PREFILL: '0' })
  await waitHealth('http://127.0.0.1:8181/v1/health')
  await (await fetch('http://127.0.0.1:8181/api/models/load', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ path: MODEL, id: MODEL_ID }) })).json()

  // warmup both (weights/page cache) — not measured
  await chat('http://127.0.0.1:8181', mkPrefill('warm'), 1); await chat('http://127.0.0.1:8181', decodeMessages, 4)
  await chat('http://127.0.0.1:8183', mkPrefill('warm'), 1); await chat('http://127.0.0.1:8183', decodeMessages, 4)

  for (let r = 0; r < REPEATS; r++) {
    const cam = await probeOnce('http://127.0.0.1:8181')   // alternate within each repeat to balance drift
    const lla = await probeOnce('http://127.0.0.1:8183')
    report.repeats_detail.push({ rep: r + 1, camelid: cam, llama: lla })
    console.error(`=== rep ${r + 1}/${REPEATS}: camelid pf ${cam.prefill_tok_s} dec ${cam.decode_tok_s} | llama pf ${lla.prefill_tok_s} dec ${lla.decode_tok_s} ===`)
  }

  const camPf = report.repeats_detail.map(x => x.camelid.prefill_tok_s)
  const camDe = report.repeats_detail.map(x => x.camelid.decode_tok_s)
  const llaPf = report.repeats_detail.map(x => x.llama.prefill_tok_s)
  const llaDe = report.repeats_detail.map(x => x.llama.decode_tok_s)
  report.median = {
    camelid_prefill_tok_s: median(camPf), camelid_decode_tok_s: median(camDe),
    llama_prefill_tok_s: median(llaPf), llama_decode_tok_s: median(llaDe),
    prefill_ratio: median(camPf) && median(llaPf) ? round(median(camPf) / median(llaPf)) : null,
    decode_ratio: median(camDe) && median(llaDe) ? round(median(camDe) / median(llaDe)) : null,
  }
  // parity proxy: greedy decode TEXT of the deterministic "count to 80" prompt (same-host, temp=0).
  // NOTE: text-level (each engine applies its own chat template); not token-id level. Labelled as such.
  const camTxt = report.repeats_detail.at(-1).camelid.decode_text
  const llaTxt = report.repeats_detail.at(-1).llama.decode_text
  report.parity = {
    level: 'decode-text (same prompt, temp=0; each engine applies its own chat template)',
    camelid_text: camTxt, llama_text: llaTxt,
    identical: camTxt === llaTxt, first_divergent_char_index: firstDivergentIndex(camTxt, llaTxt),
  }
} finally {
  try { camelid?.kill('SIGTERM') } catch {}
  try { llama?.kill('SIGTERM') } catch {}
}
console.log(JSON.stringify(report.median, null, 2))
if (OUT) { await mkdir(dirname(OUT), { recursive: true }); await writeFile(OUT, JSON.stringify(report, null, 2) + '\n'); console.error(`wrote ${OUT}`) }
