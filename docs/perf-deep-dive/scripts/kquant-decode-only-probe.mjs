// Decode-only A/B: no cold prefill probes between decode measurements.
import { spawn } from 'node:child_process'
const CAMELID_BIN = process.env.CAMELID_BIN, MODEL = process.env.MODEL_GGUF
const FLAG = process.env.KQ_FLAG // '0' or '1'
const sleep = ms => new Promise(r => setTimeout(r, ms))
async function waitHealth(url, ms = 300000) { const t0 = Date.now(); for (;;) { try { const r = await fetch(url); if (r.ok) return } catch {} if (Date.now() - t0 > ms) throw new Error('health timeout'); await sleep(400) } }
async function chat(messages, maxTokens) {
  const t0 = performance.now()
  const r = await fetch('http://127.0.0.1:8181/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: 'm', messages, max_tokens: maxTokens, temperature: 0, stream: false }) })
  if (!r.ok) throw new Error('HTTP ' + r.status)
  const j = await r.json(); const t1 = performance.now()
  return { ms: t1 - t0, n: j?.usage?.completion_tokens ?? 0, text: j?.choices?.[0]?.message?.content ?? '' }
}
const msgs = [{ role: 'system', content: 'You are a concise assistant.' }, { role: 'user', content: 'Count from 1 to 80, separated by commas. Output only the numbers.' }]
const env = { ...process.env, CAMELID_CUDA_RESIDENT_DECODE: '0', CAMELID_CUDA_RESIDENT_PREFILL: '0', CUDA_VISIBLE_DEVICES: '-1' }
if (FLAG === '1') env.CAMELID_X86_KQUANT_MATMUL_OWNER = '1'; else delete env.CAMELID_X86_KQUANT_MATMUL_OWNER
const child = spawn(CAMELID_BIN, ['serve', '--addr', '127.0.0.1:8181'], { stdio: ['ignore', 'ignore', 'ignore'], env })
try {
  await waitHealth('http://127.0.0.1:8181/v1/health')
  await (await fetch('http://127.0.0.1:8181/api/models/load', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ path: MODEL, id: 'm' }) })).json()
  await chat(msgs, 4) // warmup: caches prompt prefix
  const d1 = await chat(msgs, 1)
  const results = []
  for (let i = 0; i < 4; i++) {
    const dN = await chat(msgs, 64)
    results.push(((dN.n - 1) / ((dN.ms - d1.ms) / 1000)).toFixed(2))
  }
  console.log(`flag=${FLAG} decode tok/s: ${results.join(', ')}`)
} finally { child.kill('SIGTERM') }
