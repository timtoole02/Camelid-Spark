// Single-engine, sequential legs: off / owner+VNNI / owner+AVX2(VNNI=0) / off.
import { spawn } from 'node:child_process'
const sleep = ms => new Promise(r => setTimeout(r, ms))
async function waitHealth(url, ms = 300000) { const t0 = Date.now(); for (;;) { try { const r = await fetch(url); if (r.ok) return } catch {} if (Date.now() - t0 > ms) throw new Error('health timeout'); await sleep(400) } }
const long = Array.from({ length: 24 }, (_, i) => `Fact ${i + 1}: the quick brown fox jumps over the lazy dog near the riverbank at dawn.`).join(' ')
async function chat(nonce) {
  const t0 = performance.now()
  const r = await fetch('http://127.0.0.1:8181/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: 'm', messages: [{ role: 'system', content: `You are concise. [${nonce}]` }, { role: 'user', content: 'Read this and answer OK.\n\n' + long }], max_tokens: 1, temperature: 0, stream: false }) })
  const t1 = performance.now()
  if (!r.ok) throw new Error('HTTP ' + r.status)
  const j = await r.json()
  const q8s = j?.camelid?.timings_ms?.q8_schedule ?? {}; return { ms: t1 - t0, ptok: j?.usage?.prompt_tokens ?? 0, kq: q8s.kquant_owner_prefill_taken ?? null, rp: q8s.kquant_owner_repack8_taken ?? null, built: q8s.kquant_owner_repack8_built ?? null }
}
async function leg(label, extra) {
  const env = { ...process.env, CAMELID_CUDA_RESIDENT_DECODE: '0', CAMELID_CUDA_RESIDENT_PREFILL: '0', CUDA_VISIBLE_DEVICES: '-1', CAMELID_Q8_SCHED_TELEMETRY: '1', ...extra }
  const child = spawn(process.env.CAMELID_BIN, ['serve', '--addr', '127.0.0.1:8181'], { stdio: ['ignore', 'ignore', 'ignore'], env })
  try {
    await waitHealth('http://127.0.0.1:8181/v1/health')
    await (await fetch('http://127.0.0.1:8181/api/models/load', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ path: process.env.MODEL_GGUF, id: 'm' }) })).json()
    await chat('warm')
    const rates = [], counters = []
    for (let i = 0; i < 4; i++) { const p = await chat(`m-${i}-${Date.now()}`); rates.push(p.ptok / (p.ms / 1000)); counters.push(`${p.kq}/${p.rp}/${p.built}`) }
    rates.sort((a, b) => a - b)
    const med = (rates[1] + rates[2]) / 2
    console.log(`${label}: ${rates.map(x => x.toFixed(2)).join(', ')} | median ${med.toFixed(2)} | kq_taken: ${counters.at(-1)}`)
    return med
  } finally { child.kill('SIGTERM'); await sleep(2000) }
}
const off = await leg('off            ', {})
const vnni = await leg("owner+vnni     ", { CAMELID_X86_KQUANT_MATMUL_OWNER: "1" })
const repack = await leg("owner+repack8  ", { CAMELID_X86_KQUANT_MATMUL_OWNER: "1", CAMELID_X86_KQUANT_MATMUL_OWNER_REPACK8: "1" })

const off2 = await leg('off (drift)    ', {})
console.log(`vnni/off: ${(vnni / off).toFixed(3)}x | repack8/off: ${(repack / off).toFixed(3)}x | repack8/vnni: ${(repack / vnni).toFixed(3)}x | drift: ${(off2 / off).toFixed(3)}x`)
