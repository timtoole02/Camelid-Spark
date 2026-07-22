#!/usr/bin/env node
// Gate 1 parity/perf capture for the engine-inversion mission.
// Runs ONE camelid binary against ONE model, captures canonicalized outputs
// (greedy chat, greedy completion, greedy SSE stream) plus coarse perf, then
// shuts the server down by PID. Run once per (binary, model) leg; diff the
// canonical JSONs across binaries.
import { spawn } from 'node:child_process'
import { writeFile } from 'node:fs/promises'
import http from 'node:http'

const args = new Map()
for (let i = 2; i < process.argv.length; i += 1) {
  const a = process.argv[i]
  if (!a.startsWith('--')) continue
  const [k, inline] = a.slice(2).split('=', 2)
  args.set(k, inline ?? process.argv[++i])
}
const bin = args.get('bin')
const model = args.get('model')
const port = Number(args.get('port') || '8281')
const out = args.get('out')
if (!bin || !model || !out) throw new Error('need --bin --model --out')
const base = `http://127.0.0.1:${port}`

function req(path, body, { stream = false } = {}) {
  return new Promise((resolvePromise, reject) => {
    const data = body === undefined ? null : JSON.stringify(body)
    const r = http.request(`${base}${path}`, {
      method: data === null ? 'GET' : 'POST',
      headers: data === null ? {} : { 'content-type': 'application/json' },
    }, res => {
      const chunks = []
      const firstByteAt = { t: null }
      res.on('data', c => { if (firstByteAt.t === null) firstByteAt.t = process.hrtime.bigint(); chunks.push(c) })
      res.on('end', () => resolvePromise({
        status: res.statusCode,
        body: Buffer.concat(chunks).toString('utf8'),
        firstByteNs: firstByteAt.t,
      }))
    })
    r.on('error', reject)
    r.setTimeout(900_000, () => reject(new Error('request timeout')))
    if (data !== null) r.write(data)
    r.end()
  })
}

async function waitHealth() {
  const deadline = Date.now() + 180_000
  for (;;) {
    try {
      const res = await req('/v1/health')
      const parsed = JSON.parse(res.body)
      if (parsed.generation_ready === true) return
    } catch { /* not up yet */ }
    if (Date.now() > deadline) throw new Error('server never became generation_ready')
    await new Promise(r => setTimeout(r, 500))
  }
}

// Strip run-variant fields so canonical outputs are byte-comparable across runs.
function canonicalizeJsonResponse(text) {
  const v = JSON.parse(text)
  if (typeof v.id === 'string') v.id = '<id>'
  if (v.camelid && typeof v.camelid === 'object') delete v.camelid.timings_ms
  if (v.camelid_receipt) v.camelid_receipt = '<receipt-present>'
  return v
}
function canonicalizeSse(text) {
  return text
    .split(/\n\n/)
    .map(evt => evt.replace(/"id":"(chatcmpl|cmpl)-[0-9a-f-]+"/g, '"id":"<id>"'))
    .filter(evt => evt.trim().length > 0)
}

const PROMPTS = [
  'hello',
  'The three primary colors are',
  'Write two short sentences about llamas.',
]

const child = spawn(bin, ['serve', '--addr', `127.0.0.1:${port}`, '--model', model, '--no-open'],
  { stdio: ['ignore', 'pipe', 'pipe'] })
child.stdout.on('data', () => {})
child.stderr.on('data', () => {})
const pid = child.pid
let exited = false
child.once('exit', () => { exited = true })

try {
  await waitHealth()
  const result = { bin, model, pid, chat: [], completion: [], stream: [], perf: [] }
  for (const prompt of PROMPTS) {
    const t0 = process.hrtime.bigint()
    const chat = await req('/v1/chat/completions', {
      messages: [{ role: 'user', content: prompt }],
      max_tokens: 48,
    })
    const t1 = process.hrtime.bigint()
    if (chat.status !== 200) throw new Error(`chat ${chat.status}: ${chat.body.slice(0, 300)}`)
    const chatParsed = canonicalizeJsonResponse(chat.body)
    result.chat.push(chatParsed)
    result.perf.push({
      kind: 'chat', prompt,
      wall_ms: Number(t1 - t0) / 1e6,
      completion_tokens: chatParsed.usage?.completion_tokens,
    })

    const completion = await req('/v1/completions', { prompt, max_tokens: 32 })
    if (completion.status !== 200) throw new Error(`completion ${completion.status}: ${completion.body.slice(0, 300)}`)
    result.completion.push(canonicalizeJsonResponse(completion.body))

    const s0 = process.hrtime.bigint()
    const stream = await req('/v1/chat/completions', {
      messages: [{ role: 'user', content: prompt }],
      max_tokens: 24,
      stream: true,
    })
    if (stream.status !== 200) throw new Error(`stream ${stream.status}: ${stream.body.slice(0, 300)}`)
    result.stream.push(canonicalizeSse(stream.body))
    result.perf.push({
      kind: 'stream-ttfb', prompt,
      ttfb_ms: stream.firstByteNs === null ? null : Number(stream.firstByteNs - s0) / 1e6,
    })
  }
  await writeFile(out, JSON.stringify(result, null, 2))
  console.log(`WROTE ${out}`)
} finally {
  // Kill by PID and confirm exit — never leave an orphaned server (hard rule).
  child.kill()
  for (let i = 0; i < 40 && !exited; i += 1) await new Promise(r => setTimeout(r, 250))
  if (!exited) {
    spawn('taskkill', ['/PID', String(pid), '/F'], { stdio: 'ignore' })
    console.error(`hard-killed serve pid ${pid}`)
  } else {
    console.error(`serve pid ${pid} exited cleanly`)
  }
}
