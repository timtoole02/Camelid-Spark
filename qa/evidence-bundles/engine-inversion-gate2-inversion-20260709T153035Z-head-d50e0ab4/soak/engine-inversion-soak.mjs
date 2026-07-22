#!/usr/bin/env node
// Gate 2 concurrency smoke for the engine-inversion mission.
// One camelid binary, one small model. N parallel workers issue mixed
// stream/non-stream greedy requests with random client disconnects for
// --duration seconds; every completed non-stream response must byte-match the
// canonical text captured at startup (garble detector). RSS is sampled every
// 5s. TTFT probes: a streaming request is fired WHILE a long non-streaming
// decode runs; time-to-first-byte is recorded (D3/queue benefit metric).
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
const port = Number(args.get('port') || '8291')
const out = args.get('out')
const durationS = Number(args.get('duration') || '600')
const workers = Number(args.get('workers') || '6')
if (!bin || !model || !out) throw new Error('need --bin --model --out')
const base = `http://127.0.0.1:${port}`

function req(path, body, { abortAfterMs = null } = {}) {
  return new Promise((resolvePromise, reject) => {
    const data = body === undefined ? null : JSON.stringify(body)
    const r = http.request(`${base}${path}`, {
      method: data === null ? 'GET' : 'POST',
      headers: data === null ? {} : { 'content-type': 'application/json' },
    }, res => {
      const chunks = []
      const t0 = process.hrtime.bigint()
      let firstByteNs = null
      res.on('data', c => { if (firstByteNs === null) firstByteNs = process.hrtime.bigint(); chunks.push(c) })
      res.on('end', () => resolvePromise({
        status: res.statusCode,
        body: Buffer.concat(chunks).toString('utf8'),
        aborted: false,
        firstByteNs, t0,
      }))
      res.on('aborted', () => resolvePromise({ status: res.statusCode, body: '', aborted: true }))
    })
    r.on('error', err => {
      // Deliberate client aborts surface as ECONNRESET-ish errors; report as aborted.
      resolvePromise({ status: 0, body: String(err), aborted: true })
    })
    r.setTimeout(900_000, () => { r.destroy(); })
    if (abortAfterMs !== null) setTimeout(() => r.destroy(), abortAfterMs)
    if (data !== null) r.write(data)
    r.end()
  })
}

async function waitHealth() {
  const deadline = Date.now() + 180_000
  for (;;) {
    try {
      const res = await req('/v1/health')
      if (JSON.parse(res.body).generation_ready === true) return
    } catch { /* not up */ }
    if (Date.now() > deadline) throw new Error('server never became generation_ready')
    await new Promise(r => setTimeout(r, 500))
  }
}

const PROMPT = 'The three primary colors are'
const chatBody = { messages: [{ role: 'user', content: PROMPT }], max_tokens: 32 }
const longBody = { messages: [{ role: 'user', content: PROMPT }], max_tokens: 256 }

const child = spawn(bin, ['serve', '--addr', `127.0.0.1:${port}`, '--model', model, '--no-open'],
  { stdio: ['ignore', 'pipe', 'pipe'] })
let stderrTail = []
child.stderr.on('data', c => { stderrTail.push(String(c)); if (stderrTail.length > 200) stderrTail.shift() })
child.stdout.on('data', () => {})
const pid = child.pid
let exited = false
child.once('exit', () => { exited = true })

const stats = {
  bin, model, pid, duration_s: durationS, workers,
  ok_nonstream: 0, ok_stream: 0, aborted_by_client: 0,
  queue_full_503: 0, other_5xx: 0, garbled: 0, mismatches: [],
  rss_mb: [], ttft_under_load_ms: [], panics: 0,
}

try {
  await waitHealth()
  // Canonical answer for the garble detector.
  const canonical = await req('/v1/chat/completions', chatBody)
  if (canonical.status !== 200) throw new Error(`canonical request failed: ${canonical.body.slice(0, 300)}`)
  const canonicalText = JSON.parse(canonical.body).choices[0].message.content

  const deadline = Date.now() + durationS * 1000
  const rssTimer = setInterval(() => {
    const ps = spawn('powershell', ['-NoProfile', '-Command',
      `[math]::Round((Get-Process -Id ${pid}).WorkingSet64/1MB)`])
    let buf = ''
    ps.stdout.on('data', c => { buf += c })
    ps.on('exit', () => { const v = Number(buf.trim()); if (Number.isFinite(v)) stats.rss_mb.push(v) })
  }, 5000)

  async function worker(id) {
    while (Date.now() < deadline) {
      const dice = Math.random()
      if (dice < 0.30) {
        // stream, read fully
        const res = await req('/v1/chat/completions', { ...chatBody, stream: true })
        if (res.aborted) stats.aborted_by_client += 1
        else if (res.status === 200 && res.body.includes('[DONE]')) stats.ok_stream += 1
        else if (res.status === 503 && res.body.includes('engine_queue_full')) stats.queue_full_503 += 1
        else if (res.status >= 500) { stats.other_5xx += 1; stats.mismatches.push(res.body.slice(0, 200)) }
      } else if (dice < 0.55) {
        // stream, hang up mid-flight (the T3 trigger)
        const res = await req('/v1/chat/completions', { ...longBody, stream: true },
          { abortAfterMs: 50 + Math.random() * 400 })
        stats.aborted_by_client += 1
        void res
      } else if (dice < 0.70) {
        // non-stream, hang up mid-decode (the T1 trigger)
        await req('/v1/chat/completions', longBody, { abortAfterMs: 30 + Math.random() * 200 })
        stats.aborted_by_client += 1
      } else {
        // non-stream, full read + garble check
        const res = await req('/v1/chat/completions', chatBody)
        if (res.aborted) stats.aborted_by_client += 1
        else if (res.status === 200) {
          stats.ok_nonstream += 1
          const text = JSON.parse(res.body).choices[0].message.content
          if (text !== canonicalText) {
            stats.garbled += 1
            stats.mismatches.push(text.slice(0, 200))
          }
        }
        else if (res.status === 503 && res.body.includes('engine_queue_full')) stats.queue_full_503 += 1
        else if (res.status >= 500) { stats.other_5xx += 1; stats.mismatches.push(res.body.slice(0, 200)) }
      }
      await new Promise(r => setTimeout(r, Math.random() * 100))
    }
  }

  async function ttftProber() {
    while (Date.now() < deadline) {
      // Occupy the engine with a long decode, then probe stream TTFB.
      const longRun = req('/v1/chat/completions', { ...longBody, max_tokens: 128 })
      await new Promise(r => setTimeout(r, 150))
      const t0 = process.hrtime.bigint()
      const probe = await req('/v1/chat/completions', { ...chatBody, stream: true })
      if (!probe.aborted && probe.firstByteNs) {
        stats.ttft_under_load_ms.push(Number(probe.firstByteNs - t0) / 1e6)
      }
      await longRun
      await new Promise(r => setTimeout(r, 2000))
    }
  }

  await Promise.all([...Array.from({ length: workers }, (_, i) => worker(i)), ttftProber()])
  clearInterval(rssTimer)

  stats.panics = stderrTail.filter(l => l.includes('panicked')).length
  stats.stderr_panic_lines = stderrTail.filter(l => l.includes('panicked')).slice(0, 5)
  await writeFile(out, JSON.stringify(stats, null, 2))
  console.log(`WROTE ${out}`)
  console.log(JSON.stringify({
    ok_nonstream: stats.ok_nonstream, ok_stream: stats.ok_stream,
    aborted: stats.aborted_by_client, q503: stats.queue_full_503,
    other_5xx: stats.other_5xx, garbled: stats.garbled, panics: stats.panics,
    rss_first: stats.rss_mb[0], rss_last: stats.rss_mb[stats.rss_mb.length - 1],
    ttft_med: stats.ttft_under_load_ms.sort((a, b) => a - b)[Math.floor(stats.ttft_under_load_ms.length / 2)],
  }))
} finally {
  child.kill()
  for (let i = 0; i < 40 && !exited; i += 1) await new Promise(r => setTimeout(r, 250))
  if (!exited) {
    spawn('taskkill', ['/PID', String(pid), '/F'], { stdio: 'ignore' })
    console.error(`hard-killed serve pid ${pid}`)
  } else {
    console.error(`serve pid ${pid} exited cleanly`)
  }
}
