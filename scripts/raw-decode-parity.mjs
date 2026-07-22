#!/usr/bin/env node
// Template-agnostic raw-prompt token-level DECODE parity harness (K-quant lane).
//
// Certifies the camelid GPU-resident CUDA decode kernels (q4k_gemv / q6k_gemv)
// directly, with NO chat template in the way: for each raw prompt it greedily
// generates from both engines and checks token-AND-text parity.
//
//   camelid: POST /v1/completions   (raw prompt, GPU-resident decode)
//   llama:   POST /completion        (raw prompt, -ngl 0 CPU reference)
//
// This avoids the chat-template / BOS confounds AND the camelid_dense_diagnostics
// f32 path (which 503s on wire-only K-quant tensors). The reference's generated
// tokens may carry a trailing stop token stripped from the detokenized text, so we
// compare against the reference's CONTENT tokens (trailing stops removed) and the
// camelid text re-encoded by the camelid tokenizer — same discipline as
// chat-parity-qwen3.mjs.
//
// First prompt also asserts BOS alignment implicitly: if the engines disagreed on
// BOS the first generated token would differ and the run FAILs loudly.
//
// Usage:
//   node scripts/raw-decode-parity.mjs --camelid http://127.0.0.1:8185 \
//     --llama http://127.0.0.1:8090 --model-id "<id>" --row-id <id> \
//     --display-name "..." --comparator "..." --stop "128009,128001" \
//     --prompts-file qa/speed/prompts.json --out <bundle>/parity.json

import { writeFile, mkdir, readFile } from 'node:fs/promises'
import { dirname } from 'node:path'
import http from 'node:http'

const args = parseArgs(process.argv.slice(2))
const camelidBase = (args.get('camelid') || 'http://127.0.0.1:8185').replace(/\/$/, '')
const llamaBase = (args.get('llama') || 'http://127.0.0.1:8090').replace(/\/$/, '')
const modelId = args.get('model-id') || 'model'
const rowId = args.get('row-id') || 'row'
const displayName = args.get('display-name') || modelId
const comparatorLabel = args.get('comparator') || 'llama.cpp /completion'
const outPath = args.get('out') || null
const tokenCounts = (args.get('token-counts') || '1,5,50').split(',').map((s) => Number.parseInt(s.trim(), 10))
const STOP = new Set((args.get('stop') || '128009,128001,128000').split(',').map((s) => Number.parseInt(s.trim(), 10)))
// Node's global fetch (undici) has a ~5-min headersTimeout that fires when a CPU-only
// llama.cpp holds the connection during a multi-thousand-token f32 prefill. node:http
// has no such cap, so postJson uses it with a generous idle-timeout (default 30 min).
const requestTimeoutMs = Number.parseInt(args.get('request-timeout-ms') || '1800000', 10)
// Decoupled modes so the two engines never need to be loaded at once (safe on
// small-RAM hosts): --reference-out captures the llama.cpp oracle to a file and
// exits (no camelid); --reference-in compares camelid against that committed file
// (no llama.cpp). Run them in two phases, killing each server before the next.
const referenceOutPath = args.get('reference-out') || null
const referenceInPath = args.get('reference-in') || null
// variant/proof_chain default to the K-quant framing this harness was first written
// for; pass --variant / --proof-chain for other lanes (e.g. a Q8_0 GPU-resident row)
// so the emitted parity JSON doesn't carry wrong-quant labels.
const variant = args.get('variant') || 'kquant_gpu_resident'
const proofChain =
  args.get('proof-chain') ||
  'camelid GPU-resident CUDA decode (q4k_gemv/q6k_gemv) == llama.cpp, raw-prompt token+text parity. No chat template, no f32 diagnostics. K-quant has no camelid CPU decode path yet (Phase 2), so llama.cpp is the direct reference.'

let PROMPTS
if (args.get('prompts-file')) {
  const raw = JSON.parse(await readFile(args.get('prompts-file'), 'utf8'))
  PROMPTS = Array.isArray(raw) ? raw.map((p) => (typeof p === 'string' ? p : p.prompt ?? p.text)) : raw.prompts
} else {
  PROMPTS = JSON.parse(
    args.get('prompts-json') ||
      JSON.stringify([
        'The capital of France is',
        'Q: What is 2+2? A:',
        'Once upon a time,',
        'def fibonacci(n):',
      ]),
  )
}
PROMPTS = PROMPTS.filter(Boolean)

function postJson(base, path, body) {
  // node:http (not global fetch) so a long CPU-only llama.cpp prefill doesn't trip
  // undici's headersTimeout; the socket idle-timeout is the only cap (requestTimeoutMs).
  return new Promise((resolve, reject) => {
    const u = new URL(`${base}${path}`)
    const data = JSON.stringify(body)
    const req = http.request(
      {
        hostname: u.hostname,
        port: u.port,
        path: u.pathname + u.search,
        method: 'POST',
        headers: { 'content-type': 'application/json', 'content-length': Buffer.byteLength(data) },
      },
      (res) => {
        let buf = ''
        res.setEncoding('utf8')
        res.on('data', (c) => (buf += c))
        res.on('end', () => {
          if (res.statusCode < 200 || res.statusCode >= 300) {
            reject(new Error(`${base}${path} -> HTTP ${res.statusCode}: ${buf.slice(0, 300)}`))
          } else {
            try {
              resolve(JSON.parse(buf))
            } catch {
              reject(new Error(`${base}${path} -> non-JSON response: ${buf.slice(0, 200)}`))
            }
          }
        })
      },
    )
    req.setTimeout(requestTimeoutMs, () => req.destroy(new Error(`${base}${path} -> idle timeout after ${requestTimeoutMs}ms`)))
    req.on('error', reject)
    req.write(data)
    req.end()
  })
}

async function encodeCamelid(text) {
  const r = await postJson(camelidBase, '/api/models/tokenizer/encode', { text, add_special: false, parse_special: false })
  return r.tokens
}

async function camelidCompletion(prompt, maxTokens) {
  const r = await postJson(camelidBase, '/v1/completions', {
    model: modelId,
    prompt,
    max_tokens: maxTokens,
    temperature: 0,
    top_k: 1,
    seed: 0,
    stream: false,
  })
  return r.choices[0].text
}

async function referenceCompletion(prompt, nPredict) {
  const r = await postJson(llamaBase, '/completion', {
    prompt,
    n_predict: nPredict,
    temperature: 0,
    top_k: 1,
    seed: 0,
    cache_prompt: false,
    samplers: ['top_k'],
    return_tokens: true,
  })
  return { text: r.content, tokens: r.tokens }
}

function arraysEqual(a, b) {
  return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((v, i) => v === b[i])
}

async function main() {
  // Phase 1 (--reference-out): capture the llama.cpp oracle ALONE, then exit — no
  // camelid loaded, so the two engines never coexist in memory.
  if (referenceOutPath) {
    const captured = {}
    for (let i = 0; i < PROMPTS.length; i++) {
      for (const n of tokenCounts) {
        captured[`${i}|${n}`] = await referenceCompletion(PROMPTS[i], n)
        process.stderr.write(`captured reference ${i}|${n}\n`)
      }
    }
    await mkdir(dirname(referenceOutPath), { recursive: true })
    await writeFile(referenceOutPath, JSON.stringify(captured, null, 2))
    process.stderr.write(`wrote reference oracle ${referenceOutPath}\n`)
    return
  }
  // Phase 2 (--reference-in): compare camelid (live) vs the committed oracle — no
  // llama.cpp loaded. Live mode (neither flag) keeps the original both-live behavior.
  const referenceMap = referenceInPath ? JSON.parse(await readFile(referenceInPath, 'utf8')) : null
  const results = []
  let allPass = true
  for (let i = 0; i < PROMPTS.length; i++) {
    const prompt = PROMPTS[i]
    const perCount = {}
    for (const n of tokenCounts) {
      const ref = referenceMap ? referenceMap[`${i}|${n}`] : await referenceCompletion(prompt, n)
      const camText = await camelidCompletion(prompt, n)
      const camTokens = await encodeCamelid(camText)
      const refContentTokens = [...ref.tokens]
      while (refContentTokens.length && STOP.has(refContentTokens[refContentTokens.length - 1])) refContentTokens.pop()
      const textMatch = ref.text === camText
      const tokenMatch = arraysEqual(refContentTokens, camTokens)
      perCount[n] = {
        reference_text: ref.text,
        reference_tokens: ref.tokens,
        reference_content_tokens: refContentTokens,
        camelid_text: camText,
        camelid_content_tokens: camTokens,
        text_match: textMatch,
        token_match: tokenMatch,
        stopped_early_at_eos: ref.tokens.length < n,
      }
      if (!textMatch || !tokenMatch) allPass = false
    }
    results.push({ prompt, generations: perCount })
  }

  const report = {
    schema: 'camelid.raw_decode_parity.v1',
    variant,
    row_id: rowId,
    display_name: displayName,
    mode: 'raw_completion_greedy',
    comparator: comparatorLabel,
    proof_chain: proofChain,
    camelid_base: camelidBase,
    llama_base: llamaBase,
    token_counts: tokenCounts,
    all_pass: allPass,
    results,
  }

  const json = JSON.stringify(report, null, 2)
  if (outPath) {
    await mkdir(dirname(outPath), { recursive: true })
    await writeFile(outPath, json)
    process.stderr.write(`wrote ${outPath}\n`)
  }
  for (const r of results) {
    process.stderr.write(`\n=== ${JSON.stringify(r.prompt)} ===\n`)
    for (const n of tokenCounts) {
      const g = r.generations[n]
      process.stderr.write(`  n=${n}: text ${g.text_match ? 'PASS' : 'FAIL'} | tokens ${g.token_match ? 'PASS' : 'FAIL'} | cam=${JSON.stringify(g.camelid_text)}\n`)
    }
  }
  process.stderr.write(`\nALL_PASS: ${allPass}\n`)
  process.stdout.write(json)
  process.exitCode = allPass ? 0 : 1
}

function parseArgs(argv) {
  const map = new Map()
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a.startsWith('--')) {
      const key = a.slice(2)
      const next = argv[i + 1]
      if (next === undefined || next.startsWith('--')) map.set(key, 'true')
      else {
        map.set(key, next)
        i++
      }
    }
  }
  return map
}

await main()
