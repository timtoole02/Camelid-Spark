#!/usr/bin/env node
// A/B parity probe for the batched GPU prefill vs the serial GPU prefill.
//
// Greedy /v1/chat/completions (ChatML, thinking disabled — the parity-locked mode).
// Run twice against the same build: once with batched prefill (default) and once with
// CAMELID_CUDA_RESIDENT_PREFILL_BATCHED=0 (serial). The serial path is the one the
// committed CUDA-resident bundles validated token-identical to llama.cpp, so
// batched == serial over a 50-token completion ⇒ batched == llama.cpp transitively.
//
// Prints JSON { prompt: completionText } to stdout for a caller to diff.
//
// Usage: node scripts/qwen3-cuda-prefill-ab.mjs --base http://127.0.0.1:8185

import http from 'node:http'

const args = parseArgs(process.argv.slice(2))
const base = (args.get('base') || 'http://127.0.0.1:8185').replace(/\/$/, '')
const maxTokens = Number.parseInt(args.get('max-tokens') || '50', 10)
// --depth-tokens N: append one ~N-token prompt so the completion decodes at real
// depth (split-K attention above SPLITK_THRESHOLD). Default 0 keeps the corpus
// byte-identical for existing callers.
const depthTokens = Number.parseInt(args.get('depth-tokens') || '0', 10)

// The three fixed bundle prompts (transitive to llama.cpp via the serial-prefill
// bundle) plus one long prompt that forces many MAX_VERIFY_K prefill chunks.
const longPrompt =
  'Read the following passage carefully and then answer. ' +
  'The quick brown fox jumps over the lazy dog near the riverbank. '.repeat(12) +
  'In one word, what animal jumps?'
const PROMPTS = [
  'What is the capital of France?',
  'Name a primary color.',
  'Say hello.',
  longPrompt,
]
if (depthTokens > 0) PROMPTS.push(buildDepthPrompt(depthTokens))

function buildDepthPrompt(approxTokens) {
  // Same deterministic filler as bench-qwen3-cuda-resident.mjs (~0.75 words/token).
  const sentence = 'The quick brown fox jumps over the lazy dog near the riverbank. '
  let s = 'Summarize the following passage in one word.\n\n'
  while (s.length < approxTokens * 5) s += sentence
  return s
}

async function chat(userContent) {
  // node:http, not fetch: undici's default 300s headers timeout kills legs
  // whose prefill legitimately exceeds it (K-quant GPU prefill of a ~2080-token
  // prompt under host load measured 302s). Same fix as the #399 context
  // harness. No client timeout — the server's generation timeout governs.
  const payload = JSON.stringify({
    messages: [{ role: 'user', content: userContent }],
    max_tokens: maxTokens,
    temperature: 0,
    top_k: 1,
    camelid_enable_thinking: false,
  })
  const { status, text } = await new Promise((resolve, reject) => {
    const u = new URL(`${base}/v1/chat/completions`)
    const req = http.request(
      { hostname: u.hostname, port: u.port, path: u.pathname, method: 'POST', headers: { 'content-type': 'application/json', 'content-length': Buffer.byteLength(payload) } },
      (res) => {
        // setEncoding matters: without it each Buffer chunk decodes separately,
        // and a multi-byte UTF-8 sequence (the corpus contains emoji) split
        // across a chunk boundary would corrupt the text and break byte-parity.
        res.setEncoding('utf8')
        let d = ''
        res.on('data', (c) => (d += c))
        res.on('end', () => resolve({ status: res.statusCode, text: d }))
      },
    )
    req.on('error', reject)
    req.end(payload)
  })
  if (status !== 200) throw new Error(`chat -> HTTP ${status}: ${text.slice(0, 300)}`)
  const body = JSON.parse(text)
  const content = body.choices?.[0]?.message?.content
  if (typeof content !== 'string') {
    // A missing/undefined content would be silently DROPPED by JSON.stringify,
    // shrinking the corpus and letting a parity diff pass vacuously (the probe
    // fake-null: a mid-run script/server anomaly once produced two 4-key
    // corpora that "matched"). Fail loudly instead.
    throw new Error(`chat -> message.content is ${typeof content}: ${JSON.stringify(body).slice(0, 300)}`)
  }
  return content
}

async function main() {
  const out = {}
  for (const p of PROMPTS) out[p] = await chat(p)
  // Corpus-size stamp: lets callers assert the expected prompt count instead of
  // trusting two possibly-equally-truncated corpora to diff clean.
  console.error(`[ab-probe] corpus prompts=${PROMPTS.length} depth_tokens=${depthTokens}`)
  console.log(JSON.stringify(out, null, 2))
}

function parseArgs(argv) {
  const m = new Map()
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith('--')) {
      const key = argv[i].slice(2)
      const val = i + 1 < argv.length && !argv[i + 1].startsWith('--') ? argv[++i] : 'true'
      m.set(key, val)
    }
  }
  return m
}

main().catch((e) => {
  console.error(e.stack || String(e))
  process.exit(1)
})
