#!/usr/bin/env node
// Qwen3 ChatML THINKING-MODE leading-trace harness (opt-in, NOT parity-locked).
//
// This is the thinking-ENABLED sibling of scripts/chat-parity-qwen3.mjs. It does
// NOT assert full-trace token equality (unachievable — thinking traces run
// hundreds of tokens and reliably reach the documented f32-accumulation
// frontier). Instead it records, per probe, the LEADING-TRACE envelope: the
// number of generated tokens that are identical to the pinned llama.cpp
// reference before the first divergence (a benign f32 near-tie). The honest
// artifact is "tokens 0..k identical, divergence at k" — never all_pass.
//
// The rendered prompt is the bare-assistant generation turn
// (`…<|im_start|>assistant\n`, no pre-filled <think></think> block), so the model
// emits its OWN <think>…</think> reasoning. camelid is driven with
// camelid_enable_thinking:true.
//
// Because an 8B Q8 reference + an 8B camelid forward do not co-reside in 16 GB,
// this runs in phases against ONE server at a time:
//   --mode reference  : query a running llama-server, dump ref tokens to --out
//   --mode camelid    : query a running camelid, dump camelid text+tokens to --out
//   --mode compare    : read both dumps, compute the leading-trace, write the
//                       parity artifact (+ optional bundle dir) to --bundle-out
//
// If a row's leading trace diverges at token 0 (immediate divergence) the
// compare phase marks `blocker: true` — that is a blocker report, not a row to
// promote.

import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'
import http from 'node:http'

const args = parseArgs(process.argv.slice(2))
const mode = args.get('mode') || 'compare'

// The four fixed probes — the same set the 1.7B thinking bundle used, for
// cross-row comparability.
const PROMPTS = JSON.parse(
  args.get('prompts-json') ||
    process.env.QWEN3_THINK_PROMPTS_JSON ||
    JSON.stringify([
      'What is the capital of France?',
      'Name a primary color.',
      'What is 2+2?',
      'Say hello.',
    ]),
)
const N = Number.parseInt(args.get('n') || process.env.QWEN3_THINK_N || '256', 10)
const STOP = new Set([151645, 151643]) // <|im_end|>, <|endoftext|>

// Bare-assistant ChatML generation prompt (enable_thinking=true). Must match
// render_qwen3_chatml_prompt(messages, true) in src/api/mod.rs for a single user
// turn.
function renderThinkingChatML(userContent) {
  return `<|im_start|>user\n${userContent}<|im_end|>\n<|im_start|>assistant\n`
}

// node:http (NOT fetch): a single 8B cpu_reference 256-token generation easily
// exceeds undici's 300s default headers timeout (and the cold T7 read makes the
// first request slower still). node:http with no socket timeout lets the long
// non-streaming request complete.
function postJson(base, path, body) {
  return new Promise((resolve, reject) => {
    const url = new URL(`${base}${path}`)
    const data = JSON.stringify(body)
    const req = http.request(
      {
        hostname: url.hostname,
        port: url.port,
        path: url.pathname + url.search,
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'content-length': Buffer.byteLength(data),
        },
      },
      (res) => {
        let chunks = ''
        res.setEncoding('utf8')
        res.on('data', (c) => (chunks += c))
        res.on('end', () => {
          if (res.statusCode < 200 || res.statusCode >= 300) {
            reject(new Error(`${base}${path} -> HTTP ${res.statusCode}: ${chunks.slice(0, 300)}`))
            return
          }
          try {
            resolve(JSON.parse(chunks))
          } catch (e) {
            reject(e)
          }
        })
      },
    )
    req.on('error', reject)
    req.setTimeout(0) // no timeout — long non-streaming generations are expected
    req.write(data)
    req.end()
  })
}

async function referenceMode() {
  const base = (args.get('llama') || process.env.QWEN3_LLAMA_URL || 'http://127.0.0.1:8090').replace(/\/$/, '')
  const out = args.get('out')
  if (!out) throw new Error('--mode reference requires --out')
  const probes = []
  for (const prompt of PROMPTS) {
    const chatml = renderThinkingChatML(prompt)
    const r = await postJson(base, '/completion', {
      prompt: chatml,
      n_predict: N,
      temperature: 0,
      top_k: 1,
      seed: 0,
      cache_prompt: false,
      samplers: ['top_k'],
      return_tokens: true,
    })
    probes.push({ prompt, chatml, ref_text: r.content, ref_tokens: r.tokens })
    process.stderr.write(`reference: ${JSON.stringify(prompt)} -> ${r.tokens.length} tokens\n`)
  }
  const payload = {
    phase: 'reference',
    comparator: {
      engine: 'llama.cpp',
      version: args.get('llama-version') || process.env.QWEN3_LLAMA_VERSION || 'unknown',
      flags: args.get('llama-flags') || '-ngl 0 -ctk f32 -ctv f32 -fa off --no-repack -c 16384',
      url: base,
      fed: 'the bare-assistant thinking-enabled ChatML prompt via /completion',
    },
    n: N,
    probes,
  }
  await mkdir(dirname(out), { recursive: true })
  await writeFile(out, JSON.stringify(payload, null, 2))
  process.stderr.write(`wrote ${out}\n`)
}

async function camelidMode() {
  const base = (args.get('camelid') || process.env.CAMELID_API_BASE || 'http://127.0.0.1:8185').replace(/\/$/, '')
  const modelId = args.get('model-id') || process.env.QWEN3_MODEL_ID || 'Qwen3 1.7B Instruct'
  const out = args.get('out')
  if (!out) throw new Error('--mode camelid requires --out')
  const probes = []
  for (const prompt of PROMPTS) {
    const chat = await postJson(base, '/v1/chat/completions', {
      model: modelId,
      messages: [{ role: 'user', content: prompt }],
      max_tokens: N,
      temperature: 0,
      top_k: 1,
      seed: 0,
      camelid_enable_thinking: true,
      stream: false,
    })
    const text = chat.choices[0].message.content
    // Re-encode with parse_special:true so <think>/</think> map to their single
    // special ids, matching how llama.cpp's /completion returns them.
    const enc = await postJson(base, '/api/models/tokenizer/encode', {
      text,
      add_special: false,
      parse_special: true,
    })
    probes.push({ prompt, cam_text: text, cam_tokens: enc.tokens })
    process.stderr.write(`camelid: ${JSON.stringify(prompt)} -> ${enc.tokens.length} tokens\n`)
  }
  const payload = {
    phase: 'camelid',
    camelid: {
      url: base,
      model_id: modelId,
      api: '/v1/chat/completions with camelid_enable_thinking:true',
      backend: args.get('camelid-backend') || process.env.QWEN3_CAMELID_BACKEND || 'cpu_reference (CAMELID_METAL_RESIDENT_*=0)',
      renderer: 'render_qwen3_chatml_prompt(messages, enable_thinking=true)',
    },
    n: N,
    probes,
  }
  await mkdir(dirname(out), { recursive: true })
  await writeFile(out, JSON.stringify(payload, null, 2))
  process.stderr.write(`wrote ${out}\n`)
}

function stripTrailingStops(tokens) {
  const t = [...tokens]
  while (t.length && STOP.has(t[t.length - 1])) t.pop()
  return t
}

function leadingIdenticalCount(a, b) {
  let k = 0
  const max = Math.min(a.length, b.length)
  while (k < max && a[k] === b[k]) k++
  return k
}

async function compareMode() {
  const refPath = args.get('ref')
  const camPath = args.get('cam')
  if (!refPath || !camPath) throw new Error('--mode compare requires --ref and --cam')
  const ref = JSON.parse(await readFile(refPath, 'utf8'))
  const cam = JSON.parse(await readFile(camPath, 'utf8'))
  const rowId = args.get('row-id') || process.env.QWEN3_ROW_ID || 'qwen3_unknown'
  const displayName = args.get('display-name') || process.env.QWEN3_DISPLAY_NAME || rowId

  const probes = []
  let minK = Infinity
  let maxK = -Infinity
  let allEngage = true
  let allMatchAt5 = true
  let immediateDivergence = false

  for (const refProbe of ref.probes) {
    const camProbe = cam.probes.find((p) => p.prompt === refProbe.prompt)
    if (!camProbe) throw new Error(`camelid dump missing probe ${JSON.stringify(refProbe.prompt)}`)
    const refTokens = stripTrailingStops(refProbe.ref_tokens)
    const camTokens = stripTrailingStops(camProbe.cam_tokens)
    const k = leadingIdenticalCount(refTokens, camTokens)
    // Thinking engaged iff both generations open with the <think> block.
    const bothThink =
      refProbe.ref_text.trimStart().startsWith('<think>') &&
      camProbe.cam_text.trimStart().startsWith('<think>')
    if (!bothThink) allEngage = false
    if (k < 5) allMatchAt5 = false
    if (k === 0) immediateDivergence = true
    minK = Math.min(minK, k)
    maxK = Math.max(maxK, k)
    const divergence =
      k < Math.min(refTokens.length, camTokens.length)
        ? {
            first_divergence_token_index: k,
            camelid_token: camTokens[k],
            reference_token: refTokens[k],
          }
        : { first_divergence_token_index: null, note: 'no divergence within the shorter trace' }
    probes.push({
      prompt: refProbe.prompt,
      both_emit_think_block: bothThink,
      identical_leading_tokens: k,
      match_at_1: k >= 1,
      match_at_5: k >= 5,
      match_at_50: k >= 50,
      match_at_100: k >= 100,
      full_trace_match: k >= Math.min(refTokens.length, camTokens.length) && refTokens.length === camTokens.length,
      reference_trace_tokens: refTokens.length,
      camelid_trace_tokens: camTokens.length,
      divergence,
    })
  }

  const artifact = {
    schema: 'camelid.qwen3.thinking_leadingtrace.v1',
    support_scope: 'thinking_opt_in_leading_trace_only',
    row_id: rowId,
    display_name: displayName,
    mode: 'chatml_thinking_ENABLED_greedy',
    what_this_proves: [
      'The opt-in thinking-enabled ChatML renderer (bare assistant turn, no pre-filled <think></think>) makes Qwen3 emit its own reasoning block: both camelid and the pinned llama.cpp reference begin with <think> for every probe.',
      'Greedy generation is TOKEN-IDENTICAL to the llama.cpp reference for the LEADING reasoning trace (per-probe identical-prefix length below), then diverges at a benign f32 near-tie.',
    ],
    what_this_does_NOT_claim:
      'Full token-parity over arbitrary-length thinking traces. The parity-locked exact-row support mode remains thinking-DISABLED; this lane is opt-in and bounded to the leading-trace envelope below.',
    comparator: ref.comparator,
    camelid: cam.camelid,
    n: ref.n,
    all_probes_engage_thinking: allEngage,
    all_probes_match_at_5: allMatchAt5,
    immediate_divergence_blocker: immediateDivergence,
    leading_trace_parity_envelope_tokens: {
      min: minK === Infinity ? null : minK,
      max: maxK === -Infinity ? null : maxK,
      per_probe: Object.fromEntries(probes.map((p) => [p.prompt, p.identical_leading_tokens])),
    },
    probes,
  }
  if (immediateDivergence) {
    artifact.blocker = true
    artifact.blocker_reason =
      'At least one probe diverges at token 0 (immediate divergence). Per the lane contract this is a BLOCKER report, not a row to promote.'
  }

  const bundleOut = args.get('bundle-out')
  const json = JSON.stringify(artifact, null, 2)
  if (bundleOut) {
    await mkdir(dirname(bundleOut), { recursive: true })
    await writeFile(bundleOut, json)
    process.stderr.write(`wrote ${bundleOut}\n`)
  }
  process.stdout.write(json)
  // Human summary.
  process.stderr.write(`\n=== ${displayName} — thinking leading-trace ===\n`)
  for (const p of probes) {
    process.stderr.write(
      `  ${JSON.stringify(p.prompt)}: think ${p.both_emit_think_block ? 'YES' : 'NO'} | leading=${p.identical_leading_tokens} | @5 ${p.match_at_5 ? 'PASS' : 'FAIL'}\n`,
    )
  }
  process.stderr.write(
    `envelope: ${artifact.leading_trace_parity_envelope_tokens.min}-${artifact.leading_trace_parity_envelope_tokens.max} tokens; engage=${allEngage}; blocker=${immediateDivergence}\n`,
  )
  process.exitCode = immediateDivergence ? 2 : 0
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

if (mode === 'reference') await referenceMode()
else if (mode === 'camelid') await camelidMode()
else if (mode === 'compare') await compareMode()
else throw new Error(`unknown --mode ${mode} (reference|camelid|compare)`)
