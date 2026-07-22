#!/usr/bin/env node
// Qwen3 ChatML chat-parity harness — TWO-PHASE variant for memory-constrained hosts.
//
// Same parity contract as scripts/chat-parity-qwen3.mjs (greedy, thinking
// DISABLED, prompt-token + generated-token + generated-text parity at 1/5/50),
// but split so the llama.cpp reference and camelid never need to be resident at
// the same time. Use this for large rows (e.g. Qwen3-8B Q8_0) on a ~16 GB host
// where an ~8 GB camelid process plus an ~8 GB llama-server will not co-reside.
//
//   Phase 1 (only llama-server running):
//     node scripts/chat-parity-qwen3-twophase.mjs --mode capture \
//       --llama http://127.0.0.1:8090 --oracle <oracle.json> \
//       [--prompts-json ...] [--token-counts 1,5,50]
//
//   ... stop llama-server, start camelid serve ...
//
//   Phase 2 (only camelid running):
//     node scripts/chat-parity-qwen3-twophase.mjs --mode compare \
//       --camelid http://127.0.0.1:8185 --oracle <oracle.json> \
//       --model-id "Qwen3 8B Instruct" --row-id qwen3_8b_instruct_q8_0 \
//       --display-name "Qwen3 8B Instruct Q8_0" --comparator "llama.cpp 9632 ..." \
//       --out <parity.json>
//
// The oracle file pins the prompts + token-counts captured in phase 1, so phase 2
// compares against exactly that reference even though llama-server is gone.

import { writeFile, readFile, mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'

const args = parseArgs(process.argv.slice(2))
const mode = args.get('mode')
const llamaBase = (args.get('llama') || process.env.QWEN3_LLAMA_URL || 'http://127.0.0.1:8090').replace(/\/$/, '')
const camelidBase = (args.get('camelid') || process.env.CAMELID_API_BASE || 'http://127.0.0.1:8185').replace(/\/$/, '')
const modelId = args.get('model-id') || process.env.QWEN3_MODEL_ID || 'Qwen3 8B Instruct'
const rowId = args.get('row-id') || 'qwen3_8b_instruct_q8_0'
const displayName = args.get('display-name') || 'Qwen3 8B Instruct Q8_0'
const comparatorLabel =
  args.get('comparator') || 'llama.cpp /completion (ChatML specials parsed), -ctk f32 -ctv f32 -fa off --no-repack'
const oraclePath = args.get('oracle')
const outPath = args.get('out') || null
const tokenCounts = (args.get('token-counts') || '1,5,50').split(',').map((s) => Number.parseInt(s.trim(), 10))
const PROMPTS = JSON.parse(
  args.get('prompts-json') ||
    JSON.stringify(['What is the capital of France?', 'Say hello.', 'What is 2+2?']),
)

// ChatML, thinking DISABLED — must match render_qwen3_chatml_prompt in src/api/mod.rs
// and renderChatML in scripts/chat-parity-qwen3.mjs.
function renderChatML(userContent) {
  return `<|im_start|>user\n${userContent}<|im_end|>\n` + `<|im_start|>assistant\n<think>\n\n</think>\n\n`
}

async function postJson(base, path, body) {
  const res = await fetch(`${base}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`${base}${path} -> HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`)
  return res.json()
}

async function referenceCompletion(promptText, nPredict) {
  const r = await postJson(llamaBase, '/completion', {
    prompt: promptText,
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

async function encodeCamelid(text, parseSpecial) {
  const r = await postJson(camelidBase, '/api/models/tokenizer/encode', {
    text,
    add_special: false,
    parse_special: parseSpecial,
  })
  return r.tokens
}

async function camelidChat(userContent, maxTokens) {
  const r = await postJson(camelidBase, '/v1/chat/completions', {
    model: modelId,
    messages: [{ role: 'user', content: userContent }],
    max_tokens: maxTokens,
    temperature: 0,
    top_k: 1,
    seed: 0,
    stream: false,
  })
  return r.choices[0].message.content
}

async function camelidChatPromptTokens(userContent) {
  const r = await postJson(camelidBase, '/v1/chat/completions', {
    model: modelId,
    messages: [{ role: 'user', content: userContent }],
    max_tokens: 1,
    temperature: 0,
    top_k: 1,
    seed: 0,
    camelid_dense_diagnostics: true,
    camelid_dense_diagnostic_generated_index: 0,
    stream: false,
  })
  return r.camelid?.prompt_token_ids ?? null
}

function arraysEqual(a, b) {
  return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((v, i) => v === b[i])
}

async function capture() {
  const captured = []
  for (const userContent of PROMPTS) {
    const chatml = renderChatML(userContent)
    const perCount = {}
    for (const n of tokenCounts) {
      const ref = await referenceCompletion(chatml, n)
      perCount[n] = { reference_text: ref.text, reference_tokens: ref.tokens }
      process.stderr.write(`captured ${JSON.stringify(userContent)} n=${n}: ${JSON.stringify(ref.text)}\n`)
    }
    captured.push({ prompt: userContent, chatml_rendered: chatml, generations: perCount })
  }
  const oracle = {
    schema: 'camelid.qwen3.chatml_oracle.v1',
    comparator: comparatorLabel,
    llama_base: llamaBase,
    token_counts: tokenCounts,
    prompts: PROMPTS,
    captured,
  }
  await mkdir(dirname(oraclePath), { recursive: true })
  await writeFile(oraclePath, JSON.stringify(oracle, null, 2))
  process.stderr.write(`\nwrote oracle ${oraclePath}\n`)
}

async function compare() {
  const oracle = JSON.parse(await readFile(oraclePath, 'utf8'))
  const results = []
  let allPass = true
  for (const cap of oracle.captured) {
    const userContent = cap.prompt
    const chatml = cap.chatml_rendered
    const referencePromptTokens = await encodeCamelid(chatml, true)
    const camelidPromptTokens = await camelidChatPromptTokens(userContent)
    const promptMatch = arraysEqual(referencePromptTokens, camelidPromptTokens)

    const perCount = {}
    for (const n of oracle.token_counts) {
      const ref = cap.generations[n]
      const camText = await camelidChat(userContent, n)
      const camTokens = await encodeCamelid(camText, false)
      const STOP = new Set([151645, 151643])
      const refContentTokens = [...ref.reference_tokens]
      while (refContentTokens.length && STOP.has(refContentTokens[refContentTokens.length - 1])) refContentTokens.pop()
      const textMatch = ref.reference_text === camText
      const tokenMatch = arraysEqual(refContentTokens, camTokens)
      perCount[n] = {
        reference_text: ref.reference_text,
        reference_tokens: ref.reference_tokens,
        reference_content_tokens: refContentTokens,
        camelid_text: camText,
        camelid_content_tokens: camTokens,
        text_match: textMatch,
        token_match: tokenMatch,
        stopped_early_at_eos: ref.reference_tokens.length < n,
      }
      if (!textMatch || !tokenMatch) allPass = false
    }
    if (!promptMatch) allPass = false
    results.push({
      prompt: userContent,
      chatml_rendered: chatml,
      reference_prompt_tokens: referencePromptTokens,
      camelid_prompt_tokens: camelidPromptTokens,
      prompt_token_match: promptMatch,
      generations: perCount,
    })
  }

  const report = {
    schema: 'camelid.qwen3.chatml_chat_parity.v1',
    row_id: rowId,
    display_name: displayName,
    mode: 'chatml_thinking_disabled_greedy',
    capture_method: 'two_phase_oracle',
    comparator: oracle.comparator || comparatorLabel,
    camelid_base: camelidBase,
    llama_base: oracle.llama_base,
    token_counts: oracle.token_counts,
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
    process.stderr.write(`  prompt-token parity: ${r.prompt_token_match ? 'PASS' : 'FAIL'}\n`)
    for (const n of oracle.token_counts) {
      const g = r.generations[n]
      process.stderr.write(
        `  n=${n}: text ${g.text_match ? 'PASS' : 'FAIL'} | tokens ${g.token_match ? 'PASS' : 'FAIL'} | camelid=${JSON.stringify(g.camelid_text)}\n`,
      )
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

if (!oraclePath) {
  process.stderr.write('error: --oracle <path> is required\n')
  process.exitCode = 2
} else if (mode === 'capture') {
  await capture()
} else if (mode === 'compare') {
  await compare()
} else {
  process.stderr.write('error: --mode capture|compare is required\n')
  process.exitCode = 2
}
