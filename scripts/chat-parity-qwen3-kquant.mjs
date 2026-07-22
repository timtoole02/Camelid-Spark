#!/usr/bin/env node
// Qwen3 ChatML chat-parity harness — K-QUANT variant (Q4_K_M / Q6_K mixed GGUFs).
//
// Derived from scripts/chat-parity-qwen3.mjs. Identical substantive parity claim
// (generated-token + generated-text parity at 1/5/50 tokens, camelid vs a pinned
// llama.cpp reference fed the same ChatML string), with ONE deliberate change:
//
//   The original harness reads camelid's internally-rendered prompt tokens via
//   /v1/chat/completions with `camelid_dense_diagnostics: true`. That diagnostic
//   path runs the CPU f32 LINEAR (linear_for_role, collect_diagnostics=true), which
//   materialises f32 weights. A K-quant model is loaded WIRE-ONLY for the resident
//   GPU engine (load_kquant_wire_linear: empty f32 `data`), so the diagnostic path
//   503s ("no-row-major-data ... data_len=0"). The GENERATION path (plain chat) runs
//   on the GPU q4k_gemv/q6k_gemv engine and is unaffected.
//
//   Prompt-token parity is therefore checked CROSS-ENGINE instead, which is strictly
//   stronger than the original self-consistency check: camelid's tokenizer encoding
//   of the rendered ChatML must equal llama.cpp's /tokenize of the same string (both
//   parse the <|im_*|>/<think> specials). The ChatML renderer + tokenizer are
//   quant-independent (arch=qwen3), so this leg is identical to the Q8_0 row.
//
// Usage mirrors chat-parity-qwen3.mjs (--camelid --llama --model-id --row-id
// --display-name --comparator --out --token-counts).

import { writeFile, mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'

const args = parseArgs(process.argv.slice(2))
const camelidBase = (args.get('camelid') || 'http://127.0.0.1:8185').replace(/\/$/, '')
const llamaBase = (args.get('llama') || 'http://127.0.0.1:8090').replace(/\/$/, '')
const modelId = args.get('model-id') || 'Qwen3 4B Instruct Q4_K_M'
const rowId = args.get('row-id') || 'qwen3_4b_instruct_q4_k_m'
const displayName = args.get('display-name') || 'Qwen3 4B Instruct Q4_K_M'
const comparatorLabel = args.get('comparator') || 'llama.cpp /completion (ChatML specials parsed)'
const outPath = args.get('out') || null
const tokenCounts = (args.get('token-counts') || '1,5,50').split(',').map((s) => Number.parseInt(s.trim(), 10))

const PROMPTS = JSON.parse(
  args.get('prompts-json') ||
    JSON.stringify(['What is the capital of France?', 'Name a primary color.', 'Say hello.']),
)

// ChatML, thinking DISABLED — must match render_qwen3_chatml_prompt in src/api/mod.rs.
function renderChatML(userContent) {
  return `<|im_start|>user\n${userContent}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
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

async function encodeCamelid(text, parseSpecial) {
  const r = await postJson(camelidBase, '/api/models/tokenizer/encode', {
    text,
    add_special: false,
    parse_special: parseSpecial,
  })
  return r.tokens
}

async function tokenizeLlama(text) {
  // llama-server /tokenize parses the ChatML specials when add_special:false.
  const r = await postJson(llamaBase, '/tokenize', { content: text, add_special: false })
  return r.tokens
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

function arraysEqual(a, b) {
  return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((v, i) => v === b[i])
}

async function main() {
  const results = []
  let allPass = true
  for (const userContent of PROMPTS) {
    const chatml = renderChatML(userContent)
    // Cross-engine prompt-token parity (replaces the f32-diagnostic self-check).
    const camelidPromptTokens = await encodeCamelid(chatml, true)
    const referencePromptTokens = await tokenizeLlama(chatml)
    const promptMatch = arraysEqual(referencePromptTokens, camelidPromptTokens)

    const perCount = {}
    for (const n of tokenCounts) {
      const ref = await referenceCompletion(chatml, n)
      const camText = await camelidChat(userContent, n)
      const camTokens = await encodeCamelid(camText, false)
      const STOP = new Set([151645, 151643])
      const refContentTokens = [...ref.tokens]
      while (refContentTokens.length && STOP.has(refContentTokens[refContentTokens.length - 1])) {
        refContentTokens.pop()
      }
      const textMatch = ref.text === camText
      const tokenMatch = arraysEqual(refContentTokens, camTokens)
      const stoppedEarly = ref.tokens.length < n
      perCount[n] = {
        reference_text: ref.text,
        reference_tokens: ref.tokens,
        reference_content_tokens: refContentTokens,
        camelid_text: camText,
        camelid_content_tokens: camTokens,
        text_match: textMatch,
        token_match: tokenMatch,
        stopped_early_at_eos: stoppedEarly,
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
      prompt_token_check: 'cross_engine_tokenize (camelid encode == llama /tokenize); f32-diagnostic path N/A for wire-only K-quant',
      generations: perCount,
    })
  }

  const report = {
    schema: 'camelid.qwen3.chatml_chat_parity.v1',
    variant: 'kquant_gpu_resident',
    row_id: rowId,
    display_name: displayName,
    mode: 'chatml_thinking_disabled_greedy',
    comparator: comparatorLabel,
    proof_chain: 'camelid GPU-resident CUDA decode (q4k_gemv/q6k_gemv) == llama.cpp. The GPU==cpu_reference middle leg used for Q8_0 is N/A: K-quant has no CPU decode path yet (Phase 2), so llama.cpp is the direct reference.',
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
    process.stderr.write(`  prompt-token parity (cross-engine): ${r.prompt_token_match ? 'PASS' : 'FAIL'}\n`)
    for (const n of tokenCounts) {
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

await main()
