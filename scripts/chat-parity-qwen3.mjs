#!/usr/bin/env node
// Qwen3 ChatML chat-parity harness (Gate 3, feat/qwen3-support).
//
// Compares camelid's chat path against a pinned llama.cpp reference for the
// Qwen3-1.7B Q8_0 exact row, greedy, with THINKING DISABLED (the deterministic
// parity-locked mode: the rendered ChatML generation prompt carries the empty
// `<think>\n\n</think>\n\n` block, so the model answers directly).
//
// For each prompt it checks, at 1 / 5 / 50 generated tokens:
//   - prompt-token parity  (camelid-rendered ChatML tokens == reference tokens)
//   - generated-token parity (camelid generated ids == reference ids)
//   - generated-text parity
//
// camelid is driven via /v1/chat/completions (which uses the hardcoded ChatML
// renderer); the reference is driven via llama.cpp /completion fed the same
// ChatML string (llama-server parses the <|im_*|> / <think> specials).
//
// Usage:
//   node scripts/chat-parity-qwen3.mjs \
//     --camelid http://127.0.0.1:8185 --llama http://127.0.0.1:8090 \
//     --model-id "Qwen3 1.7B Instruct" --out qa/.../qwen3-chat-parity.json
//
// Defaults read CAMELID_API_BASE / QWEN3_LLAMA_URL / QWEN3_MODEL_ID.

import { writeFile, mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'

const args = parseArgs(process.argv.slice(2))
const camelidBase = (args.get('camelid') || process.env.CAMELID_API_BASE || 'http://127.0.0.1:8185').replace(/\/$/, '')
const llamaBase = (args.get('llama') || process.env.QWEN3_LLAMA_URL || 'http://127.0.0.1:8090').replace(/\/$/, '')
const modelId = args.get('model-id') || process.env.QWEN3_MODEL_ID || 'Qwen3 1.7B Instruct'
const rowId = args.get('row-id') || process.env.QWEN3_ROW_ID || 'qwen3_1_7b_instruct_q8_0'
const displayName = args.get('display-name') || process.env.QWEN3_DISPLAY_NAME || 'Qwen3 1.7B Instruct Q8_0'
const comparatorLabel =
  args.get('comparator') ||
  process.env.QWEN3_COMPARATOR ||
  'llama.cpp /completion (ChatML specials parsed), -ctk f32 -ctv f32 -fa off --no-repack'
const outPath = args.get('out') || process.env.QWEN3_CHAT_PARITY_OUT || null
const tokenCounts = (args.get('token-counts') || '1,5,50').split(',').map((s) => Number.parseInt(s.trim(), 10))

// The three fixed chat prompts. Single user turn each (the supported shape).
const PROMPTS = JSON.parse(
  args.get('prompts-json') ||
    process.env.QWEN3_CHAT_PROMPTS_JSON ||
    JSON.stringify([
      'What is the capital of France?',
      'Name a primary color.',
      'Say hello.',
    ]),
)

// ChatML, thinking DISABLED — must match render_qwen3_chatml_prompt in src/api/mod.rs.
function renderChatML(userContent) {
  return (
    `<|im_start|>user\n${userContent}<|im_end|>\n` +
    `<|im_start|>assistant\n<think>\n\n</think>\n\n`
  )
}

async function postJson(base, path, body) {
  const res = await fetch(`${base}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    throw new Error(`${base}${path} -> HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`)
  }
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

async function referenceCompletion(promptText, nPredict) {
  // llama.cpp /completion parses the ChatML specials in the prompt.
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

async function camelidChatPromptTokens(userContent) {
  // Dense diagnostics expose the rendered prompt token ids.
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

async function main() {
  const results = []
  let allPass = true
  for (const userContent of PROMPTS) {
    const chatml = renderChatML(userContent)
    const referencePromptTokens = await encodeCamelid(chatml, true)
    const camelidPromptTokens = await camelidChatPromptTokens(userContent)
    const promptMatch = arraysEqual(referencePromptTokens, camelidPromptTokens)

    const perCount = {}
    for (const n of tokenCounts) {
      const ref = await referenceCompletion(chatml, n)
      const camText = await camelidChat(userContent, n)
      // camelid's chat endpoint returns text only; its generated CONTENT token
      // ids are the canonical encoding of that text. The reference's generated
      // tokens may carry a trailing EOS/<|im_end|> (151645) that is stripped from
      // the detokenized text, so compare against the reference's CONTENT tokens
      // (trailing stop tokens removed).
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
      generations: perCount,
    })
  }

  const report = {
    schema: 'camelid.qwen3.chatml_chat_parity.v1',
    row_id: rowId,
    display_name: displayName,
    mode: 'chatml_thinking_disabled_greedy',
    comparator: comparatorLabel,
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
  // Human summary to stderr; machine JSON to stdout.
  for (const r of results) {
    process.stderr.write(`\n=== ${JSON.stringify(r.prompt)} ===\n`)
    process.stderr.write(`  prompt-token parity: ${r.prompt_token_match ? 'PASS' : 'FAIL'}\n`)
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
      if (next === undefined || next.startsWith('--')) {
        map.set(key, 'true')
      } else {
        map.set(key, next)
        i++
      }
    }
  }
  return map
}

await main()
