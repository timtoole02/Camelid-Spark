#!/usr/bin/env node
// Local Inference Conformance Suite
//
// Measures every runtime by the same ruler. For a fixed GGUF model and a fixed
// raw-completion prompt set (greedy, no chat template — template rendering is a
// separate axis), each runtime is probed for:
//
//   1. determinism   — the same request, repeated: are outputs identical?
//   2. agreement     — pairwise first-divergence depth between runtimes on the
//                      exact same model bytes (token-level when both report
//                      token ids, character-level otherwise)
//   3. tokenization  — prompt token ids from each runtime's /tokenize endpoint,
//                      compared pairwise (where exposed)
//   4. provability   — can the runtime emit a sealed, independently verifiable
//                      record of what it computed (a receipt)?
//
// No runtime is treated as ground truth: results are an agreement matrix, not a
// scoreboard against a privileged reference. Servers run sequentially (one at a
// time) so memory pressure never contaminates a measurement.
//
// Usage:
//   node tools/conformance/run.mjs \
//     --model /path/to/model.Q8_0.gguf \
//     --camelid-bin /path/to/camelid \
//     --llama-server brew=/opt/homebrew/bin/llama-server \
//     --llama-server pinned=/path/to/reference/llama-server-wrapper \
//     --ollama \
//     --max-tokens 64 --rounds 3 --out conformance-out
//
// Each --llama-server takes label=path (repeatable). --ollama uses the local
// daemon (spawned if absent) with a generated Modelfile over the same GGUF.

import { execFile, spawn } from 'node:child_process'
import { mkdir, writeFile, readFile } from 'node:fs/promises'
import { basename, resolve } from 'node:path'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)

const args = []
for (let i = 2; i < process.argv.length; i += 1) {
  const arg = process.argv[i]
  if (!arg.startsWith('--')) continue
  const [key, inline] = arg.slice(2).split('=', 2)
  const value = inline ?? (process.argv[i + 1]?.startsWith('--') ? 'true' : process.argv[++i] ?? 'true')
  args.push([key, value])
}
const argOne = key => args.find(([k]) => k === key)?.[1]
const argAll = key => args.filter(([k]) => k === key).map(([, v]) => v)

const modelPath = resolve(argOne('model') ?? (() => { throw new Error('--model <gguf> is required') })())
const modelLabel = argOne('model-id') || basename(modelPath)
const camelidBin = argOne('camelid-bin') || 'camelid'
const includeCamelid = !argAll('skip').includes('camelid')
const ollamaWanted = argOne('ollama') === 'true'
const maxTokens = Number.parseInt(argOne('max-tokens') || '64', 10)
const rounds = Number.parseInt(argOne('rounds') || '3', 10)
const outDir = resolve(argOne('out') || 'conformance-out')
const basePort = Number.parseInt(argOne('base-port') || '8731', 10)
const receiptReference = argOne('receipt-reference') // optional llama-server for full receipt verification

// Raw completion prompts. Deliberately template-free: every engine sees the
// identical character stream, so divergence isolates tokenizer + numerics.
const PROMPTS = [
  {
    id: 'qa',
    text: 'The three primary reasons the sky appears blue during the day are',
  },
  {
    id: 'list',
    text: 'A complete alphabetical list of the eight planets of the solar system: 1.',
  },
  {
    id: 'repeat',
    text: 'Repeat the following sentence exactly five times: The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.',
  },
]

const sleep = ms => new Promise(r => setTimeout(r, ms))

async function waitHealthy(url, attempts = 240) {
  for (let i = 0; i < attempts; i += 1) {
    try {
      const res = await fetch(url)
      if (res.ok) return
    } catch {}
    await sleep(1000)
  }
  throw new Error(`server at ${url} never became healthy`)
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  const text = await res.text()
  if (!res.ok) throw new Error(`${url} -> ${res.status}: ${text.slice(0, 300)}`)
  return JSON.parse(text)
}

function firstDivergence(a, b) {
  const limit = Math.max(a.length, b.length)
  for (let i = 0; i < limit; i += 1) {
    if (a[i] !== b[i]) return i
  }
  return -1
}

// ---------------------------------------------------------------------------
// Runtime adapters. Each yields:
//   { label, version, completions: {promptId: {rounds: [{text, tokens|null}]}},
//     tokenize: {promptId: tokenIds|null}, proof: {...}|null }
// ---------------------------------------------------------------------------

function spawnProcess(bin, argv, label) {
  const child = spawn(bin, argv, { stdio: ['ignore', 'pipe', 'pipe'] })
  let stderr = ''
  child.stderr.on('data', d => { stderr += d })
  child.once('error', err => {
    console.error(`[${label}] spawn error: ${err}`)
  })
  return { child, getStderr: () => stderr }
}

async function killAndWait(child) {
  if (!child || child.exitCode !== null) return
  child.kill('SIGTERM')
  await new Promise(r => {
    const t = setTimeout(() => { child.kill('SIGKILL'); r() }, 8000)
    child.once('exit', () => { clearTimeout(t); r() })
  })
  await sleep(1500)
}

async function probeCamelid(port) {
  const base = `http://127.0.0.1:${port}`
  console.error(`[camelid] starting on :${port} ...`)
  const { child } = spawnProcess(camelidBin, ['serve', '--addr', `127.0.0.1:${port}`, '--model', modelPath], 'camelid')
  try {
    await waitHealthy(`${base}/health`)
    let version = 'camelid'
    try {
      const caps = await (await fetch(`${base}/api/capabilities`)).json()
      version = `camelid ${caps.version ?? ''}`.trim()
    } catch {}

    const completions = {}
    for (const prompt of PROMPTS) {
      const roundsOut = []
      for (let r = 0; r < rounds; r += 1) {
        const body = await postJson(`${base}/v1/completions`, {
          prompt: prompt.text,
          max_tokens: maxTokens,
          temperature: 0,
        })
        roundsOut.push({
          text: body.choices[0].text,
          tokens: body.camelid?.generated_token_ids ?? null,
        })
      }
      completions[prompt.id] = { rounds: roundsOut }
      console.error(`[camelid] ${prompt.id}: ${rounds} rounds done`)
    }

    const tokenize = {}
    for (const prompt of PROMPTS) {
      try {
        const body = await postJson(`${base}/tokenize`, { content: prompt.text })
        tokenize[prompt.id] = body.tokens ?? null
      } catch {
        tokenize[prompt.id] = null
      }
    }

    // Provability probe: emit a sealed receipt for one greedy request over the
    // SAME raw-completions path the agreement matrix uses, then verify its
    // self-digest (and the full replay + reference chain when a reference binary
    // was supplied).
    let proof = null
    try {
      const receiptResp = await postJson(`${base}/v1/completions`, {
        prompt: PROMPTS[0].text,
        max_tokens: Math.min(maxTokens, 16),
        temperature: 0,
        camelid_receipt: true,
      })
      const receipt = receiptResp.camelid_receipt ?? receiptResp.camelid?.receipt
      if (receipt) {
        const receiptPath = resolve(outDir, 'camelid-receipt.json')
        await writeFile(receiptPath, JSON.stringify(receipt, null, 2))
        const verifyArgs = ['verify-receipt', receiptPath, '--gguf', modelPath]
        if (receiptReference) verifyArgs.push('--llama-server', receiptReference)
        else verifyArgs.push('--self-only')
        let verified = false
        let mode = receiptReference
          ? 'full chain (self-digest → lane identity → replay → independent reference re-run)'
          : 'self checks (digest, lane identity, deterministic replay)'
        try {
          await execFileAsync(camelidBin, verifyArgs, { timeout: 600000 })
          verified = true
        } catch (err) {
          verified = false
          mode += ` (exit ${err.code})`
        }
        proof = { receipt_emitted: true, receipt_id: receipt.receipt_id, verified, mode }
      } else {
        proof = { receipt_emitted: false }
      }
    } catch (err) {
      proof = { receipt_emitted: false, error: String(err).slice(0, 200) }
    }

    return { label: 'camelid', version, completions, tokenize, proof }
  } finally {
    await killAndWait(child)
  }
}

async function probeLlamaServer(label, bin, port) {
  const base = `http://127.0.0.1:${port}`
  let version = label
  try {
    const { stdout, stderr } = await execFileAsync(bin, ['--version'], { timeout: 20000 })
    version = `${(stdout + stderr).split('\n')[0]}`.trim()
  } catch {}
  console.error(`[${label}] starting on :${port} ...`)
  const { child } = spawnProcess(bin, [
    '--host', '127.0.0.1', '--port', String(port),
    '-m', modelPath, '-c', '2048', '--no-warmup',
  ], label)
  try {
    await waitHealthy(`${base}/health`)
    const completions = {}
    for (const prompt of PROMPTS) {
      const roundsOut = []
      for (let r = 0; r < rounds; r += 1) {
        const body = await postJson(`${base}/completion`, {
          prompt: prompt.text,
          n_predict: maxTokens,
          temperature: 0,
          top_k: 1,
          seed: 0,
          cache_prompt: false,
          return_tokens: true,
        })
        roundsOut.push({
          text: body.content,
          tokens: Array.isArray(body.tokens) && body.tokens.length > 0 ? body.tokens : null,
        })
      }
      completions[prompt.id] = { rounds: roundsOut }
      console.error(`[${label}] ${prompt.id}: ${rounds} rounds done`)
    }

    const tokenize = {}
    for (const prompt of PROMPTS) {
      try {
        const body = await postJson(`${base}/tokenize`, { content: prompt.text })
        tokenize[prompt.id] = body.tokens ?? null
      } catch {
        tokenize[prompt.id] = null
      }
    }

    return { label, version, completions, tokenize, proof: null }
  } finally {
    await killAndWait(child)
  }
}

async function probeOllama(port) {
  const base = 'http://127.0.0.1:11434'
  let daemon = null
  let healthy = false
  try {
    healthy = (await fetch(base)).ok
  } catch {}
  if (!healthy) {
    console.error('[ollama] daemon not running; spawning ollama serve ...')
    daemon = spawnProcess('ollama', ['serve'], 'ollama').child
    await waitHealthy(base)
  }
  let version = 'ollama'
  try {
    const v = await (await fetch(`${base}/api/version`)).json()
    version = `ollama ${v.version}`
  } catch {}

  // Import the exact GGUF through a Modelfile; raw mode bypasses templating.
  const tag = 'conformance-probe'
  const modelfilePath = resolve(outDir, 'Modelfile.conformance')
  await writeFile(modelfilePath, `FROM ${modelPath}\n`)
  console.error(`[ollama] creating model ${tag} from GGUF ...`)
  await execFileAsync('ollama', ['create', tag, '-f', modelfilePath], { timeout: 600000 })

  try {
    const completions = {}
    for (const prompt of PROMPTS) {
      const roundsOut = []
      for (let r = 0; r < rounds; r += 1) {
        const body = await postJson(`${base}/api/generate`, {
          model: tag,
          prompt: prompt.text,
          raw: true,
          stream: false,
          options: { temperature: 0, seed: 0, top_k: 1, num_predict: maxTokens },
        })
        roundsOut.push({ text: body.response, tokens: null })
      }
      completions[prompt.id] = { rounds: roundsOut }
      console.error(`[ollama] ${prompt.id}: ${rounds} rounds done`)
    }
    const tokenize = {}
    for (const prompt of PROMPTS) tokenize[prompt.id] = null // no tokenize endpoint
    return { label: 'ollama', version, completions, tokenize, proof: null }
  } finally {
    try { await execFileAsync('ollama', ['stop', tag], { timeout: 30000 }) } catch {}
    if (daemon) await killAndWait(daemon)
  }
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

function scoreDeterminism(runtime) {
  const perPrompt = {}
  for (const prompt of PROMPTS) {
    const rs = runtime.completions[prompt.id].rounds
    let identical = true
    let detail = null
    for (let i = 1; i < rs.length; i += 1) {
      if (rs[i].tokens && rs[0].tokens) {
        const d = firstDivergence(rs[0].tokens, rs[i].tokens)
        if (d !== -1) { identical = false; detail = `round ${i} diverges at token ${d}`; break }
      } else if (rs[i].text !== rs[0].text) {
        identical = false
        detail = `round ${i} diverges at char ${firstDivergence(rs[0].text, rs[i].text)}`
        break
      }
    }
    perPrompt[prompt.id] = { identical, detail }
  }
  return perPrompt
}

function scoreAgreement(runtimes) {
  const pairs = []
  for (let i = 0; i < runtimes.length; i += 1) {
    for (let j = i + 1; j < runtimes.length; j += 1) {
      const a = runtimes[i]
      const b = runtimes[j]
      const perPrompt = {}
      for (const prompt of PROMPTS) {
        const ra = a.completions[prompt.id].rounds[0]
        const rb = b.completions[prompt.id].rounds[0]
        if (ra.tokens && rb.tokens) {
          const d = firstDivergence(ra.tokens, rb.tokens)
          perPrompt[prompt.id] = {
            level: 'token',
            first_divergence: d,
            compared: Math.max(ra.tokens.length, rb.tokens.length),
          }
        } else {
          const d = firstDivergence(ra.text, rb.text)
          perPrompt[prompt.id] = {
            level: 'text',
            first_divergence: d,
            compared: Math.max(ra.text.length, rb.text.length),
          }
        }
      }
      pairs.push({ a: a.label, b: b.label, perPrompt })
    }
  }
  return pairs
}

function scoreTokenizers(runtimes) {
  const pairs = []
  for (let i = 0; i < runtimes.length; i += 1) {
    for (let j = i + 1; j < runtimes.length; j += 1) {
      const a = runtimes[i]
      const b = runtimes[j]
      const perPrompt = {}
      for (const prompt of PROMPTS) {
        const ta = a.tokenize[prompt.id]
        const tb = b.tokenize[prompt.id]
        perPrompt[prompt.id] = ta && tb ? firstDivergence(ta, tb) : null
      }
      pairs.push({ a: a.label, b: b.label, perPrompt })
    }
  }
  return pairs
}

function renderScoreboard(results) {
  const lines = []
  lines.push(`# Local Inference Conformance Results`)
  lines.push('')
  lines.push(`Model: \`${results.model.label}\` (sha256 \`${results.model.sha256.slice(0, 16)}…\`)`)
  lines.push(`Greedy raw completions, max_tokens ${maxTokens}, ${rounds} determinism rounds per prompt. No runtime is treated as ground truth.`)
  lines.push('')
  lines.push(`## Runtimes`)
  lines.push('')
  for (const r of results.runtimes) lines.push(`- **${r.label}** — ${r.version}`)
  for (const u of results.unavailable) lines.push(`- _${u.label}_ — unavailable: ${u.reason}`)
  lines.push('')
  lines.push(`## Determinism (same request, ${rounds} rounds)`)
  lines.push('')
  lines.push(`| runtime | ${PROMPTS.map(p => p.id).join(' | ')} |`)
  lines.push(`|---|${PROMPTS.map(() => '---').join('|')}|`)
  for (const r of results.runtimes) {
    const cells = PROMPTS.map(p => {
      const d = results.determinism[r.label][p.id]
      return d.identical ? '✅ identical' : `❌ ${d.detail}`
    })
    lines.push(`| ${r.label} | ${cells.join(' | ')} |`)
  }
  lines.push('')
  lines.push(`## Cross-runtime agreement (first divergence; -1 = full agreement)`)
  lines.push('')
  lines.push(`| pair | ${PROMPTS.map(p => p.id).join(' | ')} |`)
  lines.push(`|---|${PROMPTS.map(() => '---').join('|')}|`)
  for (const pair of results.agreement) {
    const cells = PROMPTS.map(p => {
      const d = pair.perPrompt[p.id]
      const mark = d.first_divergence === -1 ? '✅ -1' : `${d.first_divergence}`
      return `${mark} /${d.compared} ${d.level === 'token' ? 'tok' : 'chr'}`
    })
    lines.push(`| ${pair.a} ↔ ${pair.b} | ${cells.join(' | ')} |`)
  }
  lines.push('')
  lines.push(`## Tokenizer agreement (prompt token ids; -1 = identical, n/a = endpoint absent)`)
  lines.push('')
  lines.push(`| pair | ${PROMPTS.map(p => p.id).join(' | ')} |`)
  lines.push(`|---|${PROMPTS.map(() => '---').join('|')}|`)
  for (const pair of results.tokenizers) {
    const cells = PROMPTS.map(p => {
      const d = pair.perPrompt[p.id]
      return d === null ? 'n/a' : d === -1 ? '✅ -1' : String(d)
    })
    lines.push(`| ${pair.a} ↔ ${pair.b} | ${cells.join(' | ')} |`)
  }
  lines.push('')
  lines.push(`## Provability`)
  lines.push('')
  lines.push(`| runtime | sealed receipt | verification |`)
  lines.push(`|---|---|---|`)
  for (const r of results.runtimes) {
    if (r.proof?.receipt_emitted) {
      lines.push(`| ${r.label} | ✅ \`${r.proof.receipt_id.slice(0, 16)}…\` | ${r.proof.verified ? '✅' : '❌'} ${r.proof.mode} |`)
    } else {
      lines.push(`| ${r.label} | — none | — |`)
    }
  }
  lines.push('')
  lines.push(`Generated by tools/conformance/run.mjs`)
  return lines.join('\n')
}

// ---------------------------------------------------------------------------

async function main() {
  await mkdir(outDir, { recursive: true })
  const { stdout: shaOut } = await execFileAsync('shasum', ['-a', '256', modelPath], { timeout: 600000 })
  const modelSha = shaOut.split(/\s+/)[0]

  const runtimes = []
  const unavailable = []
  let port = basePort
  // One runtime failing to start, load, or probe must never sink the whole
  // scoreboard — it is recorded as unavailable and the rest still produce
  // results. (A near-full disk, a missing binary, an OOM are all findings, not
  // crashes.)
  const tryProbe = async (label, fn) => {
    try {
      runtimes.push(await fn())
    } catch (err) {
      console.error(`[${label}] unavailable: ${String(err).slice(0, 300)}`)
      unavailable.push({ label, reason: String(err).slice(0, 300) })
    }
  }
  if (includeCamelid) {
    await tryProbe('camelid', () => probeCamelid(port++))
  }
  for (const spec of argAll('llama-server')) {
    const [label, bin] = spec.includes('=') ? spec.split(/=(.*)/s) : [`llama-${port}`, spec]
    await tryProbe(label, () => probeLlamaServer(label, bin, port++))
  }
  if (ollamaWanted) {
    await tryProbe('ollama', () => probeOllama(port++))
  }
  if (runtimes.length < 2) {
    console.error('warning: fewer than two runtimes probed; agreement matrix will be empty')
  }

  const results = {
    schema: 'camelid.conformance/v1',
    created_utc: new Date().toISOString(),
    model: { label: modelLabel, path: modelPath, sha256: modelSha },
    settings: { max_tokens: maxTokens, rounds, prompts: PROMPTS },
    runtimes: runtimes.map(r => ({ label: r.label, version: r.version, proof: r.proof })),
    unavailable,
    determinism: Object.fromEntries(runtimes.map(r => [r.label, scoreDeterminism(r)])),
    agreement: scoreAgreement(runtimes),
    tokenizers: scoreTokenizers(runtimes),
    raw: runtimes,
  }

  await writeFile(resolve(outDir, 'results.json'), JSON.stringify(results, null, 2))
  const scoreboard = renderScoreboard(results)
  await writeFile(resolve(outDir, 'SCOREBOARD.md'), scoreboard)
  console.log(scoreboard)
  // The scoreboard is the product. Once it is written, the run succeeded —
  // unavailable runtimes are recorded findings, not failures. Exit explicitly
  // (code 2 only when fewer than two runtimes were probed, so no agreement
  // matrix exists) so a late child-process 'exit' event in the loop cannot
  // turn a complete scoreboard into a spurious failure.
  process.exit(runtimes.length >= 2 ? 0 : 2)
}

main().catch(err => {
  console.error(err)
  process.exit(1)
})
