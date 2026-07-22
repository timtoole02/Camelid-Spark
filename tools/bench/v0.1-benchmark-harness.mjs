#!/usr/bin/env node
import { createHash } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises'
import os from 'node:os'
import { dirname, isAbsolute, join, relative, resolve } from 'node:path'
import { performance } from 'node:perf_hooks'
import { spawn, spawnSync } from 'node:child_process'

const HARNESS_SCHEMA = 'camelid.v0_1.benchmark_harness.config.v1'
const BUNDLE_SCHEMA = 'camelid.v0_1.benchmark_bundle.v1'
const PLAN_SCHEMA = 'camelid.v0_1.benchmark_plan.v1'
const SUPPORTED_ENGINES = new Set(['camelid', 'llama.cpp', 'ollama', 'mlx'])
const DEFAULT_OUT_ROOT = 'qa/evidence-bundles/v0.1'
const DEFAULT_TIMEOUT_MS = 10 * 60 * 1000
const RSS_POLL_MS = 250
const REQUIRED_METADATA_FIELDS = ['release_version', 'benchmark_name', 'operator', 'purpose']
const REQUIRED_MODEL_FIELDS = ['id', 'label', 'family', 'parameters', 'quantization', 'artifact_uri']
const REQUIRED_ENTRY_FIELDS = ['id', 'label', 'engine', 'model_id', 'prompt', 'command']

const args = parseArgs(process.argv.slice(2))

if (args.has('help') || args.has('h')) {
  console.log(usage())
  process.exit(0)
}

const root = resolve(args.get('root') || process.cwd())
const configPath = args.get('config')
if (!configPath) {
  console.error('error: --config is required')
  console.error(usageSummary())
  process.exit(2)
}

const config = await readConfig(resolve(root, configPath))
validateConfig(config)

const selectedEntries = filterEntries(config.entries, valuesFor(args, 'entry'))
const timestamp = args.get('timestamp') || timestampUtc()
const outRoot = resolve(root, args.get('out-root') || config.output?.root || DEFAULT_OUT_ROOT)
const bundleDir = resolve(outRoot, timestamp)
const dryRun = args.has('dry-run') || config.dry_run === true
const printPlan = args.has('print-plan')
const hashModels = args.has('hash-models') || config.model_manifest?.hash_models === true

const machine = collectMachine(root)
const modelManifest = await buildModelManifest(config, root, hashModels)
const plan = buildPlan({
  configPath: resolve(root, configPath),
  config,
  entries: selectedEntries,
  root,
  bundleDir,
  timestamp,
  dryRun,
  hashModels,
  machine,
  modelManifest,
})

if (printPlan) {
  console.log(`${JSON.stringify(plan, null, 2)}\n`)
  process.exit(0)
}

await writeBundle(plan)

async function writeBundle(plan) {
  const rawLogsDir = join(plan.bundle_dir, 'raw_logs')
  await mkdir(rawLogsDir, { recursive: true })

  const commands = renderCommandsMarkdown(plan)
  await writeJson(join(plan.bundle_dir, 'machine.json'), plan.machine)
  await writeJson(join(plan.bundle_dir, 'model_manifest.json'), plan.model_manifest)
  await writeFile(join(plan.bundle_dir, 'commands.md'), commands)

  const results = []
  for (const rawEntry of selectedEntries) {
    const normalizedEntry = plan.entries.find((entry) => entry.id === rawEntry.id)
    const entryResult = await runEntry(rawEntry, normalizedEntry, plan, rawLogsDir)
    results.push(entryResult)
  }

  const bundle = {
    schema: BUNDLE_SCHEMA,
    generated_utc: new Date().toISOString(),
    harness: plan.harness,
    bundle_dir: plan.bundle_dir,
    metadata: plan.metadata,
    machine: plan.machine,
    model_manifest: plan.model_manifest,
    dry_run: plan.dry_run,
    entries: plan.entries.map((entry) => publicEntry(entry)),
    results,
    summary: summarizeResults(results),
  }

  await writeJson(join(plan.bundle_dir, 'results.json'), bundle)
  await writeFile(join(plan.bundle_dir, 'results.csv'), renderResultsCsv(results))
  await writeFile(join(plan.bundle_dir, 'summary.md'), renderSummaryMarkdown(bundle))
  console.log(`bundle_dir=${plan.bundle_dir}`)
}

async function runEntry(rawEntry, entry, plan, rawLogsDir) {
  const repetitions = parsePositiveInt(rawEntry.repetitions ?? plan.defaults.repetitions, `entry ${rawEntry.id} repetitions`)
  const runs = []
  for (let idx = 0; idx < repetitions; idx += 1) {
    if (plan.dry_run) {
      runs.push(await writeDryRunLog(rawEntry, idx + 1, rawLogsDir, plan))
    } else {
      runs.push(await runCommand(rawEntry, idx + 1, rawLogsDir, plan))
    }
  }

  return {
    entry_id: entry.id,
    label: entry.label,
    engine: entry.engine,
    model_id: entry.model_id,
    prompt_sha256: sha256Text(String(entry.prompt)),
    repetitions,
    status: entryStatus(runs),
    runs,
    summary: summarizeRuns(runs),
  }
}

async function writeDryRunLog(entry, runIndex, rawLogsDir, plan) {
  const stem = `${sanitizeSlug(entry.id)}-run-${runIndex}`
  const planLog = join(rawLogsDir, `${stem}.plan.json`)
  const startedAt = new Date().toISOString()
  const command = normalizeCommand(entry, plan.defaults.timeout_ms, plan.root)
  const run = {
    run_index: runIndex,
    status: 'skipped',
    skipped_reason: 'dry_run',
    command: command.public,
    timing: {
      started_utc: startedAt,
      ended_utc: startedAt,
      duration_ms: 0,
      timeout_ms: command.timeout_ms,
      timed_out: false,
    },
    memory: {
      before: memorySnapshot(),
      after: memorySnapshot(),
      peak_rss_kb: null,
      rss_samples: [],
    },
    output: {
      stdout_log: null,
      stderr_log: null,
      stdout_bytes: 0,
      stderr_bytes: 0,
      raw_plan_log: relative(plan.root, planLog),
    },
    exit: {
      code: null,
      signal: null,
      error: null,
    },
  }
  await writeJson(planLog, run)
  return run
}

async function runCommand(entry, runIndex, rawLogsDir, plan) {
  const stem = `${sanitizeSlug(entry.id)}-run-${runIndex}`
  const stdoutPath = join(rawLogsDir, `${stem}.stdout.log`)
  const stderrPath = join(rawLogsDir, `${stem}.stderr.log`)
  const metaPath = join(rawLogsDir, `${stem}.command.json`)
  const command = normalizeCommand(entry, plan.defaults.timeout_ms, plan.root)
  const startedUtc = new Date().toISOString()
  const started = performance.now()
  const before = memorySnapshot()
  const rssSamples = []
  let peakRssKb = null
  let stdout = Buffer.alloc(0)
  let stderr = Buffer.alloc(0)
  let timedOut = false
  let spawnError = null
  let timeout = null
  let rssPoll = null

  await writeJson(metaPath, {
    entry_id: entry.id,
    run_index: runIndex,
    command: command.public,
    started_utc: startedUtc,
  })

  const outcome = await new Promise((resolveRun) => {
    let child
    try {
      child = spawn(command.file, command.args, {
        cwd: command.cwd,
        env: command.env,
        shell: command.shell,
        stdio: ['ignore', 'pipe', 'pipe'],
      })
    } catch (err) {
      spawnError = err
      resolveRun({ code: null, signal: null })
      return
    }

    timeout = setTimeout(() => {
      timedOut = true
      child.kill('SIGTERM')
      setTimeout(() => {
        if (child.exitCode === null) child.kill('SIGKILL')
      }, 2000).unref()
    }, command.timeout_ms)
    timeout.unref()

    rssPoll = setInterval(() => {
      const rssKb = readProcessRssKb(child.pid)
      if (rssKb !== null) {
        peakRssKb = peakRssKb === null ? rssKb : Math.max(peakRssKb, rssKb)
        if (rssSamples.length < 240) {
          rssSamples.push({ t_ms: Math.round(performance.now() - started), rss_kb: rssKb })
        }
      }
    }, RSS_POLL_MS)
    rssPoll.unref()

    child.stdout.on('data', (chunk) => { stdout = Buffer.concat([stdout, chunk]) })
    child.stderr.on('data', (chunk) => { stderr = Buffer.concat([stderr, chunk]) })
    child.once('error', (err) => { spawnError = err })
    child.once('close', (code, signal) => resolveRun({ code, signal }))
  })

  if (timeout) clearTimeout(timeout)
  if (rssPoll) clearInterval(rssPoll)
  const endedUtc = new Date().toISOString()
  const durationMs = round(performance.now() - started)

  if (spawnError) {
    stderr = Buffer.concat([stderr, Buffer.from(`${spawnError.name}: ${spawnError.message}\n`)])
  }
  await writeFile(stdoutPath, stdout)
  await writeFile(stderrPath, stderr)

  return {
    run_index: runIndex,
    status: runStatus(outcome.code, spawnError, timedOut),
    command: command.public,
    timing: {
      started_utc: startedUtc,
      ended_utc: endedUtc,
      duration_ms: durationMs,
      timeout_ms: command.timeout_ms,
      timed_out: timedOut,
    },
    memory: {
      before,
      after: memorySnapshot(),
      peak_rss_kb: peakRssKb,
      rss_samples: rssSamples,
    },
    output: {
      stdout_log: relative(plan.root, stdoutPath),
      stderr_log: relative(plan.root, stderrPath),
      command_log: relative(plan.root, metaPath),
      stdout_bytes: stdout.byteLength,
      stderr_bytes: stderr.byteLength,
      stdout_preview: preview(stdout),
      stderr_preview: preview(stderr),
    },
    exit: {
      code: outcome.code,
      signal: outcome.signal,
      error: spawnError ? spawnError.message : null,
    },
  }
}

function buildPlan({ configPath, config, entries, root, bundleDir, timestamp, dryRun, hashModels, machine, modelManifest }) {
  const defaults = {
    repetitions: parsePositiveInt(config.defaults?.repetitions ?? 1, 'defaults.repetitions'),
    timeout_ms: parsePositiveInt(config.defaults?.timeout_ms ?? DEFAULT_TIMEOUT_MS, 'defaults.timeout_ms'),
  }

  return {
    schema: PLAN_SCHEMA,
    generated_utc: new Date().toISOString(),
    config_path: configPath,
    root,
    bundle_dir: bundleDir,
    timestamp,
    dry_run: dryRun,
    hash_models: hashModels,
    harness: {
      schema: HARNESS_SCHEMA,
      name: 'Camelid v0.1 benchmark harness',
      script: relative(root, new URL(import.meta.url).pathname),
      supported_engines: [...SUPPORTED_ENGINES],
      required_outputs: [
        'machine.json',
        'model_manifest.json',
        'commands.md',
        'raw_logs/',
        'results.json',
        'results.csv',
        'summary.md',
      ],
    },
    metadata: config.metadata,
    defaults,
    machine,
    model_manifest: modelManifest,
    entries: entries.map((entry) => normalizeEntry(entry, defaults, root)),
  }
}

function normalizeEntry(entry, defaults, root) {
  const command = normalizeCommand(entry, defaults.timeout_ms, root)
  return {
    id: entry.id,
    label: entry.label,
    engine: entry.engine,
    model_id: entry.model_id,
    prompt: entry.prompt,
    prompt_sha256: sha256Text(String(entry.prompt)),
    repetitions: parsePositiveInt(entry.repetitions ?? defaults.repetitions, `entry ${entry.id} repetitions`),
    tags: Array.isArray(entry.tags) ? entry.tags : [],
    expected: entry.expected ?? null,
    command: command.public,
  }
}

function normalizeCommand(entry, defaultTimeoutMs = DEFAULT_TIMEOUT_MS, root = process.cwd()) {
  const command = entry.command
  let file
  let args = []
  let shell = false
  let display
  let cwd = resolve(root, entry.cwd || '.')
  let envOverrides = entry.env || {}

  if (typeof command === 'string') {
    file = command
    shell = true
    display = command
  } else if (Array.isArray(command)) {
    if (command.length === 0) throw new Error(`entry ${entry.id} command array cannot be empty`)
    file = command[0]
    args = command.slice(1).map(String)
    display = quoteCommand(command)
  } else if (command && typeof command === 'object') {
    if (Array.isArray(command.argv)) {
      if (command.argv.length === 0) throw new Error(`entry ${entry.id} command.argv cannot be empty`)
      file = command.argv[0]
      args = command.argv.slice(1).map(String)
      display = quoteCommand(command.argv)
    } else if (typeof command.cmd === 'string') {
      file = command.cmd
      shell = command.shell !== false
      display = command.cmd
    } else {
      throw new Error(`entry ${entry.id} command must be a string, array, {argv}, or {cmd}`)
    }
    cwd = resolve(root, command.cwd || entry.cwd || '.')
    envOverrides = { ...envOverrides, ...(command.env || {}) }
    if (typeof command.shell === 'boolean') shell = command.shell
  } else {
    throw new Error(`entry ${entry.id} command is invalid`)
  }

  const timeoutMs = parsePositiveInt(entry.timeout_ms ?? command?.timeout_ms ?? defaultTimeoutMs, `entry ${entry.id} timeout_ms`)
  const env = { ...process.env, ...stringifyEnv(envOverrides) }
  const envPublic = Object.fromEntries(Object.entries(stringifyEnv(envOverrides)).map(([key, value]) => [key, redactEnvValue(key, value)]))

  return {
    file,
    args,
    shell,
    cwd,
    env,
    timeout_ms: timeoutMs,
    public: {
      display,
      file,
      args,
      shell,
      cwd,
      env: envPublic,
      timeout_ms: timeoutMs,
    },
  }
}

async function buildModelManifest(config, root, hashModels) {
  const models = []
  for (const model of config.models) {
    const localPath = localArtifactPath(model.artifact_uri, root)
    const file = localPath ? await fileInfo(localPath, hashModels || model.hash === true) : null
    models.push({
      ...model,
      artifact_uri: model.artifact_uri,
      artifact_file: file,
    })
  }
  return {
    schema: 'camelid.v0_1.model_manifest.v1',
    generated_utc: new Date().toISOString(),
    hash_models: hashModels,
    models,
  }
}

async function fileInfo(path, hashFile) {
  try {
    const info = await stat(path)
    return {
      path,
      exists: true,
      size_bytes: info.size,
      mtime_utc: info.mtime.toISOString(),
      sha256: hashFile ? await sha256File(path) : null,
    }
  } catch (err) {
    return {
      path,
      exists: false,
      error: err.message,
      size_bytes: null,
      mtime_utc: null,
      sha256: null,
    }
  }
}

function localArtifactPath(uri, root) {
  if (uri.startsWith('file://')) return new URL(uri).pathname
  if (/^[a-z][a-z0-9+.-]*:/i.test(uri)) return null
  return isAbsolute(uri) ? uri : resolve(root, uri)
}

async function sha256File(path) {
  const hash = createHash('sha256')
  await new Promise((resolveHash, rejectHash) => {
    const stream = createReadStream(path)
    stream.on('data', (chunk) => hash.update(chunk))
    stream.once('error', rejectHash)
    stream.once('end', resolveHash)
  })
  return hash.digest('hex')
}

function collectMachine(root) {
  return {
    schema: 'camelid.v0_1.machine.v1',
    generated_utc: new Date().toISOString(),
    hostname: os.hostname(),
    platform: os.platform(),
    release: os.release(),
    arch: os.arch(),
    type: os.type(),
    cpus: os.cpus().map((cpu) => ({ model: cpu.model, speed_mhz: cpu.speed })),
    cpu_count: os.cpus().length,
    memory: memorySnapshot(),
    node: {
      version: process.version,
      exec_path: process.execPath,
    },
    git: gitInfo(root),
  }
}

function gitInfo(root) {
  return {
    branch: runQuiet('git', ['branch', '--show-current'], root),
    commit: runQuiet('git', ['rev-parse', 'HEAD'], root),
    status_short: runQuiet('git', ['status', '--short'], root),
  }
}

function memorySnapshot() {
  return {
    total_bytes: os.totalmem(),
    free_bytes: os.freemem(),
    process_rss_bytes: process.memoryUsage().rss,
  }
}

function readProcessRssKb(pid) {
  if (!pid) return null
  const result = spawnSync('ps', ['-o', 'rss=', '-p', String(pid)], { encoding: 'utf8' })
  if (result.status !== 0) return null
  const parsed = Number.parseInt(result.stdout.trim(), 10)
  return Number.isFinite(parsed) ? parsed : null
}

function runQuiet(cmd, cmdArgs, cwd) {
  const result = spawnSync(cmd, cmdArgs, { cwd, encoding: 'utf8' })
  return result.status === 0 ? result.stdout.trim() : null
}

function validateConfig(config) {
  if (!config || typeof config !== 'object' || Array.isArray(config)) {
    throw new Error('config must be a JSON object')
  }
  if (config.schema && config.schema !== HARNESS_SCHEMA) {
    throw new Error(`config.schema must be ${HARNESS_SCHEMA}`)
  }
  requireObject(config.metadata, 'metadata')
  for (const field of REQUIRED_METADATA_FIELDS) requireNonEmptyString(config.metadata[field], `metadata.${field}`)

  if (!Array.isArray(config.models) || config.models.length === 0) {
    throw new Error('models must be a non-empty array')
  }
  const modelIds = new Set()
  for (const model of config.models) {
    requireObject(model, 'model')
    for (const field of REQUIRED_MODEL_FIELDS) requireNonEmptyString(String(model[field] ?? ''), `models[].${field}`)
    if (modelIds.has(model.id)) throw new Error(`duplicate model id ${model.id}`)
    modelIds.add(model.id)
    if (model.artifact_sha256) {
      if (!/^[a-f0-9]{64}$/i.test(model.artifact_sha256)) {
        throw new Error(`model ${model.id} artifact_sha256 must be 64 hex characters`)
      }
    } else {
      requireNonEmptyString(model.artifact_sha256_unavailable_reason, `model ${model.id} artifact_sha256_unavailable_reason`)
    }
  }

  if (!Array.isArray(config.entries) || config.entries.length === 0) {
    throw new Error('entries must be a non-empty array')
  }
  const entryIds = new Set()
  for (const entry of config.entries) {
    requireObject(entry, 'entry')
    for (const field of REQUIRED_ENTRY_FIELDS) {
      if (field === 'command') {
        if (entry.command === undefined) throw new Error('entries[].command is required')
      } else {
        requireNonEmptyString(String(entry[field] ?? ''), `entries[].${field}`)
      }
    }
    if (entryIds.has(entry.id)) throw new Error(`duplicate entry id ${entry.id}`)
    entryIds.add(entry.id)
    if (!/^[a-zA-Z0-9_.-]+$/.test(entry.id)) {
      throw new Error(`entry id ${entry.id} must contain only letters, numbers, underscores, dots, or hyphens`)
    }
    if (!SUPPORTED_ENGINES.has(entry.engine)) {
      throw new Error(`entry ${entry.id} engine must be one of ${[...SUPPORTED_ENGINES].join(', ')}`)
    }
    if (!modelIds.has(entry.model_id)) throw new Error(`entry ${entry.id} references unknown model_id ${entry.model_id}`)
    parsePositiveInt(entry.repetitions ?? config.defaults?.repetitions ?? 1, `entry ${entry.id} repetitions`)
    parsePositiveInt(entry.timeout_ms ?? config.defaults?.timeout_ms ?? DEFAULT_TIMEOUT_MS, `entry ${entry.id} timeout_ms`)
  }
}

function filterEntries(entries, selectedIds) {
  if (selectedIds.length === 0) return entries
  const selected = new Set(selectedIds)
  const filtered = entries.filter((entry) => selected.has(entry.id))
  const missing = [...selected].filter((id) => !filtered.some((entry) => entry.id === id))
  if (missing.length > 0) throw new Error(`unknown --entry value(s): ${missing.join(', ')}`)
  return filtered
}

function parseArgs(argv) {
  const parsed = new Map()
  const multi = new Map()
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]
    if (!arg.startsWith('--')) continue
    const [key, inline] = arg.slice(2).split('=', 2)
    const value = inline ?? (argv[i + 1]?.startsWith('--') ? 'true' : argv[++i] ?? 'true')
    if (parsed.has(key)) {
      if (!multi.has(key)) multi.set(key, [parsed.get(key)])
      multi.get(key).push(value)
    }
    parsed.set(key, value)
  }
  parsed.multi = multi
  return parsed
}

function valuesFor(parsed, key) {
  return parsed.multi?.get(key) || (parsed.has(key) ? [parsed.get(key)] : [])
}

async function readConfig(path) {
  try {
    return JSON.parse(await readFile(path, 'utf8'))
  } catch (err) {
    throw new Error(`failed to read config ${path}: ${err.message}`)
  }
}

async function writeJson(path, value) {
  await mkdir(dirname(path), { recursive: true })
  await writeFile(path, `${JSON.stringify(value, null, 2)}\n`)
}

function renderCommandsMarkdown(plan) {
  const lines = [
    '# Camelid v0.1 Benchmark Commands',
    '',
    `- Generated UTC: ${plan.generated_utc}`,
    `- Config: \`${plan.config_path}\``,
    `- Bundle: \`${plan.bundle_dir}\``,
    `- Dry run: ${plan.dry_run}`,
    '',
  ]
  for (const entry of plan.entries) {
    lines.push(`## ${entry.id}`)
    lines.push('')
    lines.push(`- Engine: ${entry.engine}`)
    lines.push(`- Model: ${entry.model_id}`)
    lines.push(`- Repetitions: ${entry.repetitions}`)
    lines.push(`- Timeout ms: ${entry.command.timeout_ms}`)
    lines.push(`- CWD: \`${entry.command.cwd}\``)
    if (Object.keys(entry.command.env).length > 0) {
      lines.push(`- Env overrides: \`${JSON.stringify(entry.command.env)}\``)
    }
    lines.push('')
    lines.push('```sh')
    lines.push(entry.command.display)
    lines.push('```')
    lines.push('')
  }
  return `${lines.join('\n')}\n`
}

function renderResultsCsv(results) {
  const rows = [[
    'entry_id',
    'engine',
    'model_id',
    'run_index',
    'status',
    'exit_code',
    'signal',
    'duration_ms',
    'timeout_ms',
    'timed_out',
    'peak_rss_kb',
    'stdout_bytes',
    'stderr_bytes',
    'stdout_log',
    'stderr_log',
  ]]
  for (const result of results) {
    for (const run of result.runs) {
      rows.push([
        result.entry_id,
        result.engine,
        result.model_id,
        run.run_index,
        run.status,
        run.exit.code ?? '',
        run.exit.signal ?? '',
        run.timing.duration_ms,
        run.timing.timeout_ms,
        run.timing.timed_out,
        run.memory.peak_rss_kb ?? '',
        run.output.stdout_bytes,
        run.output.stderr_bytes,
        run.output.stdout_log ?? '',
        run.output.stderr_log ?? '',
      ])
    }
  }
  return `${rows.map((row) => row.map(csvCell).join(',')).join('\n')}\n`
}

function renderSummaryMarkdown(bundle) {
  const lines = [
    '# Camelid v0.1 Benchmark Summary',
    '',
    `- Generated UTC: ${bundle.generated_utc}`,
    `- Release version: ${bundle.metadata.release_version}`,
    `- Benchmark: ${bundle.metadata.benchmark_name}`,
    `- Purpose: ${bundle.metadata.purpose}`,
    `- Dry run: ${bundle.dry_run}`,
    '',
    '| Entry | Engine | Model | Status | Runs | Avg ms | Peak RSS KB |',
    '|---|---|---|---:|---:|---:|---:|',
  ]

  for (const result of bundle.results) {
    lines.push([
      result.entry_id,
      result.engine,
      result.model_id,
      result.status,
      result.repetitions,
      result.summary.duration_ms_avg ?? '',
      result.summary.peak_rss_kb_max ?? '',
    ].map(markdownCell).join('|').replace(/^/, '|').replace(/$/, '|'))
  }

  lines.push('')
  lines.push('## Output Files')
  lines.push('')
  lines.push('- `machine.json` captures host, Node, memory, CPU, and Git context.')
  lines.push('- `model_manifest.json` records model metadata and optional local file stats/hash evidence.')
  lines.push('- `commands.md` records the exact configured command for every run.')
  lines.push('- `raw_logs/` stores per-run stdout, stderr, and command metadata.')
  lines.push('- `results.json` and `results.csv` contain machine-readable timing, exit, output, and memory fields.')
  lines.push('')
  return `${lines.join('\n')}\n`
}

function summarizeResults(results) {
  return {
    entries_total: results.length,
    entries_ok: results.filter((result) => result.status === 'ok').length,
    entries_failed: results.filter((result) => result.status === 'failed').length,
    entries_skipped: results.filter((result) => result.status === 'skipped').length,
    runs_total: results.reduce((sum, result) => sum + result.runs.length, 0),
  }
}

function summarizeRuns(runs) {
  const completed = runs.filter((run) => run.status === 'ok')
  const failed = runs.filter((run) => run.status === 'failed')
  const skipped = runs.filter((run) => run.status === 'skipped')
  const durations = completed.map((run) => run.timing.duration_ms).filter(Number.isFinite)
  const rss = runs.map((run) => run.memory.peak_rss_kb).filter(Number.isFinite)
  return {
    completed: completed.length,
    failed: failed.length,
    skipped: skipped.length,
    duration_ms_min: durations.length ? round(Math.min(...durations)) : null,
    duration_ms_avg: durations.length ? round(durations.reduce((sum, value) => sum + value, 0) / durations.length) : null,
    duration_ms_max: durations.length ? round(Math.max(...durations)) : null,
    peak_rss_kb_max: rss.length ? Math.max(...rss) : null,
  }
}

function entryStatus(runs) {
  if (runs.every((run) => run.status === 'skipped')) return 'skipped'
  return runs.every((run) => run.status === 'ok') ? 'ok' : 'failed'
}

function runStatus(exitCode, spawnError, timedOut) {
  if (timedOut || spawnError || exitCode !== 0) return 'failed'
  return 'ok'
}

function publicEntry(entry) {
  return {
    id: entry.id,
    label: entry.label,
    engine: entry.engine,
    model_id: entry.model_id,
    prompt_sha256: entry.prompt_sha256,
    repetitions: entry.repetitions,
    tags: entry.tags,
    expected: entry.expected,
    command: entry.command,
  }
}

function usage() {
  return `${usageSummary()}

Options:
  --config <path>       Required JSON config path.
  --out-root <path>     Bundle root. Defaults to ${DEFAULT_OUT_ROOT}.
  --timestamp <stamp>   Bundle directory name. Defaults to current UTC timestamp.
  --entry <id>          Run only one entry. May be repeated.
  --dry-run             Create the bundle but skip command execution.
  --print-plan          Print the normalized plan JSON and create no files.
  --hash-models         Compute sha256 for local file artifacts in model_manifest.json.
  --root <path>         Repository root for relative paths. Defaults to cwd.
  --help                Show this help.

Output bundle:
  qa/evidence-bundles/v0.1/<timestamp>/
    machine.json, model_manifest.json, commands.md, raw_logs/, results.json,
    results.csv, summary.md

Config shape:
{
  "schema": "${HARNESS_SCHEMA}",
  "metadata": {
    "release_version": "v0.1",
    "benchmark_name": "same-host release candidate",
    "operator": "Benchmark Harness Agent",
    "purpose": "Evidence-first v0.1 benchmark capture"
  },
  "models": [{
    "id": "tinyllama-q8",
    "label": "TinyLlama Q8_0",
    "family": "TinyLlama",
    "parameters": "1.1B",
    "quantization": "Q8_0",
    "artifact_uri": "models/tinyllama.gguf",
    "artifact_sha256": "64 hex characters"
  }],
  "entries": [{
    "id": "camelid-tinyllama",
    "label": "Camelid TinyLlama smoke",
    "engine": "camelid",
    "model_id": "tinyllama-q8",
    "prompt": "hello",
    "command": ["curl", "-sS", "http://127.0.0.1:8181/v1/health"]
  }]
}

Supported engines: ${[...SUPPORTED_ENGINES].join(', ')}`
}

function usageSummary() {
  return 'Usage: node tools/bench/v0.1-benchmark-harness.mjs --config qa/bench/v0.1-bench.json [--dry-run|--print-plan]'
}

function requireObject(value, name) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${name} must be an object`)
  }
}

function requireNonEmptyString(value, name) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`${name} is required`)
  }
}

function parsePositiveInt(value, name) {
  const parsed = Number.parseInt(String(value), 10)
  if (!Number.isInteger(parsed) || parsed < 1) throw new Error(`${name} must be a positive integer, got ${value}`)
  return parsed
}

function timestampUtc() {
  return new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z')
}

function sanitizeSlug(value) {
  return String(value).replace(/[^a-zA-Z0-9_.-]+/g, '_')
}

function quoteCommand(argv) {
  return argv.map((part) => {
    const text = String(part)
    if (/^[a-zA-Z0-9_./:@%+=,-]+$/.test(text)) return text
    return `'${text.replaceAll("'", "'\\''")}'`
  }).join(' ')
}

function stringifyEnv(env) {
  return Object.fromEntries(Object.entries(env || {}).map(([key, value]) => [key, String(value)]))
}

function redactEnvValue(key, value) {
  return /TOKEN|SECRET|PASSWORD|KEY/i.test(key) ? '<redacted>' : value
}

function sha256Text(value) {
  return createHash('sha256').update(value).digest('hex')
}

function preview(buffer) {
  const text = buffer.toString('utf8')
  return text.length > 2048 ? `${text.slice(0, 2048)}\n...[truncated ${text.length - 2048} chars]` : text
}

function csvCell(value) {
  const text = String(value ?? '')
  if (!/[",\n]/.test(text)) return text
  return `"${text.replaceAll('"', '""')}"`
}

function markdownCell(value) {
  return ` ${String(value ?? '').replaceAll('|', '\\|')} `
}

function round(value) {
  return Math.round(value * 100) / 100
}
