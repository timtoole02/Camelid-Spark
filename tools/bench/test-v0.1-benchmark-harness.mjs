#!/usr/bin/env node
import assert from 'node:assert/strict'
import { access, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { spawnSync } from 'node:child_process'

const repo = resolve(new URL('../..', import.meta.url).pathname)
const script = 'tools/bench/v0.1-benchmark-harness.mjs'
const tmp = await mkdtemp(join(tmpdir(), 'camelid-v01-bench-'))

try {
  const configPath = join(tmp, 'bench-config.json')
  await writeFile(configPath, `${JSON.stringify(validConfig(), null, 2)}\n`)

  const planRun = runHarness(['--config', configPath, '--out-root', join(tmp, 'plan-out'), '--timestamp', 'plan-stamp', '--print-plan'])
  assert.equal(planRun.status, 0, planRun.stderr)
  const plan = JSON.parse(planRun.stdout)
  assert.equal(plan.schema, 'camelid.v0_1.benchmark_plan.v1')
  assert.equal(plan.dry_run, false)
  assert.equal(plan.entries.length, 4)
  assert.deepEqual(plan.harness.supported_engines, ['camelid', 'llama.cpp', 'ollama', 'mlx'])
  assert.match(plan.bundle_dir, /plan-out\/plan-stamp$/)
  await assert.rejects(access(join(tmp, 'plan-out', 'plan-stamp')))

  const dryRun = runHarness(['--config', configPath, '--out-root', join(tmp, 'dry-out'), '--timestamp', 'dry-stamp', '--dry-run'])
  assert.equal(dryRun.status, 0, dryRun.stderr)
  assert.match(dryRun.stdout, /bundle_dir=.*dry-stamp/)
  await assertBundleFiles(join(tmp, 'dry-out', 'dry-stamp'))
  const dryResults = JSON.parse(await readFile(join(tmp, 'dry-out', 'dry-stamp', 'results.json'), 'utf8'))
  assert.equal(dryResults.dry_run, true)
  assert.equal(dryResults.summary.entries_skipped, 4)
  assert.equal(dryResults.results[0].runs[0].status, 'skipped')
  const dryCommands = await readFile(join(tmp, 'dry-out', 'dry-stamp', 'commands.md'), 'utf8')
  assert.match(dryCommands, /<redacted>/)
  assert.doesNotMatch(dryCommands, /secret-token/)

  const realRun = runHarness(['--config', configPath, '--out-root', join(tmp, 'real-out'), '--timestamp', 'real-stamp'])
  assert.equal(realRun.status, 0, realRun.stderr)
  await assertBundleFiles(join(tmp, 'real-out', 'real-stamp'))
  const realResults = JSON.parse(await readFile(join(tmp, 'real-out', 'real-stamp', 'results.json'), 'utf8'))
  assert.equal(realResults.dry_run, false)
  assert.equal(realResults.summary.entries_ok, 4)
  assert.equal(realResults.summary.runs_total, 4)
  for (const result of realResults.results) {
    assert.equal(result.status, 'ok')
    assert.equal(result.runs[0].exit.code, 0)
    assert.equal(typeof result.runs[0].timing.duration_ms, 'number')
    assert.equal(typeof result.runs[0].memory.before.total_bytes, 'number')
    assert.equal(typeof result.runs[0].output.stdout_log, 'string')
    await access(resolve(repo, result.runs[0].output.stdout_log))
  }
  const csv = await readFile(join(tmp, 'real-out', 'real-stamp', 'results.csv'), 'utf8')
  assert.match(csv, /entry_id,engine,model_id,run_index,status/)
  assert.match(csv, /camelid-smoke,camelid,test-model,1,ok/)
  const summary = await readFile(join(tmp, 'real-out', 'real-stamp', 'summary.md'), 'utf8')
  assert.match(summary, /Camelid v0\.1 Benchmark Summary/)
  assert.match(summary, /ollama-smoke/)

  const badConfigPath = join(tmp, 'bad-config.json')
  const bad = validConfig()
  delete bad.metadata.operator
  await writeFile(badConfigPath, `${JSON.stringify(bad, null, 2)}\n`)
  const badRun = runHarness(['--config', badConfigPath, '--out-root', join(tmp, 'bad-out'), '--timestamp', 'bad-stamp'])
  assert.notEqual(badRun.status, 0)
  assert.match(badRun.stderr, /metadata\.operator is required/)

  const helpRun = runHarness(['--help'])
  assert.equal(helpRun.status, 0, helpRun.stderr)
  assert.match(helpRun.stdout, /Supported engines: camelid, llama\.cpp, ollama, mlx/)
  assert.match(helpRun.stdout, /machine\.json/)
} finally {
  await rm(tmp, { recursive: true, force: true })
}

function runHarness(args) {
  return spawnSync(process.execPath, [script, ...args], {
    cwd: repo,
    encoding: 'utf8',
  })
}

async function assertBundleFiles(dir) {
  for (const file of [
    'machine.json',
    'model_manifest.json',
    'commands.md',
    'raw_logs',
    'results.json',
    'results.csv',
    'summary.md',
  ]) {
    await access(join(dir, file))
  }
}

function validConfig() {
  const commandFor = (engine) => ({
    argv: [
      process.execPath,
      '-e',
      `setTimeout(() => { console.log(${JSON.stringify(`${engine} ok`)}); }, 350)`,
    ],
    env: {
      BENCH_TEST_MODE: engine,
      API_TOKEN: 'secret-token',
    },
  })

  return {
    schema: 'camelid.v0_1.benchmark_harness.config.v1',
    metadata: {
      release_version: 'v0.1',
      benchmark_name: 'synthetic harness self-test',
      operator: 'Benchmark Harness Agent',
      purpose: 'Verify v0.1 benchmark bundle generation without external runtimes',
    },
    defaults: {
      repetitions: 1,
      timeout_ms: 5000,
    },
    models: [{
      id: 'test-model',
      label: 'Synthetic model artifact',
      family: 'synthetic',
      parameters: '0B',
      quantization: 'none',
      artifact_uri: 'synthetic://model',
      artifact_sha256_unavailable_reason: 'self-test uses no model file',
    }],
    entries: [
      {
        id: 'camelid-smoke',
        label: 'Camelid synthetic smoke',
        engine: 'camelid',
        model_id: 'test-model',
        prompt: 'Return the synthetic Camelid marker.',
        command: commandFor('camelid'),
      },
      {
        id: 'llamacpp-smoke',
        label: 'llama.cpp synthetic smoke',
        engine: 'llama.cpp',
        model_id: 'test-model',
        prompt: 'Return the synthetic llama.cpp marker.',
        command: commandFor('llama.cpp'),
      },
      {
        id: 'ollama-smoke',
        label: 'Ollama synthetic smoke',
        engine: 'ollama',
        model_id: 'test-model',
        prompt: 'Return the synthetic Ollama marker.',
        command: commandFor('ollama'),
      },
      {
        id: 'mlx-smoke',
        label: 'MLX synthetic smoke',
        engine: 'mlx',
        model_id: 'test-model',
        prompt: 'Return the synthetic MLX marker.',
        command: commandFor('mlx'),
      },
    ],
  }
}
