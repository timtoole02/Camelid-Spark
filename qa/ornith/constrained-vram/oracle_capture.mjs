// Item 2 — capture REF_QWEN35 greedy generations from prompt TOKEN IDS over the
// five-prompt corpus. Works for both oracle configs:
//   node oracle_capture.mjs cpu   -> llama-server -ngl 0 -ctk f32 -ctv f32 --no-repack, Q8_0
//   node oracle_capture.mjs cuda  -> llama-server -ngl 99, Q4_K_M
// Starts the server itself (detached), waits /health, runs 5 greedy completions
// (n_predict from CAMELID_PARITY_NPREDICT, default 64), kills the server, writes
// oracle_gen_<mode>.json. Run camelid's side AFTER this exits (RAM/VRAM ceiling).
import fs from 'fs';
import path from 'path';
import { spawn, execSync } from 'child_process';

const HERE = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'));
const MODE = process.argv[2] === 'cuda' ? 'cuda' : 'cpu';
const N_PREDICT = parseInt(process.env.CAMELID_PARITY_NPREDICT || '64', 10);
const SERVER = (process.env.CAMELID_LLAMACPP_BIN || 'llama.cpp/build/bin') + '/llama-server.exe';
const MODEL = process.env.CAMELID_ORACLE_MODEL || (MODE === 'cuda'
  ? (process.env.CAMELID_MODELS_DIR || 'models') + '/ornith-1.0-9b-Q4_K_M.gguf'
  : (process.env.CAMELID_MODELS_DIR || 'models') + '/ornith-1.0-9b-Q8_0.gguf');
const PORT = 8114;
const fixtures = JSON.parse(fs.readFileSync(path.join(HERE, 'FIXTURES_five_prompt_parity.json'), 'utf8'));

const args = ['-m', MODEL, '--port', String(PORT), '-c', '2048'];
if (MODE === 'cuda') args.push('-ngl', '99');
else args.push('-ngl', '0', '-ctk', 'f32', '-ctv', 'f32', '--no-repack');

console.log(`starting oracle server (${MODE}): llama-server ${args.join(' ')}`);
const child = spawn(SERVER, args, { detached: true, stdio: ['ignore', 'ignore', 'ignore'] });
child.unref();

const deadline = Date.now() + 600_000;
let healthy = false;
while (Date.now() < deadline) {
  try {
    const r = await fetch(`http://127.0.0.1:${PORT}/health`);
    if (r.ok) { healthy = true; break; }
  } catch {}
  await new Promise((res) => setTimeout(res, 2000));
}
if (!healthy) { try { process.kill(child.pid); } catch {} throw new Error('oracle server never became healthy'); }
console.log('server healthy; capturing...');

const out = [];
try {
  for (const p of fixtures.prompts) {
    const res = await fetch(`http://127.0.0.1:${PORT}/completion`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        prompt: p.prompt_token_ids,
        n_predict: N_PREDICT,
        temperature: 0,
        top_k: 1,
        seed: 0,
        cache_prompt: false,
        return_tokens: true,
      }),
    });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    const body = await res.json();
    out.push({ id: p.id, prompt_token_ids: p.prompt_token_ids, generated_token_ids: body.tokens, text: body.content });
    console.log(`prompt ${p.id}: ${body.tokens?.length} tokens`);
  }
} finally {
  try { execSync(`taskkill /F /PID ${child.pid}`, { stdio: 'ignore' }); } catch {}
}

fs.writeFileSync(path.join(HERE, `oracle_gen_${process.env.CAMELID_ORACLE_SUFFIX || MODE}.json`), JSON.stringify({
  schema: 'camelid.fixture-corpus/v1',
  oracle: `llama.cpp acd79d6 llama-server ${MODE === 'cuda' ? '-ngl 99 (CUDA)' : '-ngl 0 -ctk f32 -ctv f32 --no-repack (CPU)'}`,
  model: MODEL,
  n_predict: N_PREDICT,
  sampling: 'greedy (temperature 0, top_k 1, seed 0, cache_prompt false)',
  date: new Date().toISOString(),
  generations: out,
}, null, 2));
console.log(`wrote oracle_gen_${process.env.CAMELID_ORACLE_SUFFIX || MODE}.json`);
