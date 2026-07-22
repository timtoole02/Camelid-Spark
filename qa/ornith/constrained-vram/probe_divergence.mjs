// Item 2 — attribute the prompt-3 divergence at generated index 50: ask the oracle
// for top-10 logprobs at the divergent step (prefix = prompt ids + 50 shared
// generated ids), plus detokenizations of both tails for the receipt narrative.
import fs from 'fs';
import path from 'path';
import { spawn, execSync } from 'child_process';

const HERE = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'));
const SERVER = (process.env.CAMELID_LLAMACPP_BIN || 'llama.cpp/build/bin') + '/llama-server.exe';
const MODEL = (process.env.CAMELID_MODELS_DIR || 'models') + '/ornith-1.0-9b-Q8_0.gguf';
const PORT = 8114;
const receipt = JSON.parse(fs.readFileSync(path.join(HERE, 'RECEIPT_ITEM2_qwen35_parity.json'), 'utf8'));
const g = receipt.generations.find((x) => x.id === 3);
const fixtures = JSON.parse(fs.readFileSync(path.join(HERE, 'FIXTURES_five_prompt_parity.json'), 'utf8'));
const promptIds = fixtures.prompts.find((p) => p.id === 3).prompt_token_ids;
const prefix = [...promptIds, ...g.oracle.slice(0, 50)];

const child = spawn(SERVER, ['-m', MODEL, '--port', String(PORT), '-c', '2048', '-ngl', '0', '-ctk', 'f32', '-ctv', 'f32', '--no-repack'], { detached: true, stdio: 'ignore' });
child.unref();
const deadline = Date.now() + 600_000;
while (Date.now() < deadline) {
  try { if ((await fetch(`http://127.0.0.1:${PORT}/health`)).ok) break; } catch {}
  await new Promise((r) => setTimeout(r, 2000));
}

try {
  const res = await fetch(`http://127.0.0.1:${PORT}/completion`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ prompt: prefix, n_predict: 1, temperature: 0, top_k: 1, seed: 0, cache_prompt: false, n_probs: 10, return_tokens: true }),
  });
  const body = await res.json();
  const detok = async (ids) => (await (await fetch(`http://127.0.0.1:${PORT}/detokenize`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ tokens: ids }) })).json()).content;
  const out = {
    divergent_index: 50,
    oracle_next: body.tokens,
    top_probs: (body.completion_probabilities?.[0]?.top_logprobs ?? body.completion_probabilities?.[0]?.probs ?? body.completion_probabilities ?? null),
    camelid_tail_text: await detok(g.camelid.slice(50)),
    oracle_tail_text: await detok(g.oracle.slice(50)),
    shared_prefix_tail_text: await detok(g.oracle.slice(38, 50)),
  };
  fs.writeFileSync(path.join(HERE, 'item2_divergence_probe.json'), JSON.stringify(out, null, 2));
  console.log(JSON.stringify(out, null, 2).slice(0, 3000));
} finally {
  try { execSync(`taskkill /F /PID ${child.pid}`, { stdio: 'ignore' }); } catch {}
}
