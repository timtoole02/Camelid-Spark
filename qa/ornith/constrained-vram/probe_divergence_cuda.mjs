// Item 2 CUDA — attribute each divergence: for every non-matching prompt in
// RECEIPT_ITEM2_qwen35_parity_cuda.json, ask the CUDA oracle for top-10 logprobs at
// the divergent step (prefix = prompt ids + shared generated prefix) and detokenize
// both tails. One server session for all probes.
import fs from 'fs';
import path from 'path';
import { spawn, execSync } from 'child_process';

const HERE = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'));
const SERVER = (process.env.CAMELID_LLAMACPP_BIN || 'llama.cpp/build/bin') + '/llama-server.exe';
const MODEL = (process.env.CAMELID_MODELS_DIR || 'models') + '/ornith-1.0-9b-Q4_K_M.gguf';
const PORT = 8114;
const receipt = JSON.parse(fs.readFileSync(path.join(HERE, 'RECEIPT_ITEM2_qwen35_parity_cuda.json'), 'utf8'));
const fixtures = JSON.parse(fs.readFileSync(path.join(HERE, 'FIXTURES_five_prompt_parity.json'), 'utf8'));

const child = spawn(SERVER, ['-m', MODEL, '--port', String(PORT), '-c', '2048', '-ngl', '99'], { detached: true, stdio: 'ignore' });
child.unref();
const deadline = Date.now() + 600_000;
while (Date.now() < deadline) {
  try { if ((await fetch(`http://127.0.0.1:${PORT}/health`)).ok) break; } catch {}
  await new Promise((r) => setTimeout(r, 2000));
}

const probes = [];
try {
  const detok = async (ids) => (await (await fetch(`http://127.0.0.1:${PORT}/detokenize`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ tokens: ids }) })).json()).content;
  for (const g of receipt.generations.filter((x) => !x.match)) {
    const promptIds = fixtures.prompts.find((p) => p.id === g.id).prompt_token_ids;
    const d = g.first_divergent_generated_token_index;
    const prefix = [...promptIds, ...g.oracle.slice(0, d)];
    const res = await fetch(`http://127.0.0.1:${PORT}/completion`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ prompt: prefix, n_predict: 1, temperature: 0, top_k: 1, seed: 0, cache_prompt: false, n_probs: 10, return_tokens: true }),
    });
    const body = await res.json();
    const tp = body.completion_probabilities?.[0]?.top_logprobs ?? [];
    const camelidPick = tp.find((e) => e.id === g.camelid[d]);
    probes.push({
      prompt: g.id,
      divergent_index: d,
      oracle_pick: { id: g.oracle[d], entry: tp.find((e) => e.id === g.oracle[d]) ?? null },
      camelid_pick: { id: g.camelid[d], entry: camelidPick ?? null, rank: camelidPick ? tp.indexOf(camelidPick) + 1 : '>10' },
      top2_gap_nats: tp.length >= 2 ? tp[0].logprob - tp[1].logprob : null,
      top4: tp.slice(0, 4).map((e) => ({ id: e.id, token: e.token, logprob: e.logprob })),
      camelid_tail_text: await detok(g.camelid.slice(d, d + 16)),
      oracle_tail_text: await detok(g.oracle.slice(d, d + 16)),
    });
    console.log(`prompt ${g.id} @${d}: gap=${probes.at(-1).top2_gap_nats?.toFixed(5)} camelid_rank=${probes.at(-1).camelid_pick.rank}`);
  }
} finally {
  try { execSync(`taskkill /F /PID ${child.pid}`, { stdio: 'ignore' }); } catch {}
}
fs.writeFileSync(path.join(HERE, 'item2_cuda_divergence_probes.json'), JSON.stringify(probes, null, 2));
console.log('wrote item2_cuda_divergence_probes.json');
