// Control B — the oracle's OWN cross-backend variance on identical weights:
// oracle_gen_cpu_q4km.json (llama-server -ngl 0, Q4_K_M) vs oracle_gen_cuda.json
// (llama-server -ngl 99, same Q4_K_M). Greedy, same token-id prompts, n=64.
import fs from 'fs';
import path from 'path';
const HERE = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'));
const cpu = JSON.parse(fs.readFileSync(path.join(HERE, 'oracle_gen_cpu_q4km.json'), 'utf8'));
const cuda = JSON.parse(fs.readFileSync(path.join(HERE, 'oracle_gen_cuda.json'), 'utf8'));
const rows = cpu.generations.map((g, i) => {
  const a = g.generated_token_ids, b = cuda.generations[i].generated_token_ids;
  let d = -1;
  for (let j = 0; j < Math.max(a.length, b.length); j++) if (a[j] !== b[j]) { d = j; break; }
  return { id: g.id, first_divergence: d };
});
fs.writeFileSync(path.join(HERE, 'oracle_backend_variance.json'), JSON.stringify({
  comparison: 'llama.cpp acd79d6 CPU vs CUDA backend, same Q4_K_M weights, greedy n=64, token-id prompts',
  rows,
}, null, 2));
for (const r of rows) console.log(`prompt ${r.id}: ${r.first_divergence === -1 ? 'identical' : 'oracle backends diverge at ' + r.first_divergence}`);
