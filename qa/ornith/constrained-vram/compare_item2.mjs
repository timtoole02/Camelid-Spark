// Item 2 — compare camelid PARITY_GEN output against a captured oracle_gen_<mode>.json
// and mint RECEIPT_ITEM2_qwen35_parity[_cuda].json.
// Usage: node compare_item2.mjs <cpu|cuda> <camelid_gen_log>
//   where camelid_gen_log contains lines: PARITY_GEN[i] [ids...]
import fs from 'fs';
import path from 'path';

const HERE = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'));
const MODE = process.argv[2] === 'cuda' ? 'cuda' : 'cpu';
const LOG = process.argv[3];
const oracle = JSON.parse(fs.readFileSync(path.join(HERE, `oracle_gen_${MODE}.json`), 'utf8'));
const log = fs.readFileSync(LOG, 'utf8');

const camelid = [];
for (const m of log.matchAll(/PARITY_GEN\[(\d+)\] (\[[^\]]*\])/g)) camelid[Number(m[1])] = JSON.parse(m[2]);
if (camelid.filter(Boolean).length !== oracle.generations.length)
  throw new Error(`camelid produced ${camelid.filter(Boolean).length} generations, oracle has ${oracle.generations.length}`);

const results = oracle.generations.map((g, i) => {
  const mine = camelid[i];
  const oracleIds = g.generated_token_ids;
  let firstDivergent = -1;
  for (let j = 0; j < Math.max(mine.length, oracleIds.length); j++) {
    if (mine[j] !== oracleIds[j]) { firstDivergent = j; break; }
  }
  return { id: g.id, n_camelid: mine.length, n_oracle: oracleIds.length, first_divergent_generated_token_index: firstDivergent, match: firstDivergent === -1, camelid: mine, oracle: oracleIds };
});

const pass = results.every((r) => r.match);
const receipt = {
  schema: 'camelid.parity-receipt/v1',
  gate: MODE === 'cuda' ? 'ITEM2_qwen35_parity_cuda' : 'ITEM2_qwen35_parity',
  lane: 'ornith-9b-constrained-vram',
  date: new Date().toISOString(),
  reference: oracle.oracle,
  model: oracle.model,
  n_predict: oracle.n_predict,
  sampling: oracle.sampling,
  method: 'identical prompt token-ID arrays fed to both engines (isolates model forward from tokenization; tokenization itself certified separately by RECEIPT_ITEM1); engines never co-resident',
  camelid_lane: MODE === 'cuda'
    ? 'runnable qwen35, CUDA resident engine (CAMELID_QWEN35_CUDA=1), Q4_K_M fully GPU-resident'
    : 'runnable qwen35 (pure-f32 oracle lane, int8x18 maddubs Q8_0 matmul), CPU',
  results: results.map(({ camelid: _c, oracle: _o, ...rest }) => rest),
  generations: results,
  result: pass ? 'PASS' : 'FAIL',
};
const name = MODE === 'cuda' ? 'RECEIPT_ITEM2_qwen35_parity_cuda.json' : 'RECEIPT_ITEM2_qwen35_parity.json';
fs.writeFileSync(path.join(HERE, name), JSON.stringify(receipt, null, 2));
console.log(`ITEM2 (${MODE}): ${receipt.result}`);
for (const r of receipt.results) console.log(`  prompt ${r.id}: ${r.match ? 'identical' : 'DIVERGED at ' + r.first_divergent_generated_token_index} (${r.n_camelid}/${r.n_oracle} tokens)`);
process.exitCode = pass ? 0 : 1;
