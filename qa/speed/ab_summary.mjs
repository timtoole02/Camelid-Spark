// Summarize an interleaved A/B spike run: median decode t/s per phase, ratio,
// and greedy-parity sha across all rounds. Arg: the ab/ dir.
import { readFileSync, readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';

const dir = process.argv[2];
const rec = (f) => JSON.parse(readFileSync(`${dir}/${f}`, 'utf8').trim().split('\n').filter(Boolean)[0]);
const sha = (a) => createHash('sha256').update((a || []).join(',')).digest('hex').slice(0, 16).toUpperCase();
const median = (xs) => { const s = [...xs].sort((a, b) => a - b); const n = s.length; return n ? (n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2) : NaN; };

const files = readdirSync(dir).filter(f => /^[bc]_\d+\.json$/.test(f));
const phases = { b: [], c: [] };
for (const f of files) { const r = rec(f); phases[f[0]].push({ tps: r.tokens_per_second, sha: sha(r.output_token_ids), n: (r.output_token_ids || []).length }); }

for (const p of ['b', 'c']) {
  const ts = phases[p].map(x => x.tps);
  const shas = [...new Set(phases[p].map(x => x.sha))];
  console.log(`${p === 'b' ? 'BASELINE (flag off) ' : 'CANDIDATE (coalesced)'}: n=${ts.length}  median ${median(ts).toFixed(2)} t/s  [${ts.map(x => x.toFixed(1)).join(', ')}]  sha{${shas.join('|')}}`);
}
const mb = median(phases.b.map(x => x.tps)), mc = median(phases.c.map(x => x.tps));
const allShas = new Set([...phases.b, ...phases.c].map(x => x.sha));
console.log('---');
console.log(`PARITY: ${allShas.size === 1 ? 'PASS — all baseline+candidate runs share one token-id sha (greedy parity holds)' : 'FAIL — token-id sha differs:\n  ' + JSON.stringify({ baseline: [...new Set(phases.b.map(x => x.sha))], candidate: [...new Set(phases.c.map(x => x.sha))] })}`);
console.log(`SPEED:  candidate/baseline ratio = ${(mc / mb).toFixed(3)}x  (kill bar: >=1.15x at depth; baseline ~${mb.toFixed(1)}, candidate ~${mc.toFixed(1)} t/s)`);
console.log(`VERDICT: ${allShas.size !== 1 ? 'PARITY-NULL' : (mc / mb >= 1.15 ? 'SPEED PASS (>=+15%)' : 'SPEED-NULL (<+15%, coalescing does not stack enough on split-K)')}`);
