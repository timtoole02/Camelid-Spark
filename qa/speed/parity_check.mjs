// Greedy-parity analyzer: reads newline-delimited bench-generate JSON, extracts
// output_token_ids, computes sha256, confirms determinism, finds first divergence.
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

function loadRuns(path) {
  return readFileSync(path, 'utf8')
    .split('\n').map(l => l.trim()).filter(Boolean)
    .map(l => JSON.parse(l));
}
function ids(rec) { return rec.output_token_ids || rec.outputTokenIds || []; }
function sha(arr) { return createHash('sha256').update(arr.join(',')).digest('hex').slice(0, 16).toUpperCase(); }
function tps(rec) { return rec.tokens_per_second; }
function firstDiff(a, b) {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return i;
  return a.length === b.length ? -1 : n;
}

const [, , blPath, candPath] = process.argv;
const bl = loadRuns(blPath);
const cand = loadRuns(candPath);

const blIds = bl.map(ids), candIds = cand.map(ids);
console.log('=== BASELINE (flag off) ===');
bl.forEach((r, i) => console.log(`  iter${i}: ${blIds[i].length} tok, sha ${sha(blIds[i])}, ${tps(r).toFixed(2)} t/s`));
const blDet = blIds.every(x => sha(x) === sha(blIds[0]));
console.log(`  determinism: ${blDet ? 'OK (all iters identical)' : 'BROKEN — non-deterministic!'}`);

console.log('=== CANDIDATE (CAMELID_ATTN_COALESCED=1) ===');
cand.forEach((r, i) => console.log(`  iter${i}: ${candIds[i].length} tok, sha ${sha(candIds[i])}, ${tps(r).toFixed(2)} t/s`));

const fd = firstDiff(blIds[0], candIds[0]);
console.log('=== PARITY VERDICT ===');
if (fd === -1) {
  console.log(`  PASS — candidate token sequence IDENTICAL to baseline (greedy parity holds)`);
} else {
  console.log(`  FAIL — first divergence at token index ${fd}`);
  const lo = Math.max(0, fd - 2), hi = Math.min(blIds[0].length, fd + 3);
  console.log(`  baseline [${lo}..${hi}): ${blIds[0].slice(lo, hi).join(', ')}`);
  console.log(`  candidate[${lo}..${hi}): ${candIds[0].slice(lo, hi).join(', ')}`);
  console.log(`  -> ${fd} identical tokens before the flip (near-tie flip if fd is large + output coherent; bug if fd is tiny)`);
}
