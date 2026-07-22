// Item 5 — acceptance-rate economics reduction. Inputs (logs from the Rust harness):
//   argmax_draft.log     ARGMAX[i] ... [ids]   (Q3_K_M, GPU resident)
//   argmax_verifier.log  ARGMAX[i] ... [ids]   (Q8_0, CPU int8 path)
//   [argmax_q6k.log]     optional subset       (Q6_K, CPU generic — equivalence check)
//   verify_cost.log      VERIFY_COST rep=R k=K prefix=P secs=S
// Plus rates passed via env: DRAFT_TOKS (GPU draft tok/s), BASELINE_TOKS (best
// verifier-quality baseline tok/s). Writes RECEIPT_ITEM5_acceptance_economics.json.
import fs from 'fs';
import path from 'path';

const HERE = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'));
const read = (f) => fs.readFileSync(path.join(HERE, f), 'utf8');
const parseStreams = (log) => {
  const out = [];
  for (const m of log.matchAll(/ARGMAX\[(\d+)\] len=\d+ secs=([\d.]+) (\[[^\]]*\])/g))
    out[Number(m[1])] = { secs: Number(m[2]), ids: JSON.parse(m[3]) };
  return out;
};

const draft = parseStreams(read('argmax_draft.log'));
const verif = parseStreams(read('argmax_verifier.log'));
if (draft.length !== verif.length) throw new Error('stream count mismatch');

// Per-position agreement bits (position i = both models' greedy next-token after
// the same teacher-forced prefix). Skip i=0 (BOS-less first-token noise is real
// but irrelevant: spec rounds always start from an accepted context).
const bits = [];
const perTrace = [];
for (let t = 0; t < draft.length; t++) {
  const a = draft[t].ids, b = verif[t].ids;
  const n = Math.min(a.length, b.length);
  let agree = 0;
  const tb = [];
  for (let i = 1; i < n; i++) {
    const same = a[i] === b[i] ? 1 : 0;
    tb.push(same);
    agree += same;
  }
  bits.push(...tb);
  perTrace.push({ trace: t + 1, positions: n - 1, agreement: +(agree / (n - 1)).toFixed(4) });
}
const alpha = bits.reduce((x, y) => x + y, 0) / bits.length;

// Expected accepted-run length per draft window k, measured empirically:
// slide over the bit stream; a round starting at i accepts the run of leading
// 1s capped at k (the standard greedy-spec accept rule), +1 for the verifier's
// free corrected/bonus token.
const K = [4, 6, 8, 12];
const runStats = {};
for (const k of K) {
  let rounds = 0, accepted = 0, full = 0;
  let i = 0;
  while (i < bits.length) {
    let run = 0;
    while (run < k && i + run < bits.length && bits[i + run] === 1) run++;
    accepted += run;
    if (run === k) full++;
    rounds++;
    i += run + 1; // rejected (or bonus) position consumed by the verifier
  }
  runStats[k] = {
    rounds,
    mean_accepted_run: +(accepted / rounds).toFixed(3),
    mean_tokens_per_round: +((accepted + rounds) / rounds).toFixed(3), // +1 verifier token
    full_window_rate: +(full / rounds).toFixed(3),
  };
}

// Disagreement clustering: distribution of gaps between consecutive zeros.
const zeroIdx = bits.flatMap((b, i) => (b === 0 ? [i] : []));
const gaps = zeroIdx.slice(1).map((z, j) => z - zeroIdx[j]);
const clustered = gaps.filter((g) => g <= 3).length / Math.max(gaps.length, 1);

// Verify cost: median marginal secs for P+k over P, per k.
const vc = {};
for (const m of read('verify_cost.log').matchAll(/VERIFY_COST rep=(\d+) k=(\d+) prefix=\d+ secs=([\d.]+)/g)) {
  const k = Number(m[2]);
  (vc[k] ||= []).push(Number(m[3]));
}
const med = (a) => a.slice().sort((x, y) => x - y)[Math.floor(a.length / 2)];
const verifyMarginal = {};
for (const k of K) verifyMarginal[k] = +(med(vc[k]) - med(vc[0])).toFixed(4);
const verifyBase = +med(vc[0]).toFixed(4);

// Net tok/s model: pipelined rounds — GPU drafts window n+1 while CPU verifies
// window n, so round time = max(k/draft_rate, verify_batch_time); throughput =
// mean_tokens_per_round / round_time. Verify batch time = FULL prefill of
// (context+k)?? No: the Item 6 verifier keeps its own KV/state and only
// forwards the k new tokens + 1 — the marginal batched cost measured above.
const draftToks = Number(process.env.DRAFT_TOKS || '0');
const baselineToks = Number(process.env.BASELINE_TOKS || '0');
const model = {};
for (const k of K) {
  const draftTime = k / draftToks;
  const verifyTime = verifyMarginal[k];
  const roundTime = Math.max(draftTime, verifyTime);
  const netToks = runStats[k].mean_tokens_per_round / roundTime;
  model[k] = {
    draft_window_secs: +draftTime.toFixed(4),
    verify_marginal_secs: verifyTime,
    round_secs_pipelined: +roundTime.toFixed(4),
    net_tok_s: +netToks.toFixed(2),
    speedup_vs_baseline: baselineToks ? +(netToks / baselineToks).toFixed(2) : null,
  };
}

const goK = K.filter((k) => model[k].net_tok_s >= 1.2 * baselineToks && alpha >= 0.8);
const receipt = {
  schema: 'camelid.parity-receipt/v1',
  gate: 'ITEM5_acceptance_economics',
  lane: 'ornith-9b-constrained-vram',
  date: new Date().toISOString(),
  method: 'teacher-forced greedy agreement over the frozen 20-trace corpus (12,092 tokens): both models consume identical prefixes (the trace tokens) and their per-position greedy argmax streams are compared. Draft = Q3_K_M on the Camelid resident GPU engine; verifier = Q8_0 on the Camelid CPU int8x18 path (the fast-kernel verifier candidate; Q6_K subset equivalence check separate). Verify cost = measured marginal batched-prefill time for +k tokens.',
  agreement: {
    positions: bits.length,
    overall_top1_agreement: +alpha.toFixed(4),
    per_trace: perTrace,
    disagreement_gap_leq3_fraction: +clustered.toFixed(3),
  },
  run_stats: runStats,
  verify_cost: { base_prefill_secs_at_prefix: verifyBase, marginal_secs_per_k: verifyMarginal },
  rates: { draft_gpu_tok_s: draftToks, baseline_tok_s: baselineToks, baseline_desc: process.env.BASELINE_DESC || '' },
  net_model: model,
  thresholds: { min_speedup: 1.2, min_acceptance: 0.8 },
  go_windows: goK,
  result: goK.length ? `GO (k=${goK.join(',')})` : 'NO-GO',
};
fs.writeFileSync(path.join(HERE, 'RECEIPT_ITEM5_acceptance_economics.json'), JSON.stringify(receipt, null, 2));
console.log(JSON.stringify({ alpha: receipt.agreement.overall_top1_agreement, run_stats: runStats, verifyMarginal, net_model: model, result: receipt.result }, null, 2));
