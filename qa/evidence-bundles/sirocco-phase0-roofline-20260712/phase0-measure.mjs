// SIROCCO Phase 0 measurement harness.
// Interleaved (ABAB) decode timing over bench-generate, with per-run clock/throttle guard.
// Usage: node phase0-measure.mjs <mode> [args]
//   mode=regress   : interleave P1/P2/P3 at n=256, regress BW_eff & C, check P3 confound
//   mode=pc1       : quadratic sweep on P1 at n=64/256/1024/2048
//   mode=pc3       : context sweep on P1 at ctx=0/1k/4k/16k/32k (n=256 decode each)
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';

const CAMELID = '~/Camelid/target/release/camelid.exe';
const MODELS = {
  P1: { path: '~/Camelid/models/Llama-3.2-1B-Instruct-Q8_0.gguf', B_GB: 1.313251456, label: 'P1 1B Q8_0' },
  P2: { path: '~/Camelid/models/Llama-3.2-3B-Instruct-Q8_0.gguf', B_GB: 3.414061312, label: 'P2 3B Q8_0' },
  P3: { path: '~/Camelid/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf', B_GB: 0.799862912, label: 'P3 1B Q4KM' },
};
const PIN_SM = Number(process.env.PIN_SM || 0);   // expected pinned SM clock (MHz); 0 = don't guard
const PIN_TOL = 0.03;                              // allowed +/- fraction
const COOLDOWN = Number(process.env.COOLDOWN || 0); // seconds idle between rounds (thermal control)
const delay = ms => new Promise(r => setTimeout(r, ms));

function clockSnap() {
  try {
    const out = execFileSync('nvidia-smi',
      ['--query-gpu=clocks.sm,clocks.mem,temperature.gpu,clocks_throttle_reasons.active',
       '--format=csv,noheader,nounits'], { encoding: 'utf8' }).trim();
    const [sm, mem, temp, thr] = out.split(',').map(s => s.trim());
    return { sm: +sm, mem: +mem, temp: +temp, throttle: thr };
  } catch (e) {
    // field name differs on some drivers; fall back to clocks only
    try {
      const out = execFileSync('nvidia-smi',
        ['--query-gpu=clocks.sm,clocks.mem,temperature.gpu','--format=csv,noheader,nounits'],
        { encoding: 'utf8' }).trim();
      const [sm, mem, temp] = out.split(',').map(s => s.trim());
      return { sm: +sm, mem: +mem, temp: +temp, throttle: 'n/a' };
    } catch (e2) { return { sm: 0, mem: 0, temp: 0, throttle: 'err' }; }
  }
}

function runOnce(model, { prompt, promptFile, maxTokens }) {
  const args = ['bench-generate', model.path, '--max-tokens', String(maxTokens),
    '--temperature', '0', '--warmup', '--iterations', '1'];
  if (promptFile) args.push('--prompt-file', promptFile);
  else args.push('--prompt', prompt || 'Hi');
  const before = clockSnap();
  const stdout = execFileSync(CAMELID, args, { encoding: 'utf8', env: { ...process.env, CAMELID_LOG: 'error' } });
  const after = clockSnap();
  const line = stdout.trim().split('\n').filter(l => l.startsWith('{')).pop();
  const j = JSON.parse(line);
  // Reject runs whose post-run throttle snapshot still shows a THERMAL/HW-slowdown bit
  // (these persist briefly after a hot run). Primary thermal control is the per-round
  // cooldown; this is the secondary catch. Idle/power-cap bits are NOT rejections.
  const THERMAL = 0x08n | 0x20n | 0x40n | 0x80n;
  let thrBits = 0n; try { thrBits = BigInt(after.throttle); } catch {}
  const flagged = (thrBits & THERMAL) !== 0n;
  return {
    tok_s: j.tokens_per_second, decode_ms: j.decode_ms, gen: j.generated_tokens,
    prompt_tokens: j.prompt_tokens, prefill_ms: j.prefill_ms, ttft_ms: j.ttft_ms,
    ms_per_tok: j.decode_ms / j.generated_tokens,
    sm: after.sm, mem: after.mem, temp: after.temp, throttle: after.throttle, flagged,
  };
}

const pctl = (arr, p) => { const s = [...arr].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(p * (s.length - 1) + 0.5))]; };
const median = a => pctl(a, 0.5);

function summarize(label, runs) {
  const good = runs.filter(r => !r.flagged);
  const used = good.length ? good : runs;
  const ts = used.map(r => r.tok_s), mpt = used.map(r => r.ms_per_tok);
  return {
    label, n_runs: runs.length, n_flagged: runs.filter(r => r.flagged).length,
    tok_s_p50: median(ts), tok_s_p99: pctl(ts, 0.99), tok_s_min: Math.min(...ts), tok_s_max: Math.max(...ts),
    ms_per_tok_p50: median(mpt), var_band_pct: ((Math.max(...ts) - Math.min(...ts)) / median(ts) * 100),
    sm_clocks: used.map(r => r.sm), temps: used.map(r => r.temp), raw: runs,
  };
}

const mode = process.argv[2] || 'regress';
const ROUNDS = Number(process.argv[3] || 5);
const out = { mode, pin_sm: PIN_SM, ts: null, points: {} };

if (mode === 'regress') {
  const order = ['P1', 'P2', 'P3'];
  const acc = { P1: [], P2: [], P3: [] };
  for (let r = 0; r < ROUNDS; r++) {
    for (const k of order) {
      const res = runOnce(MODELS[k], { prompt: 'Hi', maxTokens: 256 });
      acc[k].push(res);
      console.error(`round ${r+1} ${MODELS[k].label}: ${res.tok_s.toFixed(2)} tok/s  ${res.ms_per_tok.toFixed(3)} ms/tok  sm=${res.sm} temp=${res.temp} thr=${res.throttle}${res.flagged?' [FLAG THERMAL]':''}`);
    }
    if (r < ROUNDS - 1 && COOLDOWN) { console.error(`  cooldown ${COOLDOWN}s...`); await delay(COOLDOWN * 1000); }
  }
  for (const k of order) out.points[k] = { ...summarize(MODELS[k].label, acc[k]), B_GB: MODELS[k].B_GB };
  // 2-point regression across P1,P2 using p50 ms/tok
  const t1 = out.points.P1.ms_per_tok_p50, t2 = out.points.P2.ms_per_tok_p50;
  const b1 = MODELS.P1.B_GB, b2 = MODELS.P2.B_GB, b3 = MODELS.P3.B_GB;
  const slope = (t2 - t1) / (b2 - b1);            // ms per GB
  const BW_eff = 1000 / slope;                     // GB/s
  const C = t1 - slope * b1;                        // ms
  const t3_pred = slope * b3 + C;
  const t3_meas = out.points.P3.ms_per_tok_p50;
  out.regression = { slope_ms_per_GB: slope, BW_eff_GBps: BW_eff, C_ms: C,
    P3_pred_ms: t3_pred, P3_meas_ms: t3_meas, P3_resid_pct: (t3_meas - t3_pred) / t3_pred * 100 };
  console.error(`\nREGRESSION: BW_eff=${BW_eff.toFixed(1)} GB/s  C=${C.toFixed(3)} ms  | P3 pred=${t3_pred.toFixed(3)} meas=${t3_meas.toFixed(3)} resid=${out.regression.P3_resid_pct.toFixed(1)}%`);
}

if (mode === 'pc1') {
  const ns = [64, 256, 1024, 2048];
  for (const n of ns) {
    const runs = [];
    for (let r = 0; r < ROUNDS; r++) { runs.push(runOnce(MODELS.P1, { prompt: 'Hi', maxTokens: n }));
      if (COOLDOWN) await delay(COOLDOWN * 1000); }
    out.points['n' + n] = summarize('P1 n=' + n, runs);
    console.error(`n=${n}: p50 ${out.points['n'+n].tok_s_p50.toFixed(2)} tok/s  flagged=${out.points['n'+n].n_flagged}/${ROUNDS}`);
  }
}

if (mode === 'pc3') {
  // ctx set by prompt length; build prompt files of target token counts (approx via repeated word)
  const ctxs = [0, 1024, 4096, 16384, 32768];
  const tmp = process.env.SC + '/ctxprompt.txt';
  for (const ctx of ctxs) {
    const words = Math.max(1, Math.round(ctx / 1.3)); // ~1.3 tok/word rough
    fs.writeFileSync(tmp, ctx === 0 ? 'Hi' : ('data '.repeat(words)));
    const runs = [];
    for (let r = 0; r < ROUNDS; r++) { runs.push(runOnce(MODELS.P1, { promptFile: tmp, maxTokens: 256 }));
      if (COOLDOWN) await delay(COOLDOWN * 1000); }
    out.points['ctx' + ctx] = { ...summarize('P1 ctx≈' + ctx, runs), prompt_tokens: runs[0].prompt_tokens };
    console.error(`ctx≈${ctx} (${runs[0].prompt_tokens} tok): p50 ${out.points['ctx'+ctx].tok_s_p50.toFixed(2)} tok/s`);
  }
}

const outfile = process.env.SC + `/phase0-${mode}.json`;
fs.writeFileSync(outfile, JSON.stringify(out, null, 2));
console.error('\nWrote ' + outfile);
