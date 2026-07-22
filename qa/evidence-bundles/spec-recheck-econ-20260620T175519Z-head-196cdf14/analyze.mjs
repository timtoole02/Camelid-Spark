// SPEC_RECHECK Phase 1 analysis: read results/*.jsonl, emit matrix tables to stdout.
// Usage: node analyze.mjs [kind ...]   (default: ngram draft-gpu draft-cpu)
import { readFileSync, existsSync } from "node:fs";

const kinds = process.argv.slice(2);
const want = kinds.length ? kinds : ["ngram", "draft-gpu", "draft-cpu"];
const WORKLOADS = ["code", "json", "extraction", "chat", "creative", "adversarial"];
const GAMMAS = [2, 4, 6, 7];
const dir = new URL(".", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");

function load(kind) {
  const p = `${dir}/results/${kind}.jsonl`;
  if (!existsSync(p)) return null;
  const rows = readFileSync(p, "utf8")
    .split(/\r?\n/)
    .filter((l) => l.trim().startsWith("{"))
    .map((l) => JSON.parse(l));
  return rows;
}

function cell(rows, w, g) {
  return rows.find((r) => r.workload === w && r.draft_tokens === g);
}

function fmt(n, d = 2) {
  return n === undefined || n === null ? "—" : Number(n).toFixed(d);
}

for (const kind of want) {
  const rows = load(kind);
  if (!rows) { console.log(`\n## ${kind}: (no results)\n`); continue; }
  const anyLossy = rows.some((r) => !r.lossless);
  console.log(`\n## ${kind} — ${rows.length} cells | lossless: ${anyLossy ? "FAIL (see below)" : "ALL ✓"}`);

  // S_sync matrix
  console.log(`\n### S_sync (spec t/s ÷ plain t/s)\n`);
  console.log(`| workload | γ=2 | γ=4 | γ=6 | γ=7 |`);
  console.log(`|---|---|---|---|---|`);
  for (const w of WORKLOADS) {
    const cells = GAMMAS.map((g) => {
      const c = cell(rows, w, g);
      if (!c) return "—";
      const mark = c.s_sync > 1 ? "" : " ✗";
      return `${fmt(c.s_sync)}×${mark}`;
    });
    console.log(`| ${w} | ${cells.join(" | ")} |`);
  }

  // Accept-rate matrix
  console.log(`\n### accept rate (accepted drafts ÷ drafted)\n`);
  console.log(`| workload | γ=2 | γ=4 | γ=6 | γ=7 |`);
  console.log(`|---|---|---|---|---|`);
  for (const w of WORKLOADS) {
    const cells = GAMMAS.map((g) => {
      const c = cell(rows, w, g);
      return c ? `${fmt(c.accept_rate * 100, 1)}%` : "—";
    });
    console.log(`| ${w} | ${cells.join(" | ")} |`);
  }

  // f_draft + plain/spec tps detail (best gamma per workload by S_sync)
  console.log(`\n### detail at best γ per workload (by S_sync)\n`);
  console.log(`| workload | best γ | accept% | tok/round | f_draft | plain t/s | spec t/s | S_sync | gpu/cpu verify | lossless |`);
  console.log(`|---|---|---|---|---|---|---|---|---|---|`);
  for (const w of WORKLOADS) {
    const cs = GAMMAS.map((g) => cell(rows, w, g)).filter(Boolean);
    if (!cs.length) { console.log(`| ${w} | — | | | | | | | | |`); continue; }
    const best = cs.reduce((a, b) => (b.s_sync > a.s_sync ? b : a));
    console.log(
      `| ${w} | ${best.draft_tokens} | ${fmt(best.accept_rate * 100, 1)}% | ${fmt(best.mean_accepted_tokens_per_round)} | ` +
      `${fmt(best.f_draft, 4)} | ${fmt(best.plain_tokens_per_second)} | ${fmt(best.spec_tokens_per_second)} | ` +
      `${fmt(best.s_sync)}× | ${best.gpu_verify_rounds}/${best.cpu_verify_rounds} | ${best.lossless ? "✓" : "✗ @" + best.first_divergent_generated_token_index} |`
    );
  }

  // Lossless failures, if any
  const fails = rows.filter((r) => !r.lossless);
  if (fails.length) {
    console.log(`\n### ⚠ LOSSLESS FAILURES\n`);
    for (const f of fails) {
      console.log(`- ${f.workload} γ=${f.draft_tokens}: first divergence @ generated index ${f.first_divergent_generated_token_index}`);
    }
  }
}
