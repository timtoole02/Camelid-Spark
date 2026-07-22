// Item 1 — independent BPE tokenizer parity gate (Camelid qwen35 vs llama.cpp REF_QWEN35).
// Byte-exact token-ID comparison over the five-prompt corpus + 20-trace agentic prompts +
// adversarial set, in both plain (user text) and parse-special modes, plus special-token
// round-trip and unsplittability checks. Writes RECEIPT_ITEM1_tokenizer.json.
import fs from 'fs';
import os from 'os';
import path from 'path';
import { execFileSync } from 'child_process';

const HERE = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'));
const MODEL = (process.env.CAMELID_MODELS_DIR || 'models') + '/ornith-1.0-9b-Q4_K_M.gguf';
const CAMELID = path.resolve(HERE, '../../../target/release/camelid.exe');
const LLAMA_TOKENIZE = (process.env.CAMELID_LLAMACPP_BIN || 'llama.cpp/build/bin') + '/llama-tokenize.exe';

const adversarial = JSON.parse(fs.readFileSync(path.join(HERE, 'FIXTURES_tokenizer_adversarial.json'), 'utf8'));
const fiveProm = JSON.parse(fs.readFileSync(path.join(HERE, 'FIXTURES_five_prompt_parity.json'), 'utf8'));
const agentic = JSON.parse(fs.readFileSync(path.join(HERE, 'FIXTURES_agentic_20.json'), 'utf8'));

const cases = [
  ...fiveProm.prompts.map((p) => ({ src: `five-prompt/${p.id}`, text: p.text })),
  ...agentic.traces.map((t) => ({ src: `agentic/${t.id}`, text: t.prompt })),
  ...adversarial.map((text, i) => ({ src: `adversarial/${i}`, text })),
];

const SPECIAL_MARKERS = ['<|im_start|>', '<|im_end|>', '<think>', '</think>', '<tool_call>', '</tool_call>'];

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'tokgate-'));

function oracleIds(text, parseSpecial) {
  const f = path.join(tmp, 'p.txt');
  fs.writeFileSync(f, text); // UTF-8, no BOM
  const args = ['-m', MODEL, '-f', f, '--ids', '--log-disable'];
  if (!parseSpecial) args.push('--no-parse-special');
  const out = execFileSync(LLAMA_TOKENIZE, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
  const line = out.split(/\r?\n/).find((l) => l.trim().startsWith('['));
  if (!line) throw new Error(`no ids line from llama-tokenize for ${JSON.stringify(text)}:\n${out}`);
  return JSON.parse(line);
}

function camelidBatch(texts, parseSpecial) {
  const f = path.join(tmp, 'batch.json');
  fs.writeFileSync(f, JSON.stringify(texts));
  const args = ['tokenize', '--model', MODEL, '--file', f];
  if (parseSpecial) args.push('--parse-special');
  const out = execFileSync(CAMELID, args, { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
  return out
    .split(/\r?\n/)
    .filter((l) => l.trim().startsWith('{'))
    .map((l) => JSON.parse(l));
}

const receipt = {
  schema: 'camelid.parity-receipt/v1',
  gate: 'ITEM1_tokenizer',
  lane: 'ornith-9b-constrained-vram',
  date: new Date().toISOString(),
  reference: 'llama.cpp acd79d6 (REF_QWEN35) llama-tokenize',
  model: MODEL,
  modes: {},
  special_roundtrip: [],
  unsplittability: [],
  result: null,
};

let failures = 0;

for (const parseSpecial of [false, true]) {
  const modeName = parseSpecial ? 'parse_special' : 'plain';
  const cam = camelidBatch(cases.map((c) => c.text), parseSpecial);
  if (cam.length !== cases.length) throw new Error(`camelid returned ${cam.length} lines for ${cases.length} cases`);
  const mismatches = [];
  cases.forEach((c, i) => {
    const oracle = oracleIds(c.text, parseSpecial);
    const mine = cam[i].ids;
    const same = oracle.length === mine.length && oracle.every((v, j) => v === mine[j]);
    const roundtrip = cam[i].decoded === c.text;
    if (!same) mismatches.push({ case: c.src, text: c.text, camelid: mine, oracle });
    else if (!roundtrip) mismatches.push({ case: c.src, text: c.text, decode_roundtrip_failed: cam[i].decoded });
    process.stderr.write(`[${modeName}] ${c.src}: ${same ? (roundtrip ? 'OK' : 'DECODE-FAIL') : 'ID-MISMATCH'}\n`);
  });
  receipt.modes[modeName] = { cases: cases.length, mismatches };
  failures += mismatches.length;
}

// Special-token round-trip + unsplittability.
const camSpecial = camelidBatch(SPECIAL_MARKERS, true);
const camPlain = camelidBatch(SPECIAL_MARKERS, false);
SPECIAL_MARKERS.forEach((m, i) => {
  const oracle = oracleIds(m, true);
  const mine = camSpecial[i].ids;
  const same = oracle.length === mine.length && oracle.every((v, j) => v === mine[j]);
  const single = oracle.length === 1;
  receipt.special_roundtrip.push({ marker: m, oracle, camelid: mine, match: same, single_token: single, decoded: camSpecial[i].decoded });
  if (!same) failures++;
  if (single) {
    // Plain-mode behavior must equal the oracle's. NOTE the two token kinds differ
    // by design (llama.cpp partition rule): CONTROL markers (<|im_start|>...) must
    // NOT resolve to their special id from plain user text ("unsplittable"), while
    // USER_DEFINED markers (<think>, <tool_call>...) resolve to their single id in
    // BOTH modes — the oracle itself does this, so a "leak" there is correct
    // behavior, recorded as info only. The gate criterion is oracle equality.
    const leaked = camPlain[i].ids.includes(oracle[0]);
    const oraclePlain = oracleIds(m, false);
    const oracleLeaked = oraclePlain.includes(oracle[0]);
    const plainMatch = oraclePlain.length === camPlain[i].ids.length && oraclePlain.every((v, j) => v === camPlain[i].ids[j]);
    receipt.unsplittability.push({ marker: m, special_id: oracle[0], kind: oracleLeaked ? 'user_defined (always merged — leak is correct)' : 'control (must not leak)', leaked_in_plain_mode: leaked, oracle_leaks_in_plain_mode: oracleLeaked, plain_ids_match_oracle: plainMatch });
    if (!plainMatch || (leaked && !oracleLeaked)) failures++;
  }
});

receipt.result = failures === 0 ? 'PASS' : `FAIL (${failures} failures)`;
fs.writeFileSync(path.join(HERE, 'RECEIPT_ITEM1_tokenizer.json'), JSON.stringify(receipt, null, 2));
console.log(`\nITEM 1 TOKENIZER GATE: ${receipt.result}`);
if (failures > 0) {
  for (const [mode, data] of Object.entries(receipt.modes)) {
    for (const mm of data.mismatches) console.log(`  [${mode}] ${mm.case}: ${JSON.stringify(mm).slice(0, 300)}`);
  }
  process.exitCode = 1;
}
