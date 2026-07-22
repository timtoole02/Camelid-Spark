import fs from 'fs';
import os from 'os';
import path from 'path';
import { execFileSync } from 'child_process';

const HERE = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'));
const MODEL = (process.env.CAMELID_MODELS_DIR || 'models') + '/ornith-1.0-9b-Q4_K_M.gguf';
const CAMELID = path.resolve(HERE, '../../../target/release/camelid.exe');

const adversarial = JSON.parse(fs.readFileSync(path.join(HERE, 'FIXTURES_tokenizer_adversarial.json'), 'utf8'));
const fiveProm = JSON.parse(fs.readFileSync(path.join(HERE, 'FIXTURES_five_prompt_parity.json'), 'utf8'));
const agentic = JSON.parse(fs.readFileSync(path.join(HERE, 'FIXTURES_agentic_20.json'), 'utf8'));
const cases = [
  ...fiveProm.prompts.map((p) => ({ src: `five-prompt/${p.id}`, text: p.text })),
  ...agentic.traces.map((t) => ({ src: `agentic/${t.id}`, text: t.prompt })),
  ...adversarial.map((text, i) => ({ src: `adversarial/${i}`, text })),
];
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'tokbisect-'));
for (const c of cases) {
  const f = path.join(tmp, 'one.json');
  fs.writeFileSync(f, JSON.stringify([c.text]));
  try {
    execFileSync(CAMELID, ['tokenize', '--model', MODEL, '--file', f], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
    console.log(`OK   ${c.src}`);
  } catch (e) {
    console.log(`CRASH ${c.src} :: ${JSON.stringify(c.text).slice(0, 80)} :: ${String(e.stderr).trim()}`);
  }
}
