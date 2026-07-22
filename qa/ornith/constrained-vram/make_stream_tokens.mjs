// Item 5 — build stream_tokens.json: one token array per trace (the trace's full
// raw text: prompt + think + content + tool_calls JSON), tokenized through the
// byte-certified camelid tokenizer (plain mode — raw text; user_defined markers
// like <think> merge to their ids in both modes per the ITEM1-certified rule).
import fs from 'fs';
import path from 'path';
import { execFileSync } from 'child_process';

const HERE = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'));
const MODEL = (process.env.CAMELID_MODELS_DIR || 'models') + '/ornith-1.0-9b-Q3_K_M.gguf';
const CAMELID = path.resolve(HERE, '../../../target/release/camelid.exe');
const traces = JSON.parse(fs.readFileSync(path.join(HERE, 'TRACES_agentic_20.json'), 'utf8')).traces;

const texts = traces.map((tr) =>
  [tr.prompt, ...tr.turns.flatMap((u) => [u.reasoning_content, u.content, u.tool_calls ? JSON.stringify(u.tool_calls) : null])]
    .filter(Boolean).join('\n')
);
const batch = path.join(HERE, 'stream_texts.json');
fs.writeFileSync(batch, JSON.stringify(texts));
const out = execFileSync(CAMELID, ['tokenize', '--model', MODEL, '--file', batch], { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
const tokenSets = out.split(/\r?\n/).filter((l) => l.trim().startsWith('{')).map((l) => JSON.parse(l).ids);
if (tokenSets.length !== texts.length) throw new Error(`${tokenSets.length} != ${texts.length}`);
fs.writeFileSync(path.join(HERE, 'stream_tokens.json'), JSON.stringify(tokenSets));
console.log('sequences:', tokenSets.length, '| total tokens:', tokenSets.reduce((a, t) => a + t.length, 0), '| lens:', tokenSets.map((t) => t.length).join(','));
