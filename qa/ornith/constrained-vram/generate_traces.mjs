// Item 0/4 — generate the 20 agentic-coding traces from the pinned oracle (REF_QWEN35,
// CUDA). Each fixture prompt runs once with the model-card sampling profile
// (temp 0.6 / top_p 0.95 / top_k 20, fixed seed) through the oracle's own chat
// template + tool schemas, so traces contain real <think> blocks and qwen3_xml
// <tool_call> emissions. Output: TRACES_agentic_20.json (frozen corpus for imatrix
// calibration + acceptance-rate work) and TRACES_agentic_20.txt (raw text concat for
// llama-imatrix -f).
//
// Assumes llama-server (REF_QWEN35) already running on 127.0.0.1:8113 with the
// Ornith GGUF loaded (started by the caller; see RECEIPT for the exact command).
import fs from 'fs';
import path from 'path';

const HERE = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'));
const BASE = 'http://127.0.0.1:8113';
const fixtures = JSON.parse(fs.readFileSync(path.join(HERE, 'FIXTURES_agentic_20.json'), 'utf8'));

// Plausible synthetic tool results, keyed by tool; the point is realistic multi-turn
// shape (call -> response -> continuation), not real files.
const toolResult = (name, args) => {
  if (name === 'list_dir') return 'notes.txt\nconfig.py\nutils/\ntests/\nREADME.md';
  if (name === 'read_file')
    return `# ${args.path || 'file'}\nDEFAULT_TIMEOUT = 30\nRETRIES = 3\n\ndef helper(x):\n    return x * 2\n`;
  if (name === 'write_file') return 'ok: wrote ' + (args.path || 'file');
  return 'ok';
};

async function chat(messages, tools) {
  const res = await fetch(`${BASE}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      messages,
      tools,
      temperature: 0.6,
      top_p: 0.95,
      top_k: 20,
      seed: 7,
      max_tokens: 900,
    }),
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return (await res.json()).choices[0];
}

const tools = fixtures.tools.map((t) => ({ type: 'function', function: t }));
const out = [];
for (const t of fixtures.traces) {
  const messages = [{ role: 'user', content: t.prompt }];
  const turns = [];
  for (let turn = 0; turn < 4; turn++) {
    const choice = await chat(messages, t.kind === 'reasoning' ? undefined : tools);
    const msg = choice.message;
    turns.push({ reasoning_content: msg.reasoning_content ?? null, content: msg.content ?? null, tool_calls: msg.tool_calls ?? null, finish_reason: choice.finish_reason });
    messages.push(msg);
    if (choice.finish_reason === 'tool_calls' && msg.tool_calls?.length) {
      for (const tc of msg.tool_calls) {
        let args = {};
        try { args = JSON.parse(tc.function.arguments); } catch {}
        messages.push({ role: 'tool', tool_call_id: tc.id, content: toolResult(tc.function.name, args) });
      }
      continue;
    }
    break;
  }
  out.push({ id: t.id, kind: t.kind, prompt: t.prompt, turns });
  console.log(`trace ${t.id} (${t.kind}): ${turns.length} turn(s), last finish=${turns.at(-1).finish_reason}`);
}

fs.writeFileSync(path.join(HERE, 'TRACES_agentic_20.json'), JSON.stringify({
  schema: 'camelid.fixture-corpus/v1',
  generator: 'REF_QWEN35 (llama.cpp acd79d6 CUDA) llama-server, temp 0.6/top_p 0.95/top_k 20, seed 7',
  date: new Date().toISOString(),
  traces: out,
}, null, 2));

// Raw text for imatrix calibration: prompts + think + content + tool text, concatenated.
const raw = out.map((tr) =>
  [tr.prompt, ...tr.turns.flatMap((u) => [u.reasoning_content, u.content, u.tool_calls ? JSON.stringify(u.tool_calls) : null])]
    .filter(Boolean).join('\n')
).join('\n\n----\n\n');
fs.writeFileSync(path.join(HERE, 'TRACES_agentic_20.txt'), raw);
console.log(`raw calibration text: ${raw.length} chars`);
