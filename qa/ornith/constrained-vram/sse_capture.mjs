// Item 3 — archive one full streamed SSE capture from the camelid serve
// (reasoning_content deltas + tool_calls + stream_options.include_usage), for
// RECEIPT_ITEM3_serving.json. Assumes camelid serve already running on :11434-style
// addr (pass as argv[2], default http://127.0.0.1:8080) with the Ornith model active
// (CAMELID_RUNNABLE_SERVE=1 [+ CAMELID_QWEN35_CUDA=1 for the GPU lane]).
import fs from 'fs';
import path from 'path';

const HERE = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'));
const BASE = process.argv[2] || 'http://127.0.0.1:8080';
const fixtures = JSON.parse(fs.readFileSync(path.join(HERE, 'FIXTURES_agentic_20.json'), 'utf8'));
const tools = fixtures.tools.map((t) => ({ type: 'function', function: t }));

const res = await fetch(`${BASE}/v1/chat/completions`, {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify({
    messages: [{ role: 'user', content: fixtures.traces[0].prompt }],
    tools,
    stream: true,
    stream_options: { include_usage: true },
    camelid_enable_thinking: true,
    temperature: 0,
    max_tokens: 700,
  }),
});
if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);

const raw = await res.text();
fs.writeFileSync(path.join(HERE, 'sse_capture_item3.txt'), raw);

// Summarize the stream for the receipt.
const events = raw.split(/\n\n/).filter((b) => b.startsWith('data: ') && !b.includes('[DONE]'))
  .map((b) => JSON.parse(b.slice(6)));
const summary = {
  chunks: events.length,
  reasoning_delta_chunks: events.filter((e) => e.choices?.[0]?.delta?.reasoning_content).length,
  content_delta_chunks: events.filter((e) => e.choices?.[0]?.delta?.content).length,
  tool_call_delta_chunks: events.filter((e) => e.choices?.[0]?.delta?.tool_calls).length,
  finish_reasons: [...new Set(events.map((e) => e.choices?.[0]?.finish_reason).filter(Boolean))],
  usage_chunk_present: events.some((e) => e.usage && (!e.choices || e.choices.length === 0)),
  usage: events.findLast((e) => e.usage)?.usage ?? null,
  reasoning_leaked_into_content: events.some((e) => (e.choices?.[0]?.delta?.content || '').includes('<think>')),
};
fs.writeFileSync(path.join(HERE, 'sse_capture_item3_summary.json'), JSON.stringify(summary, null, 2));
console.log(JSON.stringify(summary, null, 2));
