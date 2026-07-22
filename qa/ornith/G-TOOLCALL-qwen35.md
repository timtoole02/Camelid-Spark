# ORNITH 9B — Phase 2 G-TOOLCALL receipt (template + reasoning + Bug-1 lift)

**Gate:** G-TOOLCALL — Ornith's tool-call emission lifts cleanly into structured
`tool_calls` (correct name + args, single parse), reasoning isolated, render holds.
**Result:** **PASS.**
**Date:** 2026-06-29 · **Platform:** Windows x86_64 (MSVC), CPU. **Model:** `qwen35`
`ornith-1.0-9b-Q8_0.gguf`. **Lane:** runnable serve bridge (`CAMELID_RUNNABLE_SERVE=1`).

## What was built (Phase 2)

- **Native Ornith ChatML renderer** (`render_ornith_chatml_prompt[_with_tools]`,
  `src/api/mod.rs`): the GGUF template uses Python str-methods / `messages[::-1]` /
  macros that Camelid's minijinja cannot evaluate (it falls back to a raw prompt), so
  Ornith is rendered natively, mirroring the Qwen3 ChatML path. Tools are injected via
  the EXACT `# Tools … <tools> … </tools>` system block + the model's custom
  `<tool_call><function=…>` instruction literal (byte-for-byte from the template —
  `ORNITH_TOOL_INSTRUCTIONS`). Generation prompt opens `<think>` (or an empty think
  block when thinking is disabled). Tool results are wrapped in `<tool_response>` turns.
- **Reasoning channel** (`split_ornith_think`): the model's `<think>…</think>` is
  split out into `reasoning_content`; it never leaks into `content` or tool parsing.
- **Bug-1 tool lift**: Ornith's custom XML
  `<tool_call><function=NAME><parameter=ARG>VALUE</parameter>…</function></tool_call>`
  (NOT JSON — `parse_hermes` would return nothing) is lifted by a dedicated parser:
  `parse_ornith` (chat lane, `src/chat/tool_parse.rs`, routed by the `ornith`/`qwen35`
  family before the qwen/hermes JSON arm) and `parse_ornith_tool_calls_json` (api lane,
  OpenAI `tool_calls`). 5 unit tests cover reasoning-isolation, multi-param, JSON
  values, two-calls, and plain-answer (no false fire).

## Live evidence (served `/v1/chat/completions`, 1-tool probe)

Prompt: user asks the line count of `notes.txt`, with the `read_file` tool supplied.
Ornith's served response (greedy):

```
content:    <tool_call>\n<function=read_file>\n<parameter=path>\nnotes.txt\n</parameter>\n</function>\n</tool_call>
tool_calls: [{"id":"call_0","type":"function",
              "function":{"name":"read_file","arguments":"{\"path\":\"notes.txt\"}"}}]
finish_reason: tool_calls
```

The model emitted the exact trained format; the bridge lifted it into a single,
correct structured call (`name=read_file`, `args={path: notes.txt}`) with no
double-parse and no contamination. The content retains the tool-call text so the
agent loop's client-side `parse_ornith` lifts it identically.

## Scope / honesty

- This certifies the render + reasoning-split + tool-lift (the Bug-1 gate). The full
  agent loop (execute → use result → answer) over a multi-case battery is the separate
  Phase-4 `agent-eval` certificate, which gates `tool_capable`.
- Speed: the runnable lane is pure-f32 (~1 s/token for this 9B); a long tools prompt
  prefills in minutes. The agent-client read timeout was raised to support the slow
  oracle lane. A SIMD/optimized-lane kernel for practical speed is a documented
  follow-up — it does not affect tool-call CAPABILITY, only latency.
