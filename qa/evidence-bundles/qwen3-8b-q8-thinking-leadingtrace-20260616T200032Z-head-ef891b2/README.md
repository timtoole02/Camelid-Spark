# Qwen3-8B Q8_0 — thinking mode (opt-in, leading-trace lane) — HOST-BOUNDED

This bundle documents Qwen3 **thinking mode** for the 8B Q8_0 row, enabled via
`camelid_enable_thinking: true` (also `--enable-thinking` on serve/chat and the
WebUI **Thinking** toggle). When set, the ChatML renderer emits a bare
`<|im_start|>assistant\n` generation turn (no pre-filled `<think></think>`
block), so the model generates its own `<think>…</think>` reasoning. Default /
`false` keeps the parity-locked **thinking-DISABLED** mode.

**Support scope: `thinking_opt_in_leading_trace_only`.** Not a parity-locked
exact row.

## What is proven

- **Thinking engages correctly.** For all four probes both camelid and the
  llama.cpp reference begin with `<think>` (`all_probes_engage_thinking = true`).
- **Leading-trace parity to the captured window.** Greedy generation is
  **token-identical to the full llama.cpp 256-token reference for the entire
  captured 16-token window of every probe** — no divergence observed within the
  window. All four probes match through ≥5 tokens (`all_probes_match_at_5 =
  true`).
- **Template-shape byte parity** (separate, stronger, size-independent
  guarantee): the `enable_thinking=true` rendering is byte-locked by
  `qa/prompt-packs/qwen3-chatml-thinking-template-pack-v1.json` and was
  cross-checked against the same llama.cpp build's `--jinja /apply-template`.

## ⚠️ Host bound — why the window is only 16 tokens

This is **not** an observed divergence at 16 — every probe matched the reference
for the **whole** 16-token window. 16 is a **host-imposed lower bound** on the
leading-trace envelope:

- Qwen3-8B Q8_0 is **8.1 GB**. On this **16 GB** host the Q8 weight pages do not
  stay RAM-resident — they thrash the page cache and are re-read from the
  **degraded T7 (~14 MB/s)** every forward pass (**~53 s/token measured**, both
  on the resident-block and the file-backed paths). A full 256-token × 4-probe
  trace is **~15 hours**.
- The **GPU-resident lane is unavailable** in this environment (`GPU: none
  detected` — CPU-only), so the one-read-then-fast-decode path is not on.
- The internal SSD has **3.8 GB free**, so the 8.1 GB model cannot be staged on
  fast local storage.

So the leading trace was captured to a bounded 16-token window. The **full**
llama.cpp 8B reference (256 tokens/probe) **is** captured in
`qwen3-8b-thinking-leadingtrace.json` — only the camelid side is host-bounded.

The full envelope is expected to be substantially larger (cf. 1.7B 26–205, 4B
35–235, 0.6B 6–126) and **remains to be captured on a host where 8B stays
resident** — a GPU-resident Mac, a >16 GB host, or one with fast local SSD (e.g.
the **mini2** path used for the 8B thinking-DISABLED chatml-parity row). Until
then the 8B thinking lane carries: template-shape byte parity (full) +
leading-trace parity to ≥16 tokens (host-bounded).

## What is NOT claimed

Full token-parity over an entire thinking trace, and any leading-trace envelope
beyond the captured 16-token window for this row. The parity-locked exact-row
support mode remains thinking-DISABLED.

## Comparator & path

- Reference: `llama.cpp b9430 (d48a56eff)`, `-ngl 0 -ctk f32 -ctv f32 -fa off
  --no-repack -c 16384`, fed the identical bare-assistant thinking-enabled ChatML
  prompt via `/completion` (full 256-token reference captured).
- camelid: `cpu_reference` (`CAMELID_METAL_NOCOPY=0 CAMELID_METAL_RESIDENT_*=0`),
  16-token host-bounded window, via `/v1/chat/completions` with
  `camelid_enable_thinking:true`.

## Reproduce

```
# reference (full 256-token trace):
llama-server -m Qwen3-8B-Q8_0.gguf -ngl 0 -ctk f32 -ctv f32 -fa off --no-repack -c 16384 --port 8090
node scripts/thinking-leadingtrace-qwen3.mjs --mode reference --llama http://127.0.0.1:8090 --out ref.json
# camelid (raise --n on a host where 8B stays resident):
CAMELID_METAL_NOCOPY=0 CAMELID_METAL_RESIDENT_DECODE=0 CAMELID_METAL_RESIDENT_PREFILL=0 camelid serve --addr 127.0.0.1:8185 --model Qwen3-8B-Q8_0.gguf --no-open
node scripts/thinking-leadingtrace-qwen3.mjs --mode camelid --camelid http://127.0.0.1:8185 --model-id "Qwen3 8B Instruct" --n 16 --out cam.json
node scripts/thinking-leadingtrace-qwen3.mjs --mode compare --ref ref.json --cam cam.json --row-id qwen3_8b_instruct_q8_0 --display-name "Qwen3 8B Instruct Q8_0" --bundle-out qwen3-8b-thinking-leadingtrace.json
```
