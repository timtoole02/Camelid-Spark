# Recording the Camelid terminal demo

This file holds the exact commands to record the README's terminal demo of the
**supported TinyLlama 1.1B Chat Q8_0 path** — server start → load → first token
`29907`/`"C"` → a 50-token completion. Nothing here is run for you: record it on
a real machine against a real build so the cast shows real output, never a mock.

> The embed point is marked in `README.md` with
> `<!-- TODO(tim): record and embed demo -->`. Once the SVG is rendered, replace
> that comment with the `![…](assets/camelid-demo.svg)` image line.

## 0. Prerequisites

```bash
# one-time tooling (Homebrew + npm)
brew install asciinema
npm install -g svg-term-cli

# a Camelid binary + the baseline supported row
./camelid pull tinyllama   # downloads tinyllama-1.1b-chat-v1.0.Q8_0.gguf into ./models
```

## 1. Record the cast

Record a tight, scripted session. Keep it short (≈30–40s) so the SVG stays small
and the 15-second read still holds.

```bash
asciinema rec assets/camelid-demo.cast \
  --title "Camelid — TinyLlama 1.1B Q8_0 (supported gate)" \
  --idle-time-limit 1.5 \
  --cols 90 --rows 24
```

Inside the recording shell, run the supported path end to end:

```bash
# 1) start the server on the baseline supported row (new terminal or backgrounded)
./camelid serve --model models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf --no-open

# 2) confirm the loaded model id comes straight from the GGUF metadata
curl -s http://127.0.0.1:8181/v1/models | jq -r '.data[].id'

# 3) the canonical "hello" smoke — first token is 29907 / "C", greedy/deterministic
curl -s http://127.0.0.1:8181/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"TinyLlama 1.1B Chat","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":50,"temperature":0}' \
  | jq -r '.choices[0].message.content'
```

Exit the recording shell (`Ctrl-D` or `exit`) to finalize `assets/camelid-demo.cast`.

> Even simpler: `scripts/smoke.sh` pulls TinyLlama, serves it, does one real chat
> round-trip, and asserts on the reply. Recording `asciinema rec … -c scripts/smoke.sh`
> captures the whole proven path in a single non-interactive command.

## 2. Render an embeddable SVG

```bash
svg-term --in assets/camelid-demo.cast --out assets/camelid-demo.svg \
  --window --width 90 --height 24
```

`svg-term` renders the cast to a self-contained SVG that GitHub displays inline —
no GIF weight, selectable text, crisp at any zoom. (A GIF via `agg camelid-demo.cast
assets/camelid-demo.gif` is an acceptable fallback if SVG rendering is unavailable.)

## 3. Embed it

In `README.md`, replace the `<!-- TODO(tim): record and embed demo -->` placeholder
near the quickstart with:

```markdown
![Camelid serving TinyLlama 1.1B Q8_0 — first token 29907/"C", 50-token completion](assets/camelid-demo.svg)
```

## Honesty notes

- Record against the **real** binary and the **real** TinyLlama Q8_0 row. Do not
  hand-edit the cast to change tokens, timings, or output.
- The demo shows the **supported gate only** — it must not imply that any
  unsupported row, quant, or context length works. Keep the title row-specific.
- If the first token is ever not `29907`/`"C"` for this exact prompt and row,
  that is a regression to investigate, not a cast to trim around.
