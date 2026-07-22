# FLINT_HANDOFF.md — DGX Spark tester steps

This is the **acceptance smoke** for the FLINT test build. It is purely *"does it run"* —
there is no comparison to any reference. **Pass = all four steps below produce tokens
without crashing.** Everything runs on the Spark itself.

Build first (see [`README.md`](README.md)):

```bash
git clone https://github.com/timtoole02/Camelid-Spark && cd Camelid-Spark
./install.sh
BIN=./target/release/camelid
```

## Model files to fetch

| Weights | How to get it |
|---|---|
| **Q8_0** (steps 2 & 4) | `"$BIN" pull` lists the catalog; `"$BIN" pull llama32_3b` downloads a known-good Q8_0 GGUF into `./models`. Any catalog model works. |
| **NVFP4** (steps 1, 3, 4) | The NVFP4 gemma-4-E4B pilot is a **local requantization, not in the catalog**. Copy your `gemma-4-E4B…NVFP4….gguf` onto the Spark. Below, `NVFP4=./models/gemma-4-E4B-it-NVFP4-mm.gguf` — set it to your actual path. |

```bash
Q8=./models/<the-file-pull-downloaded>.gguf
NVFP4=./models/gemma-4-E4B-it-NVFP4-mm.gguf     # your NVFP4 gemma-4 file
```

---

## Step 1 — boots, reports the device, launches a CUDA kernel

Any CUDA run prints the device banner at first GPU init. Use the NVFP4 one-shot with a
tiny budget:

```bash
"$BIN" gemma4-cuda-generate "$NVFP4" --max-tokens 4
```

**Expect** a line on stderr like:

```
[cuda] selected device 0 of 1: "NVIDIA GB10" (compute capability 12.1) | VRAM ... MiB free / ... MiB total
```

That single line is the whole of step 1: GB10, **cc 12.1**, and a kernel then launches to
produce the 4 tokens.

## Step 2 — Q8_0 greedy decode → tokens

```bash
"$BIN" serve --model "$Q8" --addr 127.0.0.1:8181 &
sleep 5
MODEL=$(curl -s http://127.0.0.1:8181/v1/models | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
curl -s http://127.0.0.1:8181/v1/chat/completions \
  -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"The capital of France is\"}],\"temperature\":0,\"max_tokens\":16}"
kill %1
```

`temperature:0` = greedy. **Expect** a JSON completion with non-empty `content`.

## Step 3 — NVFP4 decode → tokens (the gate lift + dp4a GEMV on sm_121)

```bash
"$BIN" gemma4-cuda-generate "$NVFP4" --prompt "The capital of France is" --max-tokens 24
```

**Expect** generated text on stdout. This is the load-bearing proof: NVFP4 now **admits on
Linux** (upstream would refuse with *"NVFP4 is Windows/macOS-only"*), and the `__dp4a`
GEMV kernel — compiled to `compute_61` PTX — forward-JITs and runs on GB10 (`sm_121`).

## Step 4 — serve comes up and answers one curl (OpenAI endpoint)

Same shape as step 2, but point it at the NVFP4 model to prove the served path reaches the
CUDA-resident NVFP4 lane too:

```bash
"$BIN" serve --model "$NVFP4" --addr 127.0.0.1:8181 &
sleep 5
curl -s http://127.0.0.1:8181/v1/models        # 200 + the loaded model id
MODEL=$(curl -s http://127.0.0.1:8181/v1/models | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
curl -s http://127.0.0.1:8181/v1/chat/completions \
  -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"temperature\":0,\"max_tokens\":16}"
kill %1
```

**Expect** a 200 from `/v1/models` and a non-empty completion.

---

## If it doesn't run

- `no kernel image is available for execution on the device` → driver too old to JIT `sm_121`;
  update to a CUDA-13-class driver. Confirm `nvidia-smi --query-gpu=compute_cap --format=csv` = `12.1`.
- **NVRTC / driver symbol error** at first GPU init → the pinned cudarc (CUDA 12.x ABI) is
  missing a symbol on your 13.x driver. Put `/usr/local/cuda-13/compat` on `LD_LIBRARY_PATH`
  (install.sh does this if present). If it persists, this is the one known item that may need a
  re-cut (bump cudarc to a `cuda-130xx` release) — report it and we turn it around.
- NVFP4 refused *"Windows/macOS-only"* → you built upstream Camelid, not this repo.

## What to send back

For each of the four steps: the command, its stdout/stderr (including the `[cuda] selected
device …` banner), and whether tokens came out. If any step crashed, the full error text.
That's enough to call the test pass/fail.
