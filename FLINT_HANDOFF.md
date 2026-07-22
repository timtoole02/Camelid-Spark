# FLINT_HANDOFF.md — DGX Spark tester steps

This is the **acceptance smoke** for the FLINT test build. Purely *"does it run"* — no comparison
to any reference. The primary test is the **GUI**: open the chat UI, download a supported model,
load it, and chat — the same flow as on Windows/macOS, now GPU-accelerated on the Spark.

Build first (see [`README.md`](README.md)):

```bash
git clone https://github.com/timtoole02/Camelid-Spark && cd Camelid-Spark
./install.sh
BIN=./target/release/camelid
```

---

## Part A — the GUI (primary acceptance)

### 1. Start the server; the chat UI opens
```bash
"$BIN" serve
```
- On a desktop session a browser opens **http://127.0.0.1:8181** automatically.
- Headless/SSH: `"$BIN" serve --addr 0.0.0.0:8181 --no-open` then browse to `http://<spark-ip>:8181`,
  or tunnel: `ssh -L 8181:127.0.0.1:8181 <spark>` then `http://127.0.0.1:8181`.
- The server log prints the GPU banner at first GPU init:
  `[cuda] selected device 0 of 1: "NVIDIA GB10" (compute capability 12.1) | VRAM …`

**Expect:** the real Camelid chat UI (sidebar: Models, Chat, Analytics, …), not a blank page.

### 2. Download a supported model in the UI
Open **Models** → **Get models** → pick a curated Q8_0 model (e.g. *Llama 3.2 1B Instruct*, ~1.3 GB)
→ **Download**. Progress shows inline; when it lands it appears under **Supported** and loads.

**Expect:** the model shows **SUPPORTED**, and *"No model loaded"* turns into *"● Loaded — active chat model."*

### 3. Chat
Go to **Chat**, type a message (e.g. *"What is the capital of France? Answer in one sentence."*), send.

**Expect:** a streamed reply with live tok/s diagnostics — running on the GB10 GPU.

> Verified on macOS during development (identical UI/flow, CPU/Metal there): the download → load →
> chat path produced *"The capital of France is Paris."* On the Spark the only difference is the CUDA
> GPU lane doing the decode.

### CLI equivalents (if you prefer the shell)
```bash
"$BIN" pull llama32_1b                              # download into ./models
"$BIN" serve --model ./models/Llama-3.2-1B-Instruct-Q8_0.gguf
# then, greedy:
curl -s http://127.0.0.1:8181/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"Llama 3.2 1B Instruct","messages":[{"role":"user","content":"The capital of France is"}],"temperature":0,"max_tokens":16}'
```

### Stretch the 128 GB — bigger models

The 1B model is just the first light. The catalog now includes larger picks so you can lean on the
Spark's memory (the GPU-resident CUDA lane keeps weights **quantized**, so RAM ≈ the file size):

| Model | Quant | Size | `pull` id |
|---|---|---|---|
| Qwen3 14B | Q8_0 | 14.6 GB | `qwen3_14b_q8_0` |
| Qwen3 32B | Q8_0 | 32.4 GB | `qwen3_32b_q8_0` |
| Llama 3.3 70B Instruct | Q4_K_M | 39.6 GB | `llama33_70b_instruct_q4_k_m` |
| **Llama 3.3 70B Instruct** | **Q8_0** | **~70 GB (2-part, auto-merged)** | `llama33_70b_instruct_q8_0` |

Download them from the **Models** page (they'll show a *fits* badge on the 128 GB box) or `"$BIN" pull
<id>`, then load + chat exactly like Part A. These are **experimental** (runnable, not parity-anchored)
— the point is "does a big model run here on the GPU", not a correctness claim. Any other covered-arch
GGUF works too via the Models-page Hugging Face search or `serve --model <path>`. If a big load ever
errors `cpu_weight_materialization_exceeds_budget`, see the README troubleshooting row.

> Gemma 3 27B was pulled from this list: gemma3 has **no GPU lane** in this engine
> (its serve path is a CPU-only reference runtime), so a 27 GB download would decode
> at unusable speed. The audit that found this is in `SPARK_SPEED.md`.

**The 70B Q8_0 is special:** the Hub only ships it as two gguf-split shards (~40 GB + ~35 GB). Both
the Models-page download and `pull` handle that automatically — two parts fetched under one progress
row, then **merged into a single loadable `.gguf`** (parts deleted afterwards; the merge is proven
byte-exact by a split→merge sha256 round trip on a real model). Budget ~140 GB transient disk. If you
copied shard files onto the box by hand instead: `"$BIN" gguf-merge <any-shard>.gguf` merges them
(siblings are discovered automatically).

**70B Q8_0 load path (`CAMELID_CUDA_HOSTREG`, default ON on this unified box):** projections stream
from disk straight into page-aligned host-registered buffers the GPU reads in place — one steady copy
in the pool (~80.5 GB peak for the 70B build), no VRAM upload, no offload split. Note the Models-page
fit badge for this row reads *unknown* — the advisor deliberately keeps the quant-blind 2× margin;
the load itself is the authority. **Receipts wanted from this box:** (1) the load-time stderr lines
`[cuda] hostreg attrs: …` and `[cuda] hostreg: N zero-copy / M uploaded` (expect N = 561 = 80 layers
× 7 + head for the 70B; any M > 0 means the per-tensor fallback engaged — send the line either way);
(2) greedy decode tok/s; (2b) the **mapped-vs-device A/B on the 70B itself**: the same short greedy
run under `CAMELID_CUDA_HOSTREG=1` vs `CAMELID_CUDA_HOSTREG=upload` (both are streamed 1× loads —
`upload` device-copies instead of registering; token streams must be identical, and the tok/s ratio
IS the mapped-read penalty measurement, decode and TTFT both); (3) an upload-vs-hostreg A/B **on the
14B Q8_0 row** (small enough that
`=0` really takes the historical VRAM-upload path): short greedy runs `CAMELID_CUDA_HOSTREG=0` vs
`=1` — token streams must be identical (they are token-identical across the whole matrix on the 3060
reference card, `qa/evidence-bundles/flint-w3-*`). On the **70B**, a `=0` leg does NOT reach the
upload path — it lands on the CPU lane (the pre-hostreg posture: ~2× can't fit the pool and unified
refuses offload); a short greedy A/B there is still expected token-identical, just minutes-slow.
`CAMELID_CUDA_HOSTREG=0` remains the global kill-switch if anything misbehaves.

---

## Part B — NVFP4 on the GPU (the load-bearing FLINT change)

The NVFP4 gemma-4-E4B pilot is a local requantization, **not** in the download catalog — copy your
`gemma-4-E4B…NVFP4….gguf` onto the Spark, then:

```bash
NVFP4=./models/gemma-4-E4B-it-NVFP4-mm.gguf                       # your file
"$BIN" gemma4-cuda-generate "$NVFP4" --prompt "The capital of France is" --max-tokens 24
```

**Expect:** generated text on stdout. This proves NVFP4 now **admits on Linux** (upstream refuses
with *"NVFP4 is Windows/macOS-only"*) and the `__dp4a` GEMV (`compute_61` PTX) forward-JITs onto
`sm_121`. It also serves through the UI/API: `"$BIN" serve --model "$NVFP4"` then chat as in Part A.
(Use `gemma4-cuda-generate`, **not** `gemma4-generate-gpu` — that one is macOS/Metal-only.)

---

## If it doesn't run

- **Blank / placeholder UI** → the binary was built without the web UI. Re-run `./install.sh` (it runs `npm run build` first).
- **No browser opens** → headless session; open the URL yourself (see step 1).
- `no kernel image is available for execution on the device` → driver too old to JIT `sm_121`; update to a CUDA-13-class driver. `nvidia-smi --query-gpu=compute_cap --format=csv` must read `12.1`.
- **NVRTC / driver symbol error** → cudarc (CUDA 12.x ABI) missing a symbol on your 13.x driver. Put `/usr/local/cuda-13/compat` on `LD_LIBRARY_PATH` (install.sh does this if present); if it persists, bump cudarc to a `cuda-130xx` release — the one item that may need a second turn.
- NVFP4 refused *"Windows/macOS-only"* → you built upstream Camelid, not this repo.

## What to send back

Whether the UI opened, the download+load worked, and chat produced a reply (a screenshot is ideal),
the `[cuda] selected device …` banner, and — for Part B — the NVFP4 command's output. If anything
crashed, the full error text. That's enough to call the test pass/fail.
