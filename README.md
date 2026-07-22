# Camelid-Spark (FLINT test build)

> **This is a test build.** Its only goal is to show that [Camelid](https://github.com/timtoole02/Camelid)
> — a Rust-native local GGUF inference engine — **installs, boots on the GPU, and generates
> tokens** on an NVIDIA **DGX Spark** (GB10 / `sm_121`), for both Q8_0 and NVFP4 weights.
> It makes **no** correctness, quality, or performance claim. Acceptance = it runs.

Built from Camelid `main` (the full upstream README is preserved as
[`README_UPSTREAM.md`](README_UPSTREAM.md)). The only source change vs. upstream is lifting
the NVFP4 platform gate to admit Linux (see [`FLINT_CONDUCTOR.md`](FLINT_CONDUCTOR.md)); the
CUDA kernels are unchanged — they compile to virtual `compute_61` PTX and the driver
forward-JITs them onto Blackwell.

## 60-second quickstart

```bash
git clone https://github.com/timtoole02/Camelid-Spark
cd Camelid-Spark
./install.sh                      # preflight → build (CUDA) → binary/GPU smoke

# Fetch a known-good Q8_0 model and serve it:
./target/release/camelid pull                       # list the catalog
./target/release/camelid pull llama32_3b            # download into ./models
./target/release/camelid serve --model ./models/<file>.gguf --addr 127.0.0.1:8181

# In another shell — one curl on the OpenAI-compatible endpoint:
curl -s http://127.0.0.1:8181/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"The capital of France is"}],"temperature":0,"max_tokens":16}'
```

The full four-step acceptance smoke (Q8_0 **and** NVFP4, one-shot **and** serve) with
exact commands and the model files to fetch is in **[`FLINT_HANDOFF.md`](FLINT_HANDOFF.md)**.

## Prerequisites (the target box)

| | |
|---|---|
| GPU | NVIDIA **GB10** Blackwell, compute capability **12.1** (`sm_121`) |
| CPU | aarch64 **Grace** (ARMv9.2) |
| OS | DGX OS 7.x (Ubuntu 24.04 aarch64) |
| CUDA | driver of the CUDA **13.x** class (the build itself needs no toolkit) |
| Rust | installed via **rustup** — the channel is pinned by `rust-toolchain.toml` (1.95.0) |
| APT | `git cmake build-essential pkg-config libssl-dev libcurl4-openssl-dev` |

`install.sh` checks all of these and prints a fix hint for whatever is missing.
To compile on a non-Spark box (no GB10), run `FLINT_ALLOW_NON_SPARK=1 ./install.sh`.

> **Why no CUDA toolkit to build?** cudarc is pinned with `fallback-dynamic-loading`:
> it `dlopen()`s `libcuda` at runtime, so `cargo build --features cuda` needs no `nvcc`.
> The kernels are JIT-compiled by the driver at load time.

## Fetching models

- **Q8_0:** `camelid pull` (with no argument) lists the catalog of known-good Q8_0 GGUFs;
  `camelid pull <id>` downloads one into `./models`.
- **NVFP4:** the NVFP4 gemma-4 pilot is a local requantization and is **not** in the pull
  catalog — copy your `gemma-4-E4B …NVFP4….gguf` onto the Spark and pass its path directly.
  See [`FLINT_HANDOFF.md`](FLINT_HANDOFF.md).

## Troubleshooting

| Symptom | Cause & fix |
|---|---|
| `no kernel image is available for execution on the device` | The driver couldn't JIT the PTX for this GPU — usually a driver too old to know `sm_121`, or the wrong GPU selected. Update to a CUDA-13-class driver; confirm `nvidia-smi --query-gpu=compute_cap --format=csv` prints `12.1`. |
| NVRTC / driver **symbol** error at startup (e.g. a missing `cu*`/NVRTC symbol) | The pinned cudarc targets the CUDA 12.x ABI. It usually binds fine on a 13.x driver, but if a symbol is missing: (a) ensure `/usr/local/cuda-13/compat` is on `LD_LIBRARY_PATH` (install.sh does this if present), or (b) this is the one item that may need a re-cut — bump cudarc to a release exposing a `cuda-130xx` feature. |
| Build fails on `openssl-sys` / `curl-sys` | `sudo apt-get install -y pkg-config libssl-dev libcurl4-openssl-dev`. |
| NVFP4 model refused with *"Windows/macOS-only"* | You're on an upstream Camelid build, not this one. This repo lifts that gate for Linux. |

## CI

[`.github/workflows/flint-ci.yml`](.github/workflows/flint-ci.yml) compiles the whole
tree for `aarch64-unknown-linux-gnu` **with `--features cuda`** on an arm64 Ubuntu
runner (no GPU needed, thanks to the runtime-loading note above). A green check means
the Spark build compiles; it does not run the token smoke — that's the Spark's job.
