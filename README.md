# Camelid-Spark (FLINT test build)

> **Test build** to evaluate that [Camelid](https://github.com/timtoole02/Camelid) — a
> Rust-native local GGUF inference engine — runs on an NVIDIA **DGX Spark** (GB10 / `sm_121`).
> You get the **same experience as on Windows/macOS**: run `camelid serve`, the **chat UI opens
> in your browser**, you **download a supported model** from the Models page, load it, and
> **chat** — now GPU-accelerated on the Spark. No correctness/quality claim; acceptance = it runs.

Built from Camelid `main` (upstream README preserved as [`README_UPSTREAM.md`](README_UPSTREAM.md)).
The one source change vs. upstream is lifting the NVFP4 platform gate to admit Linux
([`FLINT_CONDUCTOR.md`](FLINT_CONDUCTOR.md)); the CUDA kernels are unchanged (`compute_61` PTX
forward-JITs onto Blackwell). The web UI, model downloader, and chat are the **same** ones that
ship on every platform — they're built and embedded into the binary by `install.sh`.

## Quickstart — the GUI, like Windows/macOS

```bash
git clone https://github.com/timtoole02/Camelid-Spark
cd Camelid-Spark
./install.sh            # builds the web UI + the CUDA engine into one binary
./target/release/camelid serve
```

`serve` opens **http://127.0.0.1:8181** in your browser (on a desktop session). Then, entirely in the UI:

1. **Models** page → pick a supported model under **Get models** → **Download** (progress shows inline).
2. The model appears under **Supported** and loads — *"No model loaded"* turns green.
3. **Chat** → type a message → it streams back, running on the GPU.

That's the whole test: open the UI, download a supported model, load it, chat.

### Headless Spark / over SSH

No desktop on the Spark? Bind all interfaces (or tunnel) and open the URL yourself:

```bash
./target/release/camelid serve --addr 0.0.0.0:8181 --no-open   # then http://<spark-ip>:8181
# or, safer, keep it local and tunnel from your laptop:
ssh -L 8181:127.0.0.1:8181 <spark>                             # then http://127.0.0.1:8181
```

> ⚠️ `--addr 0.0.0.0` exposes the UI/API to everything that can reach the host. Only do that on a
> trusted network. The SSH tunnel keeps it bound to localhost.

## Prerequisites (the target box)

| | |
|---|---|
| GPU | NVIDIA **GB10** Blackwell, compute capability **12.1** (`sm_121`) |
| CPU | aarch64 **Grace** (ARMv9.2) |
| OS | DGX OS 7.x (Ubuntu 24.04 aarch64) |
| CUDA | driver of the CUDA **13.x** class (the build itself needs no toolkit) |
| Rust | via **rustup** — channel pinned by `rust-toolchain.toml` (1.95.0) |
| Node | **≥ 20** (builds the web UI) — `install.sh` installs Node 22 if missing |
| APT | `git cmake build-essential pkg-config libssl-dev libcurl4-openssl-dev xdg-utils` |

`install.sh` checks all of these and prints a fix hint for whatever is missing. To compile on a
non-Spark box (no GB10), run `FLINT_ALLOW_NON_SPARK=1 ./install.sh`.

> **Two things get built into the one binary:** the **web UI** (`frontend/` → `npm run build` →
> embedded via rust-embed) and the **engine** (`cargo build --release --features cuda`). If the UI
> isn't built first, the binary embeds a blank placeholder page — so `install.sh` always builds it.
> **Why no CUDA toolkit to build?** cudarc uses `fallback-dynamic-loading` — it `dlopen()`s
> `libcuda` at runtime, so the build needs no `nvcc`; the driver JIT-compiles the kernels at load.

## Models

- **In the UI (recommended):** the **Models** page downloads curated, known-good **Q8_0** models
  and shows live support status from `/api/capabilities`. This is the primary test path.
- **CLI equivalent:** `camelid pull` lists the catalog; `camelid pull llama32_1b` downloads one.
- **NVFP4:** the NVFP4 gemma-4-E4B pilot is a local requantization, **not** in the catalog — copy
  your GGUF onto the Spark and use it by path. See [`FLINT_HANDOFF.md`](FLINT_HANDOFF.md).

## Troubleshooting

| Symptom | Cause & fix |
|---|---|
| Browser opens a **blank / placeholder** page | The UI wasn't built before the binary. Re-run `./install.sh` (it runs `npm run build` first). |
| `serve` prints the URL but no browser opens | Headless/SSH session — `xdg-open` has no display. Open the URL yourself (see *Headless* above). |
| `no kernel image is available for execution on the device` | Driver too old to JIT `sm_121`, or wrong GPU. Update to a CUDA-13-class driver; `nvidia-smi --query-gpu=compute_cap --format=csv` must print `12.1`. |
| NVRTC / driver **symbol** error at first GPU use | Pinned cudarc targets the CUDA 12.x ABI. Put `/usr/local/cuda-13/compat` on `LD_LIBRARY_PATH` (install.sh does this if present); if it persists, bump cudarc to a `cuda-130xx` release. |
| Build fails on `openssl-sys` / `curl-sys` | `sudo apt-get install -y pkg-config libssl-dev libcurl4-openssl-dev`. |
| NVFP4 model refused *"Windows/macOS-only"* | You built upstream Camelid, not this repo — this one lifts that gate for Linux. |

## CI

[`.github/workflows/flint-ci.yml`](.github/workflows/flint-ci.yml) builds the **web UI** and then
the release binary **with `--features cuda`** for `aarch64-unknown-linux-gnu` on an arm64 Ubuntu
runner (no GPU needed to compile). Green = the full product (UI + CUDA engine) builds for the Spark;
the token/chat smokes run on the real hardware — see [`FLINT_HANDOFF.md`](FLINT_HANDOFF.md).
