# FLINT_CONDUCTOR.md — Camelid → Camelid-Spark (TEST BUILD)

**Goal:** get Camelid building and running on an NVIDIA DGX Spark (GB10 / sm_121) with GPU + NVFP4, easy to install, for an external tester. This is a **test build**. No parity, no receipts, no oracle, no llama.cpp comparison of any kind. Acceptance = it installs, boots on the GPU, and generates tokens (Q8_0 and NVFP4). That's the whole bar.

**Source:** `github.com/timtoole02/Camelid` → **target:** `github.com/timtoole02/Camelid-Spark` (exists, `main` initialized).

## Step 0 — seed the repo [AGENT, do this first]

Fork/copy the Camelid tree into Camelid-Spark, commit this conductor at the repo root, push. Everything below lands as commits on Camelid-Spark. Do NOT modify the upstream Camelid repo.

## Why this is easy (confirm in G0, don't re-derive)

Three things that would normally make a Spark port hard are already done in the tree:

1. **Linux CUDA is already wired, just opt-in.** `Cargo.toml` has the `cfg(all(not(macos), not(windows)))` `cudarc` dep and `cuda = ["dep:cudarc"]`. `build.rs` auto-enables `cuda` on Windows; on Linux it's `--features cuda`.
2. **Kernels are architecture-agnostic.** `src/cuda.rs` / `src/cuda_resident.rs` compile `arch=compute_61` PTX via NVRTC and let the driver JIT to the present device. compute_61 PTX forward-JITs to sm_121; `__dp4a` exists on Blackwell → the NVFP4 dp4a GEMV runs on GB10 with no gencode/sm change. Leave all kernel compile options exactly as-is (incl. `--fmad=false`) — this is a test build, don't tune kernels and introduce new variables.
3. **aarch64 CPU path already exists** (from the Mac/Pi work): `aarch64-i8mm`/`dotprod`/`scalar` ISA paths, x86 intrinsics fall through to scalar on non-x86, GAIT detects NEON. Grace CPU (ARMv9.2, i8mm) compiles and runs.

So FLINT is: enable CUDA on Linux + lift the NVFP4 platform gate + install script + README.

## Target box (verify on the machine in G4)

* **GPU:** GB10 Blackwell, compute capability 12.1 = **sm_121** (NOT sm_100/B200, NOT sm_120/RTX50)
* **CPU:** aarch64 Grace (ARMv9.2 + i8mm/dotprod/SVE2)
* **Memory:** 128 GB unified LPDDR5X, 273 GB/s (bandwidth-bound — NVFP4's small footprint is the win here, not FLOPs)
* **OS:** DGX OS 7.x (Ubuntu 24.04-based, aarch64)
* **Toolchain (known-good):** CUDA 13.0.2, driver 580.126.09, kernel 6.17, cmake 3.31+
* **Rust:** pinned by `rust-toolchain.toml` (1.89); target `aarch64-unknown-linux-gnu`
* **APT deps:** `git cmake build-essential libssl-dev libcurl4-openssl-dev`

## G0 — RECON [AGENT, blind]

Confirm the port surface with file:line. Minimum:

* The two NVFP4 platform gates (runtime `cfg!(target_os)` checks, NOT `#[cfg]` walls): `src/runnable/admit.rs` (~L312–L342) and `src/gemma4_runtime.rs::nvfp4_windows_only_check` (~L1075). Current message: `"NVFP4 is Windows/macOS-only in this release; see SUPPORT_MATRIX"`.
* The tests asserting off-Windows/macOS refusal (flip in G1): `gemma4_nvfp4_pilot_refuses_off_windows_with_platform_gate`, `gemma4_nvfp4_with_bf16_refuses_off_windows_platform_gate`, the admit.rs refusal test (~L682), and `src/runnable/smoke.rs::gemma4_nvfp4_smoke_refusal_is_not_yet_anchored`.
* CUDA compile sites use `arch=compute_61` (`src/cuda.rs` ~L383, `src/cuda_resident.rs` ~L20/L3145). Confirm no hardcoded sm_86 anywhere.
* `camelid-desktop` (Tauri) is a `-p`-only workspace member with `default-members = ["."]` — confirm the default build never pulls it. Web UI (`frontend/dist`) embeds via a placeholder in `build.rs`, so no Node needed to build.
* **Gate:** `cargo check --target aarch64-unknown-linux-gnu --features cuda` is clean (`rustup target add aarch64-unknown-linux-gnu` first; `check` needs no CUDA installed because cudarc uses `fallback-dynamic-loading`). This is the only compile signal without the box — get it green in G1.

## G1 — PORT [AGENT, blind]

1. **Lift the NVFP4 platform gate to include Linux.** In BOTH gate sites, add `target_os = "linux"` to the allowed set (simple lift — no toggle, no receipt; this is a test build). Update the tests in G0 to assert Linux admits instead of refusing. Keep Windows/macOS behavior byte-identical.
   * Keep the existing NVFP4 NaN-sentinel scan (`crate::tensor::nvfp4_find_nan_scale`, refusing UE4M3 `0x7F`/`0xFF`) wired on the CUDA NVFP4 lane — it's a fail-closed data-hygiene guard, not a parity check, so a corrupt-scale file errors cleanly instead of emitting garbage.
2. **Default-on CUDA for the Spark.** Extend `build.rs` so `target_os = "linux"` + `aarch64` enables the `cuda` cfg (bare `cargo build --release` gets the GPU backend). Minimal alternative if you'd rather not touch build.rs: have `install.sh` always pass `--features cuda`. Either is fine — pick one.
3. **Keep `camelid-desktop` out of the default build.** FLINT builds `--bin camelid` only.
4. **Surface device facts at boot** (name, cc major.minor, VMM, free/total unified memory) — already read via `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_*`; just print them so the tester sees "GB10, cc 12.1" on startup.

* **Gate:** `cargo check --target aarch64-unknown-linux-gnu --features cuda` stays clean.

## G2 — INSTALL [AGENT authors, TESTER runs]

`install.sh` — one command, idempotent, loud on failure. Preflight (each with a fix hint):

* `nvidia-smi --query-gpu=compute_cap --format=csv` == 12.1 (else: not a Spark / wrong GPU)
* `nvcc --version` CUDA 13.x (else: update toolkit — older nvcc doesn't know sm_121)
* `uname -m` == aarch64; `cmake --version` ≥ 3.31
* `apt install` the deps above; `rustup` present and honoring `rust-toolchain.toml`

Then: `cargo build --release` (CUDA on per G1) → run the smoke check (G4) → print a green/red summary. Do not swallow build errors — surfacing the first real aarch64+CUDA13+sm_121 build is the point of the test.

**CUDA-13 runtime note (the one real unknown):** cudarc is pinned `0.19` / `cuda-12090` with `fallback-dynamic-loading`. Driver API is backward-compatible and compute_61 PTX JIT is universal, so it most likely runs on a CUDA-13 driver as-is. Have the script export `LD_LIBRARY_PATH="/usr/local/cuda-13/compat:$LD_LIBRARY_PATH"` and print the driver/NVRTC version at startup. If the tester hits an NVRTC/driver symbol error: bump cudarc to a release exposing a `cuda-130xx` feature and re-cut. This is the single item most likely to need a second turn — everything else should just work.

## G3 — README [AGENT]

`README.md` for Camelid-Spark: 60-second quickstart (`git clone` → `./install.sh` → `curl` the OpenAI-compatible endpoint), the prereq list, a model-fetch step, and a troubleshooting section keyed to the real failure modes:

* `no kernel image is available for execution on the device` → wrong CUDA arch / driver
* NVRTC/driver symbol error → CUDA-13 cudarc bump (G2 note)

Terse and copy-pasteable. State plainly at the top that this is a test build for evaluating that Camelid runs on the Spark — no correctness/quality claims.

## G4 — SMOKE [AGENT authors, TESTER runs]

The acceptance test — purely "does it run", no comparison to anything:

1. Boots, reports device GB10, cc 12.1, a CUDA kernel launches.
2. Greedy-decodes a canned prompt (`"The capital of France is"`) on a Q8_0 model → tokens out.
3. Same on an NVFP4 model → tokens out (proves the gate lift + dp4a GEMV on sm_121).
4. `serve` comes up and answers one `curl` on the OpenAI-compatible endpoint.

If all four produce output without crashing, the test build passes. Ship `FLINT_HANDOFF.md` with the exact tester commands + which model files to fetch.

## Done when

1. `Camelid-Spark` has the fork + G1 changes; `cargo check --target aarch64-unknown-linux-gnu --features cuda` is clean. **[AGENT]**
2. `install.sh`, `README.md`, `FLINT_HANDOFF.md` committed. **[AGENT]**
3. Tester runs `./install.sh` on the Spark and the four G4 smoke steps produce tokens. **[TESTER]**

No receipts. No parity. No oracle. Just a working test build.
