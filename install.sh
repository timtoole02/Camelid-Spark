#!/usr/bin/env bash
# FLINT install.sh — build Camelid for an NVIDIA DGX Spark (GB10 / sm_121).
#
# This is a TEST BUILD. Its whole job is to compile the `camelid` binary with the
# CUDA backend on aarch64 Linux and prove the GPU is reachable. It does NOT make
# any correctness/quality claim. The four token-producing smoke steps (which need
# model files) live in FLINT_HANDOFF.md.
#
#   Usage:   ./install.sh
#   Rebuild: ./install.sh              # idempotent — safe to re-run
#   Non-Spark build (skip GPU preflight): FLINT_ALLOW_NON_SPARK=1 ./install.sh
set -euo pipefail

# ---- pretty output ---------------------------------------------------------
if [ -t 1 ]; then RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; BLD=$'\033[1m'; RST=$'\033[0m'
else RED=''; GRN=''; YLW=''; BLD=''; RST=''; fi
ok()   { printf '%s✓%s %s\n' "$GRN" "$RST" "$*"; }
warn() { printf '%s!%s %s\n' "$YLW" "$RST" "$*"; }
die()  { printf '\n%s✗ %s%s\n' "$RED" "$*" "$RST" >&2; exit 1; }
step() { printf '\n%s== %s ==%s\n' "$BLD" "$*" "$RST"; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# ---- G2 preflight (each check names its fix) -------------------------------
step "Preflight"

# aarch64 (Grace CPU)
arch="$(uname -m)"
[ "$arch" = "aarch64" ] || die "This build targets aarch64 (Grace); found '$arch'. Run on the DGX Spark."
ok "CPU arch: aarch64"

# GPU: GB10, compute capability 12.1 (sm_121)
if [ "${FLINT_ALLOW_NON_SPARK:-0}" = "1" ]; then
  warn "FLINT_ALLOW_NON_SPARK=1 — skipping GPU preflight (build-only)"
elif command -v nvidia-smi >/dev/null 2>&1; then
  cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')"
  name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/^ *//')"
  drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')"
  if [ "$cc" = "12.1" ]; then
    ok "GPU: ${name:-unknown} (compute capability $cc = sm_121), driver $drv"
  else
    die "GPU compute capability is '$cc', expected 12.1 (GB10/sm_121). Not a DGX Spark, or wrong GPU selected. To build anyway: FLINT_ALLOW_NON_SPARK=1 ./install.sh"
  fi
else
  die "nvidia-smi not found — no NVIDIA driver. Install the DGX OS driver stack, or FLINT_ALLOW_NON_SPARK=1 ./install.sh to build without a GPU."
fi

# CUDA toolkit 13.x (nvcc must know sm_121)
if command -v nvcc >/dev/null 2>&1; then
  nvcc_rel="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
  case "$nvcc_rel" in
    13.*) ok "CUDA toolkit: nvcc release $nvcc_rel" ;;
    "")   warn "Could not parse nvcc version — proceeding (cudarc uses runtime dynamic loading, so the toolkit is not required to BUILD)." ;;
    *)    warn "nvcc release $nvcc_rel is not 13.x. sm_121 support wants CUDA 13.x — update the toolkit if the runtime smoke fails with 'no kernel image'." ;;
  esac
else
  warn "nvcc not found. The build does NOT need it (cudarc dynamically loads libcuda at runtime), but the runtime needs a CUDA-13-class driver. See FLINT_HANDOFF.md."
fi

# cmake >= 3.31
if command -v cmake >/dev/null 2>&1; then
  cmv="$(cmake --version | sed -n 's/cmake version \([0-9.]*\).*/\1/p' | head -1)"
  ok "cmake: $cmv"
else
  warn "cmake not found — will attempt to apt-install it below."
fi

# ---- APT dependencies (idempotent) ----------------------------------------
step "System dependencies"
APT_PKGS="git cmake build-essential pkg-config libssl-dev libcurl4-openssl-dev"
if command -v apt-get >/dev/null 2>&1; then
  SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  printf 'Installing: %s\n' "$APT_PKGS"
  $SUDO apt-get update -y
  $SUDO apt-get install -y $APT_PKGS
  ok "apt dependencies present"
else
  warn "apt-get not found — ensure these are installed: $APT_PKGS"
fi

# ---- Rust toolchain (rust-toolchain.toml pins the channel) ----------------
step "Rust toolchain"
if command -v rustup >/dev/null 2>&1; then
  # rustup honors rust-toolchain.toml (channel 1.95.0) automatically on first cargo call.
  rustup show active-toolchain 2>/dev/null || rustup toolchain install 1.95.0
  ok "rustup present — toolchain pinned by rust-toolchain.toml ($(sed -n 's/.*channel = "\(.*\)".*/\1/p' rust-toolchain.toml))"
elif command -v cargo >/dev/null 2>&1; then
  warn "cargo present but no rustup — the pinned toolchain ($(sed -n 's/.*channel = "\(.*\)".*/\1/p' rust-toolchain.toml)) may not be honored. Prefer rustup: https://rustup.rs"
else
  die "No Rust toolchain. Install rustup (https://rustup.rs), then re-run ./install.sh"
fi

# ---- CUDA-13 runtime link path (the one real unknown) ---------------------
# cudarc is pinned to the CUDA 12.x driver ABI with fallback-dynamic-loading, so
# the build needs no CUDA. At RUNTIME the driver JIT-compiles compute_61 PTX to
# sm_121. If your driver is CUDA-13, the compat libs help older-ABI clients bind.
if [ -d /usr/local/cuda-13/compat ]; then
  export LD_LIBRARY_PATH="/usr/local/cuda-13/compat:${LD_LIBRARY_PATH:-}"
  ok "LD_LIBRARY_PATH includes /usr/local/cuda-13/compat"
fi

# ---- Build (CUDA backend on; server binary only, no Tauri desktop) ---------
step "Build: cargo build --release --bin camelid --features cuda"
# NOTE: build errors are surfaced, never swallowed — the first real
# aarch64 + CUDA-13 + sm_121 build is the point of this test.
cargo build --release --bin camelid --features cuda
BIN="$REPO_DIR/target/release/camelid"
[ -x "$BIN" ] || die "Build reported success but $BIN is missing."
ok "Built: $BIN"

# ---- Post-build smoke (binary runs + GPU visible; NO models needed) -------
step "Smoke: binary + GPU visibility"
"$BIN" --help >/dev/null 2>&1 && ok "camelid binary runs"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader | sed 's/^/    GPU: /'
fi
if command -v nvcc >/dev/null 2>&1; then nvcc --version | sed -n 's/.*release/    nvcc release/p'; fi

# ---- Summary ---------------------------------------------------------------
printf '\n%s%s========================================%s\n' "$BLD" "$GRN" "$RST"
ok "FLINT build complete."
printf '\nNext: run the four token-producing smoke steps in %sFLINT_HANDOFF.md%s\n' "$BLD" "$RST"
printf '  1. device facts + a CUDA kernel launch\n'
printf '  2. Q8_0 greedy decode  →  tokens\n'
printf '  3. NVFP4 decode        →  tokens (gate lift + dp4a GEMV on sm_121)\n'
printf '  4. serve + one curl on the OpenAI-compatible endpoint\n\n'
printf 'Quick start:  %scamelid pull%s to fetch a Q8_0 model, then %scamelid serve --model ./models/<file>.gguf%s\n' "$BLD" "$RST" "$BLD" "$RST"
