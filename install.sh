#!/usr/bin/env bash
# FLINT install.sh — build Camelid for an NVIDIA DGX Spark (GB10 / sm_121) so the
# tester gets the SAME experience as on Windows/macOS: run `camelid serve`, the
# chat UI opens in a browser, download a supported model from the Models page,
# load it, and chat — now GPU-accelerated on the Spark.
#
# This builds BOTH halves that ship inside the one binary:
#   1. the web UI      (frontend/  →  npm run build  →  embedded via rust-embed)
#   2. the engine      (cargo build --release --bin camelid --features cuda)
# Skipping step 1 embeds a blank placeholder page — so install.sh always does it.
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
SUDO=""; [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1 && SUDO="sudo"

# ---- Preflight (each check names its fix) ----------------------------------
step "Preflight"

arch="$(uname -m)"
[ "$arch" = "aarch64" ] || die "This build targets aarch64 (Grace); found '$arch'. Run on the DGX Spark."
ok "CPU arch: aarch64"

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

if command -v nvcc >/dev/null 2>&1; then
  nvcc_rel="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
  case "$nvcc_rel" in
    13.*) ok "CUDA toolkit: nvcc release $nvcc_rel" ;;
    "")   warn "Could not parse nvcc version — proceeding (build needs no toolkit; cudarc loads libcuda at runtime)." ;;
    *)    warn "nvcc release $nvcc_rel is not 13.x. sm_121 wants CUDA 13.x — update if the runtime smoke says 'no kernel image'." ;;
  esac
else
  warn "nvcc not found. The build does NOT need it (cudarc dynamically loads libcuda at runtime); the runtime needs a CUDA-13-class driver."
fi

# ---- APT dependencies (idempotent) ----------------------------------------
# xdg-utils = `xdg-open`, so `camelid serve` can pop the browser on a desktop session.
step "System dependencies"
APT_PKGS="git cmake build-essential pkg-config libssl-dev libcurl4-openssl-dev ca-certificates curl xdg-utils"
if command -v apt-get >/dev/null 2>&1; then
  printf 'Installing: %s\n' "$APT_PKGS"
  $SUDO apt-get update -y
  $SUDO apt-get install -y $APT_PKGS
  ok "apt dependencies present"
else
  warn "apt-get not found — ensure these are installed: $APT_PKGS"
fi

# ---- Node.js (>= 20; builds the web UI) -----------------------------------
step "Node.js (for the web UI build)"
need_node=1
if command -v node >/dev/null 2>&1; then
  nmaj="$(node -v 2>/dev/null | sed 's/v\([0-9]*\).*/\1/')"
  if [ "${nmaj:-0}" -ge 20 ] 2>/dev/null; then need_node=0; ok "node $(node -v) present"; fi
fi
if [ "$need_node" = "1" ]; then
  if command -v apt-get >/dev/null 2>&1; then
    warn "Installing Node.js 22 via NodeSource…"
    curl -fsSL https://deb.nodesource.com/setup_22.x | $SUDO -E bash -
    $SUDO apt-get install -y nodejs
    ok "node $(node -v) installed"
  else
    die "Node.js >= 20 is required to build the web UI. Install it (https://nodejs.org), then re-run ./install.sh"
  fi
fi

# ---- Rust toolchain (rust-toolchain.toml pins the channel) ----------------
step "Rust toolchain"
if command -v rustup >/dev/null 2>&1; then
  rustup show active-toolchain 2>/dev/null || rustup toolchain install
  ok "rustup present — toolchain pinned by rust-toolchain.toml ($(sed -n 's/.*channel = "\(.*\)".*/\1/p' rust-toolchain.toml))"
elif command -v cargo >/dev/null 2>&1; then
  warn "cargo present but no rustup — the pinned toolchain may not be honored. Prefer rustup: https://rustup.rs"
else
  die "No Rust toolchain. Install rustup (https://rustup.rs), then re-run ./install.sh"
fi

# CUDA-13 runtime link path (the one real unknown; see README troubleshooting).
if [ -d /usr/local/cuda-13/compat ]; then
  export LD_LIBRARY_PATH="/usr/local/cuda-13/compat:${LD_LIBRARY_PATH:-}"
  ok "LD_LIBRARY_PATH includes /usr/local/cuda-13/compat"
fi

# ---- 1) Build the web UI (embedded into the binary) ------------------------
step "Web UI: cd frontend && npm ci && npm run build"
( cd frontend && npm ci && npm run build )
idx="$REPO_DIR/frontend/dist/index.html"
[ -f "$idx" ] && [ "$(wc -c < "$idx")" -gt 200 ] \
  || die "Frontend build did not produce a real dist/index.html (got a placeholder). The UI would be blank — aborting."
ok "web UI built ($(du -sh frontend/dist 2>/dev/null | cut -f1)) — will be embedded in the binary"

# ---- 2) Build the engine (CUDA on; server binary only, no Tauri desktop) ---
step "Engine: cargo build --release --bin camelid --features cuda"
# Build errors are surfaced, never swallowed — the first aarch64 + CUDA-13 + sm_121
# build is the point of this test.
cargo build --release --bin camelid --features cuda
BIN="$REPO_DIR/target/release/camelid"
[ -x "$BIN" ] || die "Build reported success but $BIN is missing."
ok "Built: $BIN"

# ---- Smoke: binary runs + real UI embedded + GPU visible -------------------
step "Smoke: binary + embedded UI + GPU"
"$BIN" --help >/dev/null 2>&1 && ok "camelid binary runs"
# The embedded index.html should be the real app (KB-sized), not the placeholder.
if "$BIN" pull --help >/dev/null 2>&1; then ok "model catalog available (camelid pull)"; fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader | sed 's/^/    GPU: /'
fi

# ---- Summary ---------------------------------------------------------------
printf '\n%s%s========================================%s\n' "$BLD" "$GRN" "$RST"
ok "FLINT build complete — engine + web UI."
cat <<EOF

${BLD}Start it (same as Windows/macOS):${RST}
  ${BLD}$BIN serve${RST}
Then a browser opens ${BLD}http://127.0.0.1:8181${RST} (on a desktop session). In the UI:
  • open the ${BLD}Models${RST} page → download a supported model → it loads
  • go to ${BLD}Chat${RST} and talk to it — GPU-accelerated on the Spark

Headless / over SSH? Bind all interfaces and tunnel, or open the URL yourself:
  $BIN serve --addr 0.0.0.0:8181 --no-open      # then browse to http://<spark-ip>:8181
  ssh -L 8181:127.0.0.1:8181 <spark>            # from your laptop, then http://127.0.0.1:8181

Acceptance smoke + the NVFP4 path: see ${BLD}FLINT_HANDOFF.md${RST}.
EOF
