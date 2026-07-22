#!/bin/bash
# Build the macOS `camelid` release binary with a FRESH embedded web UI.
#
# Why this script exists: `frontend/dist` is gitignored and is baked into the
# binary at COMPILE time via rust-embed (`src/web_ui.rs`, `#[folder="frontend/dist"]`).
# A plain `cargo build` does NOT rebuild the frontend and does NOT recompile on a
# gitignored asset change — so it can silently ship a STALE web UI (e.g. a model
# download with no progress bar). Always build the Mac release through this script.
#
# Usage:  scripts/build-mac.sh
#   Respects CARGO_TARGET_DIR if set (e.g. an external-disk target dir).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[1/3] frontend: vite build -> frontend/dist"
npm --prefix frontend run build

echo "[2/3] force rust-embed to pick up the fresh dist (cargo won't on a gitignored change)"
touch src/web_ui.rs

echo "[3/3] release binary (embeds the fresh frontend/dist)"
cargo build --release --bin camelid

BIN="${CARGO_TARGET_DIR:-target}/release/camelid"
echo "built: $BIN"
"$BIN" --version
