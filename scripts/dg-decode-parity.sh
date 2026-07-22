#!/usr/bin/env bash
# DiffusionGemma lane Phase 3 gate runner: single denoise step parity of
# camelid vs pinned llama.cpp on the tracked model — host RNG (canvas init +
# step draws), the unified zero-SC [prompt | canvas] forward's canvas
# logits, and one Entropy-Bound sampler step (credit: llama.cpp / ggml
# authors — the reference graph, sampler, and CPU kernels are theirs,
# compiled from the pinned checkout; the RNG distribution semantics are the
# LLVM libc++ authors').
#
# KERNEL CONTRACT (Phase 3 finding): the default macOS build registers the
# ggml BLAS (Accelerate) backend, whose device claims every contiguous
# mul_mat with ne0/ne1/ne10 >= 32 — at the unified forward's 273 rows that
# routes all dense projections, KQV, the router, and the lm_head through
# dequant-to-f32 + cblas_sgemm (closed-source blocking, macOS-version
# dependent). Phase 2's prompt (17 rows) never crossed the threshold, so its
# sealed result is untouched. This gate therefore runs the reference from a
# BLAS-free build of the SAME pinned commit (build-cpu: GGML_BLAS=OFF,
# GGML_METAL=OFF, GGML_ACCELERATE=ON so the CPU backend's vDSP diversions
# stay), keeping one named kernel contract across phases: generic CPU
# vec_dot, no repack, no BLAS.
#
# Env:
#   DG_GGUF      tracked model file (required)
#   DG_PIN       pinned llama.cpp checkout (required)
#   DG_PIN_BUILD reference build dir (default: ${DG_PIN}/build-cpu)
#   DG_PROMPT    prompt id file (default: the Phase 2 bundle's prompt-ids.i32)
#   DG_SEED      EB sampler seed (default 0)
#   DG_BUNDLE    bundle dir (default: target/dg-decode-parity-<utc>/)
set -euo pipefail

DG_GGUF="${DG_GGUF:?set DG_GGUF to the tracked GGUF path}"
DG_PIN="${DG_PIN:?set DG_PIN to the pinned llama.cpp checkout}"
DG_PIN_BUILD="${DG_PIN_BUILD:-${DG_PIN}/build-cpu}"
DG_PROMPT="${DG_PROMPT:-target/dg-encoder-parity-20260611T223204Z/prompt-ids.i32}"
DG_SEED="${DG_SEED:-0}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
DG_BUNDLE="${DG_BUNDLE:-target/dg-decode-parity-${TS}}"
mkdir -p "${DG_BUNDLE}/ref" "${DG_BUNDLE}/rng" "${DG_BUNDLE}/eb"

if [ ! -f "${DG_PIN_BUILD}/bin/libllama.dylib" ] && [ ! -f "${DG_PIN_BUILD}/bin/libllama.so" ]; then
    cmake -S "${DG_PIN}" -B "${DG_PIN_BUILD}" -DGGML_BLAS=OFF -DGGML_METAL=OFF \
        -DGGML_ACCELERATE=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
    cmake --build "${DG_PIN_BUILD}" --target llama -j10
fi

ENCDUMP="${DG_PIN_BUILD}/bin/dg-encoder-dump"
RNGDUMP="${DG_PIN_BUILD}/bin/dg-rng-dump"
EBSTEP="${DG_PIN_BUILD}/bin/dg-eb-step"
c++ -std=c++17 -O2 -I "${DG_PIN}/include" -I "${DG_PIN}/common" \
    -I "${DG_PIN}/ggml/include" scripts/dg-encoder-dump.cpp \
    -L "${DG_PIN_BUILD}/bin" -lllama -lggml -lggml-base \
    -Wl,-rpath,"${DG_PIN_BUILD}/bin" -o "${ENCDUMP}"
c++ -std=c++17 -O2 scripts/dg-rng-dump.cpp -o "${RNGDUMP}"
c++ -std=c++17 -O2 scripts/dg-eb-step.cpp -o "${EBSTEP}"

cp "${DG_PROMPT}" "${DG_BUNDLE}/prompt-ids.i32"
# record the reference build's kernel-contract flags in the bundle
{ echo "pin: $(git -C "${DG_PIN}" rev-parse HEAD)";
  /usr/bin/grep -E "^(GGML_BLAS|GGML_METAL|GGML_ACCELERATE|GGML_NATIVE|CMAKE_BUILD_TYPE):" \
      "${DG_PIN_BUILD}/CMakeCache.txt"; } > "${DG_BUNDLE}/ref-build-config.txt"

# n_vocab: authoritative from the head's tied embedding [n_embd, n_vocab]
N_VOCAB=262144

# 1) reference RNG ground truth (canvas init + step-0 u/renoise)
"${RNGDUMP}" "${DG_SEED}" "${N_VOCAB}" 256 1 "${DG_BUNDLE}/rng"

# 2) reference unified checkpoints + logits (CPU backend, zero-SC)
"${ENCDUMP}" "${DG_GGUF}" "${DG_BUNDLE}/prompt-ids.i32" "${DG_BUNDLE}/ref" \
    "${DG_BUNDLE}/rng/canvas-ids.i32"

# 3) reference EB step-0 outputs from the reference logits' canvas rows
python3 - "${DG_BUNDLE}" "${N_VOCAB}" <<'PY'
import struct, sys
bundle, n_vocab = sys.argv[1], int(sys.argv[2])
# slice the canvas rows (last 256) out of result_output [n_vocab, N]
raw = open(f"{bundle}/ref/result_output.bin", "rb").read()
rows = len(raw) // (4 * n_vocab)
open(f"{bundle}/ref-canvas-logits.f32", "wb").write(raw[(rows - 256) * 4 * n_vocab:])
print(f"result_output rows={rows}, wrote canvas rows to ref-canvas-logits.f32")
PY
"${EBSTEP}" "${DG_BUNDLE}/ref-canvas-logits.f32" "${DG_SEED}" "${N_VOCAB}" 256 48 \
    0.4 0.8 0.1 "${DG_BUNDLE}/eb"

# 4) camelid RNG + unified forward + EB step + comparison
CAMELID_DG_GGUF="${DG_GGUF}" \
CAMELID_DG_DEC_REF="$(pwd)/${DG_BUNDLE}/ref" \
CAMELID_DG_DEC_IDS="$(pwd)/${DG_BUNDLE}/prompt-ids.i32" \
CAMELID_DG_DEC_RNG="$(pwd)/${DG_BUNDLE}/rng" \
CAMELID_DG_DEC_EB="$(pwd)/${DG_BUNDLE}/eb" \
CAMELID_DG_DEC_SEED="${DG_SEED}" \
CAMELID_DG_PIN_SHA="$(git -C "${DG_PIN}" rev-parse HEAD)" \
CAMELID_DG_DEC_OUT="$(pwd)/${DG_BUNDLE}/compare.json" \
    cargo test --release --test dg_decode_parity -- --nocapture

echo "bundle: ${DG_BUNDLE}"
