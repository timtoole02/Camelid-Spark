#!/usr/bin/env bash
# DiffusionGemma lane Phase 1 gate runner: tokenizer parity of camelid's
# GGUF-bound tokenizer vs pinned llama.cpp on the tracked model (credit:
# llama.cpp / ggml authors — tokenization, chat-template rendering (minja),
# and detokenization reference semantics are theirs, compiled from the pinned
# checkout).
#
# Env:
#   DG_GGUF      tracked model file (required)
#   DG_PIN       pinned llama.cpp checkout with a cmake build (required)
#   DG_PACK      prompt pack (default: qa/prompt-packs/diffusiongemma-tokenizer-parity-v1.json)
#   DG_GATE_OUT  gate artifact path (default: target/dg-tokenizer-parity-<utc>.json)
set -euo pipefail

DG_GGUF="${DG_GGUF:?set DG_GGUF to the tracked GGUF path}"
DG_PIN="${DG_PIN:?set DG_PIN to the pinned llama.cpp checkout}"
DG_PACK="${DG_PACK:-qa/prompt-packs/diffusiongemma-tokenizer-parity-v1.json}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
DG_GATE_OUT="${DG_GATE_OUT:-target/dg-tokenizer-parity-${TS}.json}"

DUMPER="${DG_PIN}/build/bin/dg-tokenize-dump"
if [ ! -x "${DUMPER}" ]; then
    echo "building dg-tokenize-dump against the pinned checkout..."
    cmake --build "${DG_PIN}/build" --target llama-common -j8 >/dev/null
    c++ -std=c++17 -O2 -I "${DG_PIN}/include" -I "${DG_PIN}/common" \
        -I "${DG_PIN}/ggml/include" -I "${DG_PIN}/vendor" \
        scripts/dg-tokenize-dump.cpp \
        -L "${DG_PIN}/build/bin" -lllama -lllama-common \
        -Wl,-rpath,"${DG_PIN}/build/bin" -o "${DUMPER}"
fi

REF="target/dg-tokenizer-reference-${TS}.json"
mkdir -p target
"${DUMPER}" "${DG_GGUF}" "${DG_PACK}" "${REF}"

CAMELID_DG_GGUF="${DG_GGUF}" \
CAMELID_DG_TOK_REF="$(pwd)/${REF}" \
CAMELID_DG_PIN_SHA="$(git -C "${DG_PIN}" rev-parse HEAD)" \
CAMELID_DG_TOK_OUT="$(pwd)/${DG_GATE_OUT}" \
    cargo test --test dg_tokenizer_parity -- --nocapture

echo "reference dump: ${REF}"
echo "gate artifact:  ${DG_GATE_OUT}"
