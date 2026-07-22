#!/usr/bin/env bash
# DiffusionGemma lane Phase 2 gate runner: encoder (PREFILL) per-layer
# checkpoint parity of camelid's DgEncoderRuntime vs pinned llama.cpp on the
# tracked model (credit: llama.cpp / ggml authors — the reference graph,
# checkpoints, and CPU kernels are theirs, compiled from the pinned checkout).
#
# Env:
#   DG_GGUF      tracked model file (required)
#   DG_PIN       pinned llama.cpp checkout with a cmake build (required)
#   DG_PACK      tokenizer pack for the prompt (default: the Phase 1 pack)
#   DG_CASE      pack case id used as the prompt (default: chat-hello)
#   DG_BUNDLE    bundle dir (default: target/dg-encoder-parity-<utc>/)
set -euo pipefail

DG_GGUF="${DG_GGUF:?set DG_GGUF to the tracked GGUF path}"
DG_PIN="${DG_PIN:?set DG_PIN to the pinned llama.cpp checkout}"
DG_PACK="${DG_PACK:-qa/prompt-packs/diffusiongemma-tokenizer-parity-v1.json}"
DG_CASE="${DG_CASE:-chat-hello}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
DG_BUNDLE="${DG_BUNDLE:-target/dg-encoder-parity-${TS}}"
mkdir -p "${DG_BUNDLE}/ref"

TOKDUMP="${DG_PIN}/build/bin/dg-tokenize-dump"
ENCDUMP="${DG_PIN}/build/bin/dg-encoder-dump"
if [ ! -x "${TOKDUMP}" ]; then
    cmake --build "${DG_PIN}/build" --target llama-common -j8 >/dev/null
    c++ -std=c++17 -O2 -I "${DG_PIN}/include" -I "${DG_PIN}/common" \
        -I "${DG_PIN}/ggml/include" -I "${DG_PIN}/vendor" \
        scripts/dg-tokenize-dump.cpp \
        -L "${DG_PIN}/build/bin" -lllama -lllama-common \
        -Wl,-rpath,"${DG_PIN}/build/bin" -o "${TOKDUMP}"
fi
if [ ! -x "${ENCDUMP}" ]; then
    c++ -std=c++17 -O2 -I "${DG_PIN}/include" -I "${DG_PIN}/common" \
        -I "${DG_PIN}/ggml/include" scripts/dg-encoder-dump.cpp \
        -L "${DG_PIN}/build/bin" -lllama -lggml -lggml-base \
        -Wl,-rpath,"${DG_PIN}/build/bin" -o "${ENCDUMP}"
fi

# 1) prompt ids from the committed pack via the pinned tokenizer (Phase 1 gate
#    proved camelid matches these ids 100%)
"${TOKDUMP}" "${DG_GGUF}" "${DG_PACK}" "${DG_BUNDLE}/tokens.json"
python3 - "$DG_BUNDLE" "$DG_CASE" <<'PY'
import json, struct, sys
bundle, case_id = sys.argv[1], sys.argv[2]
d = json.load(open(f"{bundle}/tokens.json"))
case = next(c for c in d["cases"] if c["id"] == case_id)
ids = case["tokens"]
open(f"{bundle}/prompt-ids.i32", "wb").write(struct.pack(f"<{len(ids)}i", *ids))
print(f"prompt {case_id}: {len(ids)} tokens: {ids}")
PY

# 2) reference checkpoints (CPU backend, PREFILL phase)
"${ENCDUMP}" "${DG_GGUF}" "${DG_BUNDLE}/prompt-ids.i32" "${DG_BUNDLE}/ref"

# 3) camelid encoder + comparison
CAMELID_DG_GGUF="${DG_GGUF}" \
CAMELID_DG_ENC_REF="$(pwd)/${DG_BUNDLE}/ref" \
CAMELID_DG_ENC_IDS="$(pwd)/${DG_BUNDLE}/prompt-ids.i32" \
CAMELID_DG_PIN_SHA="$(git -C "${DG_PIN}" rev-parse HEAD)" \
CAMELID_DG_ENC_OUT="$(pwd)/${DG_BUNDLE}/compare.json" \
    cargo test --release --test dg_encoder_parity -- --nocapture

echo "bundle: ${DG_BUNDLE}"
