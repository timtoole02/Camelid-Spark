#!/usr/bin/env bash
# DiffusionGemma lane Phase 0.5 gate runner: dequant parity of camelid's lazy
# wire path vs llama.cpp's reference dequant on the SAME blocks of the SAME
# tracked GGUF (credit: llama.cpp / ggml authors — the reference side is their
# dequantization, compiled from the pinned checkout).
#
# Env:
#   DG_GGUF       tracked model file (required)
#   DG_PIN        pinned llama.cpp checkout with a cmake build (required)
#   DG_OUT_DIR    dump/working dir   (default: target/dg-quant-dumps-<utc>)
#   DG_GATE_OUT   gate artifact path (default: target/dg-quant-parity-<utc>.json)
#
# Tensor set: two tensors per quantized format in the tracked file (2D + 3D /
# early + late file offsets) plus the F32 router row; two block ranges each
# (head of tensor + middle of tensor) so parity is not a first-bytes accident.
set -euo pipefail

DG_GGUF="${DG_GGUF:?set DG_GGUF to the tracked GGUF path}"
DG_PIN="${DG_PIN:?set DG_PIN to the pinned llama.cpp checkout}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
DG_OUT_DIR="${DG_OUT_DIR:-target/dg-quant-dumps-${TS}}"
DG_GATE_OUT="${DG_GATE_OUT:-target/dg-quant-parity-${TS}.json}"
mkdir -p "${DG_OUT_DIR}"

DUMPER="${DG_PIN}/build/bin/dg-dequant-dump"
if [ ! -x "${DUMPER}" ]; then
    echo "building dg-dequant-dump against the pinned checkout..."
    c++ -std=c++17 -O2 -I "${DG_PIN}/ggml/include" scripts/dg-dequant-dump.cpp \
        -L "${DG_PIN}/build/bin" -lggml -lggml-base \
        -Wl,-rpath,"${DG_PIN}/build/bin" -o "${DUMPER}"
fi

# <tensor> <blocks-per-range>  (64 super/legacy blocks; 4096 for F32 scalars)
TENSORS=(
    "blk.0.attn_q.weight 64"            # Q4_K, 2D, early
    "blk.29.ffn_gate_up_exps.weight 64" # Q4_K, 3D fused experts, late
    "self_cond_down.weight 64"          # Q5_0, self-conditioning
    "blk.3.ffn_down_exps.weight 64"     # Q5_0, 3D experts
    "token_embd.weight 64"              # Q6_K, tied head
    "blk.0.attn_v.weight 64"            # Q6_K, attention V
    "blk.0.ffn_down_exps.weight 64"     # Q8_0, 3D experts, early
    "blk.29.ffn_down.weight 64"         # Q8_0, dense FFN, late
    "blk.0.ffn_gate_inp.weight 4096"    # F32 MoE router
)

MANIFEST="${DG_OUT_DIR}/manifest.json"
: > "${MANIFEST}"

idx=0
for spec in "${TENSORS[@]}"; do
    tensor="${spec% *}"
    n_blocks="${spec#* }"
    # head-of-tensor range
    line="$("${DUMPER}" "${DG_GGUF}" "${tensor}" 0 "${n_blocks}" "${DG_OUT_DIR}/dump-${idx}.bin")"
    echo "${line/${DG_OUT_DIR}\//}" >> "${MANIFEST}"
    total_blocks="$(printf '%s' "${line}" | sed -n 's/.*"total_blocks":\([0-9]*\).*/\1/p')"
    idx=$((idx + 1))
    # middle-of-tensor range
    mid=$(( (total_blocks / 2 / n_blocks) * n_blocks ))
    if [ "${mid}" -gt 0 ] && [ $((mid + n_blocks)) -le "${total_blocks}" ]; then
        line="$("${DUMPER}" "${DG_GGUF}" "${tensor}" "${mid}" "${n_blocks}" "${DG_OUT_DIR}/dump-${idx}.bin")"
        echo "${line/${DG_OUT_DIR}\//}" >> "${MANIFEST}"
        idx=$((idx + 1))
    fi
done
echo "wrote ${idx} reference dumps to ${DG_OUT_DIR}"

CAMELID_DG_QUANT_PARITY_DIR="$(cd "${DG_OUT_DIR}" && pwd)" \
CAMELID_DG_GGUF="${DG_GGUF}" \
CAMELID_DG_PIN_SHA="$(git -C "${DG_PIN}" rev-parse HEAD)" \
CAMELID_DG_QUANT_PARITY_OUT="$(pwd)/${DG_GATE_OUT}" \
    cargo test --test dg_quant_parity -- --nocapture

echo "gate artifact: ${DG_GATE_OUT}"
