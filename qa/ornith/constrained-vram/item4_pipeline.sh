#!/usr/bin/env bash
# Item 4 pipeline (run stages one at a time; each stage assumes the box is quiet):
#   ./item4_pipeline.sh imatrix   — imatrix over Q8_0 (host-limit fallback; bf16 > RAM), calibration = TRACES_agentic_20.txt
#   ./item4_pipeline.sh quantize  — bf16 -> IQ3_XXS / Q3_K_M / IQ4_XS with the imatrix
#   ./item4_pipeline.sh ppl <model.gguf> <ngl>  — perplexity of one model on the held-out coding slice
# All tools are the pinned REF_QWEN35 build (acd79d6).
set -euo pipefail
BIN=${CAMELID_LLAMACPP_BIN:-$HOME/llama.cpp/build/bin}
MODELS=${CAMELID_MODELS_DIR:-models}
HERE="$(cd "$(dirname "$0")" && pwd)"

case "${1:-}" in
  imatrix)
    # Q8_0 host (9.5GB fits RAM; bf16 17.9GB would page-thrash 15.7GB) + partial GPU offload.
    # Deviation from the conductor's implied bf16-activations imatrix is documented in
    # RECEIPT_ITEM4 (Q8_0 is bit-certified vs the runnable oracle; activation stats at
    # Q8_0 fidelity are the standard practical proxy).
    "$BIN/llama-imatrix.exe" \
      -m "$MODELS/ornith-1.0-9b-Q8_0.gguf" \
      -f "$HERE/TRACES_agentic_20.txt" \
      -o "$HERE/imatrix_ornith_agentic.gguf" \
      -ngl 12 -c 2048 --seed 7
    ;;
  quantize)
    for t in IQ3_XXS Q3_K_M IQ4_XS; do
      out="$MODELS/ornith-1.0-9b-$t.gguf"
      if [ -f "$out" ]; then echo "skip $t (exists)"; continue; fi
      echo "== quantizing $t =="
      "$BIN/llama-quantize.exe" --imatrix "$HERE/imatrix_ornith_agentic.gguf" \
        "$MODELS/ornith-1.0-9b-bf16.gguf" "$out" "$t" 8
    done
    ls -la "$MODELS"/ornith-1.0-9b-{IQ3_XXS,Q3_K_M,IQ4_XS}.gguf
    ;;
  ppl)
    model="$2"; ngl="$3"
    echo "== perplexity: $(basename "$model") ngl=$ngl =="
    "$BIN/llama-perplexity.exe" -m "$model" -f "$HERE/heldout_coding.txt" \
      -c 2048 -ngl "$ngl" 2>&1 | tail -4
    ;;
  *) echo "usage: $0 imatrix|quantize|ppl <model> <ngl>"; exit 1 ;;
esac
