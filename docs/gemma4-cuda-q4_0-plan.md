# Mixed-quant Q4_0 gemma4 E4B CUDA lane (6 GB-fit, mission C)

Branch `feat/gemma4-cuda-q4_0`. Goal: load + run the **mixed-quant** export
`gemma-4-E4B-it-Q4_0.gguf` with greedy parity vs the CPU oracle, fitting resident in 6 GB.
Builds on the merged Q8_0 CUDA lane (PR #316). Synthesized from a recon workflow
(2026-06-23); corrections below are load-bearing.

## Load-bearing facts (verified in recon)
- The file is a **mixed QAT export**: Q4_0 projections, **Q4_1** ffn_down (early layers),
  **Q4_K** tied head, **Q5_K** per_layer_token_embd, **BF16** per_layer_model_proj. The CPU
  oracle's `WireFormat` is `{Q8_0,Q4_0,Q6K}` and **fails closed** on the rest — it CANNOT load
  the file today. Extending it is the bulk of Phase 1.
- **Q4_1 block = 20 bytes** (f16 scale + f16 min + 16 nibble bytes, 32 values). `Q4_1_BLOCK_BYTES`
  in `tensor/mod.rs`. BUG: `gguf/reader.rs:~94` reports Q4_1 as `(32,18)` — must be `(32,20)`
  (Q4_0 stays 18). With 18 the wire byte-size check rejects ffn_down.
- **Q4_1 dequant = `q*scale + min`** (unsigned nibble, NO -8 bias), per `decode_q4_1_tensor`
  / `Q4_1Block`. NOT `scale*(q+min)`. The CUDA kernel must match exactly.
- **Head is Q4_K** → add `HeadLane::Q4K` (q4k_gemv exists), not the CPU fallback.
- **Reuse, do not reimplement**: `Q4_1Block`, `Q5KBlock`/`decode_q5_k_tensor`,
  `q4_k_wire_row_dot` (currently `#[allow(dead_code)]` — wire it up), `quantize_q8_k_blocks`.
- **VRAM ~2.52 GB resident** (per the original doc): Q5_K per_layer_token_embd (~1.94 GB) stays
  CPU/mmap-gathered; only the current token's row is dequant'd per step. Q4_K head (~0.38 GB) on
  GPU is fine (2.52+0.38 < 6). Default max_positions 8192.

## Phases (parity-gated, in order)
1. **CPU oracle loads the mixed file (THE parity gate).** Fix `gguf/reader.rs` Q4_1→(32,20).
   Extend `gemma4_runtime.rs` `WireFormat`/`WireQuant::new`/`values_per_block`/`bytes_per_block`
   /`matvec` (Q4_1→matvec_q, Q5K/Q4K→matvec_q8k)/`dequantize_elements` (Q4_1,Q5K). Add
   `q4_1_wire_row_dot` in inference.rs (mirrors `decode_q4_1_tensor`). Handle BF16
   per_layer_model_proj (check if it goes via WireQuant or a dense path). Capture a 64-token CPU
   greedy golden.
2. **q4_1_gemv CUDA kernel + unit parity test.** Clone `q4_0_gemv` (warp/row, Q8_0 activations,
   raw wire) but WIRE=20, read f16 scale@[0..2]+min@[2..4], nibbles@[4..], unsigned, `q*scale+min`.
   Add field/loader/`launch_q4_1_gemv`/`ProjQuant::Q4_1`. Unit test vs `q4_1_wire_row_dot`. **GATE.**
3. **Engine per-projection dispatch.** Per-projection `ProjQuant` tags on `Gemma4LayerWeightsDev`;
   `repack_proj` (Q8_0→SoA, Q4_0/Q4_1→raw); `dispatch_gemv` in the 7-projection loop. All layer
   projections take Q8_0 activations (Q4_0+Q4_1) → keep the single Q8_0 quantize (skip dual-buffer).
4. **Q4_K head lane.** `HeadLane::Q4K` + `WireFormat::Q4K` head arm → `rms_norm_quantize_q8k` →
   `q4k_gemv` → softcap → argmax. Confirm q4k_gemv weight layout (raw vs SoA) from the existing
   llama/qwen3 q4k_gemv caller — match it exactly.
5. **End-to-end parity + fit.** Oracle = CPU `Gemma4Runtime` on the SAME mixed file (identical
   weights → any divergence is a kernel bug). `argmax_stable` 64-token match + per-step logit tol.
   Log real `mem_get_info` to confirm ~2.52 GB resident. Verify serve gate routes the file.

## Risks
- Q4_1 18-vs-20 byte bug (fix reader first). Q4_1 formula (anchor to `decode_q4_1_tensor`).
- q4k_gemv raw-vs-SoA weight layout (read the existing caller, don't infer).
- BF16 per_layer_model_proj loading path (grep first).
- Recon flagged a possible GGUF-alignment quirk on the file — VERIFY it opens before kernel work.
