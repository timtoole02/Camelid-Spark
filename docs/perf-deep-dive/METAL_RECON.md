# METAL_RECON.md

## 1. Header

**Purpose.** Phase 0 deliverable of the Camelid-vs-llama.cpp Metal (Apple Silicon) parity campaign (spec: `METAL_PARITY_AGENT_SPEC`). This document is a *recon inventory*: it enumerates every Metal compute kernel in both engines, traces the ordered M=1 decode dispatch sequence side-by-side, and classifies each llama.cpp kernel/family against Camelid's coverage to isolate the *real* gaps (as opposed to quant-breadth llama.cpp carries that Camelid does not target by design). No optimization work happens here.

**Pins.**
- llama.cpp @ `acd79d603`
- Camelid @ HEAD `2323033`

**Host.** Apple M4 (16-core-class GPU).

**Hard rule (non-negotiable).**
- **Study, do not copy.** llama.cpp Metal kernels are read for *strategy* (tile shapes, reduction structure, dispatch batching). Camelid implementations are Rust-native MSL of our own authorship.
- **Rust-native.** All Camelid kernels live in our MSL/Rust stack; no vendored `.metal` from llama.cpp.
- **No perf claim without a receipt.** Every WORSE / faster / slower statement in this document is tagged `suspected` and deferred to Phase 1 measurement. Phase 0 makes *zero* speed claims that are not backed by a receipt — and no receipts exist yet.

---

## 2. llama.cpp Metal kernel inventory

llama.cpp's Metal backend carries broad quant coverage: 26 distinct quant/dtype paths in `mul_mv`, plus `ext` variants (`r1ptg=2..5`), full `get_rows`, full MoE `mul_mv_id`, and a single templated GEMM (`mul_mm`) instantiated across all quants × {f32,f16} output.

### 2.1 GEMV / matvec — `kernel_mul_mv_*`

Single quantized matvec, M=1 decode core. SIMD-group reduction via `simd_sum()` (full 32-wide), `tiisg==0` writes. `SZ_SIMDGROUP=16` packing, `N_SG` simdgroups/threadgroup, `N_R0` rows/simdgroup.

| Kernel family | dtype/quant | N_R0 | N_SG | shmem |
|---|---|---|---|---|
| `kernel_mul_mv_q1_0_f32` | Q1_0 | 8 | 2 | no |
| `kernel_mul_mv_q4_0_f32` | Q4_0 | 4 | 2 | yes |
| `kernel_mul_mv_q4_1_f32` | Q4_1 | 4 | 2 | yes |
| `kernel_mul_mv_q5_0_f32` | Q5_0 | 4 | 2 | yes |
| `kernel_mul_mv_q5_1_f32` | Q5_1 | 4 | 2 | yes |
| `kernel_mul_mv_q8_0_f32` | Q8_0 | 2 | 4 | yes |
| `kernel_mul_mv_q2_K_f32` | Q2_K | 4 | 2 | no |
| `kernel_mul_mv_q3_K_f32` | Q3_K | 2 | 2 | no |
| `kernel_mul_mv_q4_K_f32` | Q4_K | 2 | 2 | no |
| `kernel_mul_mv_q5_K_f32` | Q5_K | 1 | 2 | no |
| `kernel_mul_mv_q6_K_f32` | Q6_K | 2 | 2 | no |
| `kernel_mul_mv_iq1_s_f32` | IQ1_S | 4 | 2 | no |
| `kernel_mul_mv_iq1_m_f32` | IQ1_M | 4 | 2 | no |
| `kernel_mul_mv_iq2_xxs_f32` | IQ2_XXS | 4 | 2 | yes (grid) |
| `kernel_mul_mv_iq2_xs_f32` | IQ2_XS | 4 | 2 | yes (grid+signs) |
| `kernel_mul_mv_iq2_s_f32` | IQ2_S | 4 | 2 | — |
| `kernel_mul_mv_iq3_xxs_f32` | IQ3_XXS | 4 | 2 | yes (grid) |
| `kernel_mul_mv_iq3_s_f32` | IQ3_S | 4 | 2 | — |
| `kernel_mul_mv_iq4_nl_f32` | IQ4_NL | 2 | 2 | yes (table) |
| `kernel_mul_mv_iq4_xs_f32` | IQ4_XS | 2 | 2 | yes (kvalues) |
| `kernel_mul_mv_mxfp4_f32` | MXFP4 | 2 | 2 | — |

Dense matvec (`kernel_mul_mv_t_t_disp`), with scalar / float4 (`_4`) / short (`_short`, small K) variants:

| Kernel | in→out | variant |
|---|---|---|
| `kernel_mul_mv_f32_f32` / `_4` / `_short` | F32→F32 | scalar/vec4/short |
| `kernel_mul_mv_f16_f32` / `_4` / `_short` | F16→F32 | scalar/vec4/short |
| `kernel_mul_mv_f16_f16` / `_4` / `_short` | F16→F16 | scalar/vec4/short |
| `kernel_mul_mv_bf16_f32` / `_4` / `_short` | BF16→F32 | scalar/vec4/short (gated on `GGML_METAL_HAS_BF16`) |
| `kernel_mul_mv_bf16_bf16` / `_4` / `_short` | BF16→BF16 | scalar/vec4/short |

**Ext matvec** (`kernel_mul_mv_ext_*`), function-constant dispatched over `r1ptg ∈ {2,3,4,5}` (rows-per-thread) for variable-batch decode:

| Family | dtypes | chpb |
|---|---|---|
| `kernel_mul_mv_ext_{f32,f16,bf16}_f32_r1_2..5` | F32 / F16 / BF16 | 4 |
| `kernel_mul_mv_ext_q1_0_f32_r1_2..5` | Q1_0 | 128 |
| `kernel_mul_mv_ext_{q4_0,q4_1,q5_0,q5_1,q8_0,mxfp4,iq4_nl}_f32_r1_2..5` | legacy 32-blk | 32 |
| `kernel_mul_mv_ext_{q2_K,q3_K,q4_K,q5_K,q6_K}_f32_r1_2..5` | K-quants (q4x4 path) | 16 |

**MoE expert-select matvec** (`kernel_mul_mv_id_*`): full mirror of the above quant/dtype set wrapped in `kernel_mul_mv_id<mmv_fn<...>>`, dispatched per-expert-row. Covers F32/F16/BF16 (scalar+`_4`), Q1_0/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/MXFP4, Q2_K…Q6_K, IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_NL/IQ4_XS.

### 2.2 GEMM / matmul — `kernel_mul_mm`, `kernel_mul_mm_id`

Single templated simdgroup-matrix GEMM, prefill / batched. `SZ_SIMDGROUP=16`.

| Kernel | role | dtypes |
|---|---|---|
| `kernel_mul_mm` | prefill GEMM | all quants × {f32,f16} out: f32/f16/bf16, Q1_0/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/MXFP4, Q2_K…Q6_K, IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_NL/IQ4_XS |
| `kernel_mul_mm_id` | MoE prefill GEMM | same coverage as `mul_mm` |
| `kernel_mul_mm_id_map0` | MoE token→expert index map | templated on `n_expert_used ∈ {1,2,4,5,6,8,10,16,22}`; not a compute kernel |

Two impl paths per `mul_mm`:
- **`GGML_METAL_HAS_TENSOR`**: cooperative-tensor `matmul2d` (`mpp::tensor_ops`), tile `NRB=128 × NRA=64 × N_MM_NK_TOTAL=32`, dequant-on-the-fly into threadgroup SRAM, `N_MM_BLOCK_{X,Y}={4,2}`, `N_MM_SIMD_GROUP_{X,Y}={2,2}`.
- **Fallback**: classic `simdgroup_float8x8` outer-product accumulation, `NR0=64` M-tile × `NR1=32` N-tile × `NK=32` K-chunk, ~8KB shared SRAM.

### 2.3 flash-attn — `kernel_flash_attn_ext*`

Selection: `vec` when `ne01 (batch/seq) < 20` AND `head-dim % 32 == 0` (decode / small batch); `non-vec` otherwise (prefill / large batch). Both use **online softmax** (M max, S accumulator) and fuse Q@Kᵀ → scale/mask/ALiBi/softcap → softmax → @V in one pass per KV block. Sequential KV block iteration (context-flat, no depth scaling).

| Kernel | role | K/V dtypes | notes |
|---|---|---|---|
| `kernel_flash_attn_ext_pad` | pre-pass | F16, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 | pad KV seq to 64-elem block, `-MAXHALF` mask pad |
| `kernel_flash_attn_ext_blk` | pre-pass | F16 (mask) | mask-block classifier (0=skip / 1=process / 2=zero) via `simd_min/max` |
| `kernel_flash_attn_ext` | prefill/large batch | F16/F32/BF16, Q4_0/Q4_1/Q5_0/Q5_1 | tile Q=8 × C=64; `simdgroup_multiply_accumulate`; head-dim ∈ {32,40,48,64,72,80,96,112,128,192,256,320,512,576}; NSG=4 or 8 |
| `kernel_flash_attn_ext_vec` | decode/small batch | F16/F32/BF16, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 | nqpsg=1, ncpsg=32; head-dim ∈ {32,64,96,128,192,256}; NWG (1–4) work-group batching |
| `kernel_flash_attn_ext_vec_reduce` | reduce | float accum | 2-stage gather across NWG workers when DV large |

### 2.4 rope — `kernel_rope_*`

| Kernel | dtypes | notes |
|---|---|---|
| `kernel_rope_norm` | f32, f16 | standard paired rotation |
| `kernel_rope_neox` | f32, f16 | NeoX paired-halves |
| `kernel_rope_multi` | f32, f16 | MROPE sectioned dims |
| `kernel_rope_vision` | f32, f16 | 2D-aware vision MROPE |

### 2.5 norms — `kernel_*norm*`

| Kernel | dtypes | notes |
|---|---|---|
| `kernel_rms_norm_fuse_impl` | f32, f32_4 | RMS norm, 3 fuse variants (none / mul / mul+add) |
| `kernel_norm_fuse_impl` | f32, f32_4 | layer norm, 3 fuse variants |
| `kernel_l2_norm_impl` | f32, f32_4 | L2 normalize |
| `kernel_group_norm_f32` | f32 | per-group stats |

### 2.6 softmax — `kernel_soft_max*`

| Kernel | dtypes | notes |
|---|---|---|
| `kernel_soft_max` | f16, f32 | ALiBi mask + attn scale |
| `kernel_soft_max_4` | f16_4, f32_4 | float4 vectorized |

### 2.7 dequant / get_rows — `kernel_get_rows_*`, `kernel_cpy_*`

| Kernel | dtypes |
|---|---|
| `kernel_get_rows_f{32,16}`, `_i32`, `_bf16` | dense / int |
| `kernel_get_rows_q{1_0,4_0,4_1,5_0,5_1,8_0}`, `_mxfp4` | legacy (nl=2/8) |
| `kernel_get_rows_q{2_K,3_K,4_K,5_K,6_K}` | K-quant (nl=256) |
| `kernel_get_rows_iq{2_xxs,2_xs,3_xxs,3_s,2_s,1_s,1_m,4_nl,4_xs}` | IQ (nl=256, IQ4_NL nl=2) |
| `kernel_cpy_f32_q` | f32 → {q8_0,q1_0,q4_0,q4_1,q5_0,q5_1,iq4_nl} (inline quantize) |
| `kernel_cpy_q_f32` | {q1_0,q4_0,q4_1,q5_0,q5_1,q8_0,q2_K…q6_K,iq*} → f32 (dequantize) |
| `kernel_cpy_t_t` | f32/f16/bf16/i32 cross-cast copy |

### 2.8 KV-cache — `kernel_get_rows_{q,f}`

KV gather is served by `kernel_get_rows_q` (q1_0…q6_K) and `kernel_get_rows_f` (f32/f16). llama.cpp has no dedicated KV-scatter kernel in this inventory; KV write is fused via the attention/cpy path.

### 2.9 sampling — `kernel_argmax_f32`, `kernel_argsort_*`

| Kernel | dtypes | notes |
|---|---|---|
| `kernel_argmax_f32` | f32 | greedy argmax |
| `kernel_argsort_f32_i32` | f32 | bitonic sort, top-k |
| `kernel_argsort_merge_f32_i32` | f32 | merge phase |

### 2.10 elementwise / misc

`kernel_unary_impl` (scale, fill, clamp, sqr, sqrt, sin, cos, log, leaky_relu, tanh, relu, sigmoid), `kernel_bin_fuse_impl` (add/sub/mul/div, broadcast + multi-src fusion), `kernel_add_id`, `kernel_cumsum_add`, `kernel_diag_f32`, `kernel_repeat`, `kernel_concat`, `kernel_im2col` / `_ext`, and SSM (`kernel_ssm_conv_f32_f32` ±`_4`/`_batched`, `kernel_ssm_scan_f32`).

---

## 3. Camelid Metal kernel inventory (~63 kernels)

Camelid is **Q8_0-focused** (Q8_0 / Q4_0 only by design), F32Y default (all kernels output F32, no in-decode quantize), optional F16 KV. Grouped the same way:

### 3.1 norms (6)

| Kernel | dtypes |
|---|---|
| `rms_norm_f32` | F32 |
| `rms_norm_quantize_f32` | F32 → Q8_0 |
| `rms_norm_batch_f32` | F32 |
| `rms_norm_batch_f16o` | F16 |
| `rms_norm_batch_h` | F16 |
| `rms_norm_per_head_f32` | F32 |

### 3.2 GEMV / matvec (13)

| Kernel | dtypes |
|---|---|
| `linear_row_f32` | F32 |
| `linear_row_transposed_f32` | F32 |
| `q8_0_encoded_linear_row` | Q8_0 |
| `q8_0_encoded_linear_rows` | Q8_0 |
| `q8_0_block_linear_row` | Q8_0 |
| `q8_0_block_linear_row_simd` | Q8_0 |
| `q8_0_block_linear_row_simd_mr` | Q8_0 |
| `q8_0_block_linear_row_simd_qmv4` | Q8_0 |
| `q8_0_block_linear_row_ksplit` | Q8_0 |
| `q8_0_block_linear_row_ksplit_f32y` | Q8_0 → F32 |
| `q8_0_block_linear_row_ksplit_f32y_wire` | Q8_0 (wire) → F32 *(decode default)* |
| `q8_0_block_linear_row_ksplit_f32y_wire_nsg8` | Q8_0 (wire) → F32 |
| `q4_0_block_linear_row_ksplit_f32y_wire` | Q4_0 (wire) → F32 |

### 3.3 GEMM / matmul (10)

| Kernel | dtypes |
|---|---|
| `q8_0_block_linear_ksplit_f32y_wire_gemm` | Q8_0 (wire) → F32 |
| `q8_0_block_wire_mm` | Q8_0 (wire) → F32 |
| `q8_0_block_wire_mm_f16o` | Q8_0 (wire) → F16 |
| `kernel_mul_mm_q8_0_f32` | Q8_0 → F32 |
| `steel_q8_mm` | Q8_0 |
| `steel_q8_mm_dual` | Q8_0 |
| `half_mm_batched` | F16 |
| `half_mm_batched_f16o` | F16 → F16 |
| `mma_probe` | F32 (probe) |

### 3.4 decode-attn (9)

| Kernel | dtypes | notes |
|---|---|---|
| `attention_decode_f32` | F32 | |
| `attention_decode_kv16` | F16 KV | |
| `attention_decode_v2_f32` | F32 | tiled, 4 simdgroups/head (default) |
| `attention_decode_v2_kv16` | F16 KV | |
| `attention_decode_splitk_f32` | F32 | split-K over positions (default on) |
| `attention_decode_splitk_kv16` | F16 KV | |
| `attention_decode_splitk_kv16_direct` | F16 KV | direct-read, head_dim==128 |
| `attention_decode_splitk_merge_f32` | F32 | split-K merge |
| `attention_splitk_kv16_stageonly` | F16 KV | |

### 3.5 flash-attn / prefill-attn (3)

| Kernel | dtypes |
|---|---|
| `attention_prefill_flash_f32` | F32 |
| `attention_prefill_v2_f32` | F32 |
| `attention_prefill_v3_f32` | F32 |

### 3.6 rope (4)

| Kernel | dtypes |
|---|---|
| `rope_rotate_f32` | F32 |
| `rope_rotate_batch_f32` | F32 |
| `rope_scatter_qh_batch` | F32 |
| `rope_scatter_qh_batch_h` | F16 |

### 3.7 KV-cache (6)

| Kernel | dtypes |
|---|---|
| `kv_scatter_f32` | F32 |
| `kv_scatter_kv16` | F16 KV |
| `kv_scatter_batch_f32` | F32 |
| `transpose_v16` | F16 |

### 3.8 activation / elementwise (11)

| Kernel | dtypes |
|---|---|
| `silu_mul_f32` | F32 |
| `silu_mul_f16o` | F16 |
| `silu_mul_h2` | F16 |
| `silu_mul_quantize_f32` | F32 → Q8_0 |
| `gelu_mul_f32` | F32 |
| `soft_cap_f32` | F32 |
| `scale_f32` | F32 |
| `residual_add_f32` | F32 |
| `residual_add_h` | F16 |

### 3.9 softmax (1)

| Kernel | dtypes |
|---|---|
| `softmax_causal_rows` | F32 |

### 3.10 quantize / dequant / embed (4)

| Kernel | dtypes |
|---|---|
| `quantize_q8_0_f32` | F32 → Q8_0 |
| `f32_to_f16` | F32 → F16 |
| `embed_row_gather_q8_wire` | Q8_0 (wire) → F32 |

### 3.11 sampling (1)

| Kernel | dtypes |
|---|---|
| `argmax_f32_greedy` | F32 |

**Coverage summary.** Camelid carries Q8_0 (wire + encoded + block variants) and a single Q4_0 wire matvec; no K-quant, no IQ-*, no MXFP4, no BF16, no dense-F16/F32 quant matvec beyond the F32 reference rows. This is by design (Q8_0/Q4_0-only).

---

## 4. M=1 decode path trace (one token, ordered Metal dispatch)

Both engines run a per-layer loop then a final norm + logits + sample tail. The side-by-side below is the *ordered dispatch sequence* for one decode token.

### 4.1 Ordered dispatch — side by side

| # | llama.cpp | Camelid |
|---|---|---|
| embed | `get_rows` (embed lookup) | (embed gathered at tail of *previous* token via `embed_row_gather_q8_wire`) |
| **per layer ↓** | | |
| 1 | `rms_norm` (attn norm, fused) | `rms_norm_f32` |
| 2 | `mul_mat_q8_0` (Q proj) → `kernel_mul_mv_q8_0_f32` | `q8_0_block_linear_row_ksplit_f32y_wire` (Q) |
| 3 | `mul_mat_q8_0` (K proj) | `q8_0_block_linear_row_ksplit_f32y_wire` (K) |
| 4 | `mul_mat_q8_0` (V proj) | `q8_0_block_linear_row_ksplit_f32y_wire` (V) |
| 5 | `rope` (Q+K positions) | `rope_rotate_f32` (Q), `rope_rotate_f32` (K) |
| 6 | (KV write fused in attn/cpy path) | `kv_scatter_f32` (K,V → cache slot) |
| 7 | `flash_attn_ext` (vec variant; may add `pad`/`blk` sub-kernels) | `attention_decode_v2_f32` *(or `attention_decode_splitk_f32` + `attention_decode_splitk_merge_f32` when positions ≥ 128)* |
| 8 | `mul_mat_q8_0` (O proj) | `q8_0_block_linear_row_ksplit_f32y_wire` (O) |
| 9 | `add` (attn residual) | `residual_add_f32` |
| 10 | `rms_norm` (ffn norm, fused) | `rms_norm_f32` |
| 11 | `mul_mat_q8_0` (gate proj) | `q8_0_block_linear_row_ksplit_f32y_wire` (gate) |
| 12 | `mul_mat_q8_0` (up proj) | `q8_0_block_linear_row_ksplit_f32y_wire` (up) |
| 13 | `mul` / GLU (silu·gate) | `silu_mul_f32` |
| 14 | `mul_mat_q8_0` (down proj) | `q8_0_block_linear_row_ksplit_f32y_wire` (down) |
| 15 | `add` (ffn residual) | `residual_add_f32` |
| **tail ↓** | | |
| 16 | `rms_norm` (final) | `rms_norm_f32` |
| 17 | `mul_mat_q8_0` (output/logits proj) | `q8_0_block_linear_row_ksplit_f32y_wire` (logits) |
| 18 | `softmax` (logits) → sampling | `argmax_f32_greedy` (greedy) |
| 19 | — | `embed_row_gather_q8_wire` (gather sampled embed → next token input) |

Notes on op-count parity: the two paths are structurally identical through the transformer block. Camelid uses one matvec kernel (`q8_0_block_linear_row_ksplit_f32y_wire`) for all six projections; llama.cpp uses the same `kernel_mul_mv_q8_0_f32` for all six. The visible differences are (a) Camelid's explicit `kv_scatter_f32` vs llama.cpp's fused KV write, (b) llama.cpp's potential `flash_attn_ext_pad`/`_blk` sub-dispatches, (c) Camelid's tail `embed_row_gather_q8_wire` that pre-stages the *next* token's input, and (d) sampling: Camelid uses on-GPU `argmax_f32_greedy` (greedy-only), llama.cpp emits `softmax` + argsort/argmax for general sampling.

### 4.2 Command-buffer / encoder batching + residency

**llama.cpp.**
- Graph split into `GGML_METAL_MAX_COMMAND_BUFFERS = 8` + 1 main. First `n_main = MAX(64, 0.1·n_nodes)` nodes encoded on the main thread; remainder distributed across `n_cb` worker threads.
- **One `MTLComputeCommandEncoder` per command buffer**, created `MTLDispatchTypeConcurrent` when `use_concurrency=true`. Each op = one `dispatchThreadgroups`; no batching of ops into a single dispatch. `ggml_metal_encoder_memory_barrier` serializes dependent ops within the concurrent encoder.
- Residency: no persistent `setBuffers` by default — buffers set per-op. `MTLHeap` residency sets supported on macOS 15+ (`ggml_metal_rsets_t`) but activation policy unclear from the source.
- Pipeline cache: per-op PSO cached in a name-keyed hashmap (`ggml_metal_pipelines_t`), compiled lazily on first use.
- Fusion: op-level only — `rms_norm`+mul+add → 1 dispatch; add-chains → `add_id`. No graph-level fusion before encoding.
- For M=1: a single token is **multiple command buffers** (graph split across the 8+1 budget), each with one concurrent encoder, dependent ops barrier-ordered.

**Camelid.**
- **Single compute encoder for the entire token** (`cb.new_compute_command_encoder()`), all dispatches batched into ONE encoder; Metal hazard-tracking orders dependent ops and overlaps independent ones within it.
- One command buffer per token: `e.end_encoding()` → `cb.encode_signal_event(done_event)` → `cb.commit()` (gated on a CPU-side gate event). GPU-stamped completion polled via `done_event` (no kernel-wake).
- **Persistent residency:** `buf_a`, `buf_b`, `cache_k[]`, `cache_v[]` (+ optional `cache_k16[]`/`cache_v16[]`) pre-allocated to `max_positions`, grown on demand to `kv_cap`. Weights cached in `metal_linear_cache` keyed by pointer+len; wire-format Q8_0 (34-byte blocks) bypass host copies via no-copy wire pages.
- **Encode-ahead pipeline:** while the GPU runs token *t*, the CPU prepares token *t+1*'s graph (`prepare_token` with `next_rope` tables for position+1). When the next graph has a sample stage, `gate_event` is signaled so the GPU embedding gather feeds the next graph's input — greedy token-to-token with ~zero CPU on the critical path. Skipped only at KV-growth reallocation edges.
- Per-token reuse: `forward_token` reuses a pending graph when (position, logits_stage, sample_stage) match, avoiding re-encode.

**Recon read.** llama.cpp = many command buffers / per-op buffer binding / barrier-serialized concurrent encoder. Camelid = one encoder + one command buffer per token, persistent buffers, no-copy wire weights, and CPU-overlapped encode-ahead. This is the **dispatch-overhead / command-buffer axis** and is the most structurally divergent part of the two engines — Phase 1 should measure per-kernel `GPUStartTime`/`GPUEndTime` gaps and CPU-side encode time here.

---

## 5. Gap table

Status legend: **HAVE** = Camelid has an equivalent; **MISSING (real)** = a capability Camelid lacks that matters for its supported set; **MISSING (N/A)** = llama quant/dtype outside Camelid's Q8_0/Q4_0-only design (not a real gap); **WORSE (suspected)** = Camelid has it but is suspected slower — *measured reason deferred to Phase 1, no speed numbers asserted*.

### 5.1 GEMV / matvec

| llama.cpp family | Camelid status | Mapped Camelid kernel |
|---|---|---|
| `kernel_mul_mv_q8_0_f32` | **HAVE** | `q8_0_block_linear_row_ksplit_f32y_wire` (+ simd/mr/qmv4/nsg8 variants) |
| `kernel_mul_mv_q4_0_f32` | **HAVE** | `q4_0_block_linear_row_ksplit_f32y_wire` |
| `kernel_mul_mv_{f32,f16,bf16}_f32` (+`_4`/`_short`) | partial — **HAVE F32 only** | `linear_row_f32`, `linear_row_transposed_f32`; F16/BF16 dense matvec MISSING (N/A — Camelid decodes from Q8_0, not dense quant) |
| `kernel_mul_mv_q4_1/q5_0/q5_1/q1_0` | **MISSING (N/A)** | not in Camelid's supported quant set |
| `kernel_mul_mv_q{2,3,4,5,6}_K` | **MISSING (N/A)** | K-quants not supported |
| `kernel_mul_mv_iq*` (IQ1_S…IQ4_XS), `mxfp4` | **MISSING (N/A)** | extreme/MX quants not supported |
| `kernel_mul_mv_ext_*` (`r1ptg=2..5`) | **MISSING (real, narrow)** | no rows-per-thread function-constant ext path; Camelid's `ksplit` covers M=1 but lacks the small-variable-batch `r1ptg` tuning lane — relevant only for Q8_0/Q4_0 |
| `kernel_mul_mv_id_*` (MoE expert matvec) | **MISSING (real, if MoE)** | no MoE expert-select matvec in inventory; gates MoE models (out of current dense scope) |

### 5.2 GEMM / matmul

| llama.cpp family | Camelid status | Mapped Camelid kernel |
|---|---|---|
| `kernel_mul_mm` (Q8_0 prefill GEMM) | **HAVE / WORSE (suspected)** | `q8_0_block_wire_mm`, `kernel_mul_mm_q8_0_f32`, `steel_q8_mm`, `steel_q8_mm_dual`, `q8_0_block_linear_ksplit_f32y_wire_gemm`. **Suspected WORSE** vs llama's `kernel_mul_mm` — **on this M4 the live llama path is the `simdgroup_float8x8` fallback (NR0=64×NR1=32×NK=32), NOT the cooperative-tensor `matmul2d` (which is M5+/A19+ only — see §6d)**. So the M4 comparison is simdgroup-matrix vs simdgroup-matrix; measured reason deferred to Phase 1. This is a **suspected real gap (prefill/GEMM)**. |
| `kernel_mul_mm` (F16 prefill) | **HAVE** | `half_mm_batched`, `half_mm_batched_f16o` |
| `kernel_mul_mm` (Q4_0 prefill) | **MISSING (real, narrow)** | Camelid has Q4_0 *matvec* (`q4_0_block_linear_row_ksplit_f32y_wire`) but no Q4_0 *GEMM* in inventory; relevant for Q4_0 prefill |
| `kernel_mul_mm` (other quants/dtypes) | **MISSING (N/A)** | K-quant/IQ/MXFP4/BF16 GEMM outside supported set |
| `kernel_mul_mm_id`, `kernel_mul_mm_id_map0` (MoE GEMM) | **MISSING (real, if MoE)** | no MoE prefill GEMM / expert-map kernel |

### 5.3 flash-attn

Camelid splits attention into a **decode-attn** family (M=1) and a **prefill flash** family; llama.cpp uses one `flash_attn_ext` family with `vec`/`non-vec` selection.

| llama.cpp kernel | Camelid status | Mapped Camelid kernel |
|---|---|---|
| `kernel_flash_attn_ext_vec` (decode, nq<20) | **HAVE** | `attention_decode_v2_f32` / `attention_decode_splitk_f32` (+`_merge`); KV16 mirrors `*_kv16` / `_kv16_direct`. Online-softmax, tiled, split-K-over-positions at depth ≥128. |
| `kernel_flash_attn_ext` (prefill / large batch) | **HAVE / WORSE (suspected)** | `attention_prefill_flash_f32`, `attention_prefill_v2_f32`, `attention_prefill_v3_f32`. **Suspected WORSE at depth** vs llama's tiled (Q=8×C=64) `simdgroup_multiply_accumulate` path with its wide head-dim specialization set — measured reason deferred to Phase 1. **Suspected real gap (flash-attn at depth).** |
| `kernel_flash_attn_ext_pad` (KV seq pad to 64) | **partial / suspected** | Camelid handles unaligned KV via in-attn pad-buffer swaps (per the decode trace), not a dedicated pre-pass kernel; equivalence is structural, parity-impact deferred to Phase 1 |
| `kernel_flash_attn_ext_blk` (mask-block classifier) | **MISSING (real, narrow)** | no mask-block skip classifier; Camelid does not skip fully-masked KV blocks — a *potential* depth optimization, only relevant where masking creates skippable blocks |
| `kernel_flash_attn_ext_vec_reduce` (multi-WG gather) | **HAVE (analog)** | `attention_decode_splitk_merge_f32` is the analogous split-K partial-merge |
| Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 KV-quant attention | **MISSING (N/A mostly)** | Camelid KV is F32 or F16; quantized-KV attention outside supported set (F16 KV is the supported lower-precision lane) |

**Parity flag.** llama.cpp's `vec` path and any coalesced/multi-workgroup KV reduction (`flash_attn_ext_vec_reduce`, NWG batching) re-associate the softmax/V reduction across workers. Such paths are **faster-but-parity-unsafe** for bit-exact greedy parity and are off-limits to copy even if Phase 1 shows them faster (see §6c).

### 5.4 rope

| llama.cpp | Camelid status | Mapped Camelid kernel |
|---|---|---|
| `kernel_rope_norm` | **HAVE** | `rope_rotate_f32`, `rope_rotate_batch_f32` |
| `kernel_rope_neox` | **HAVE** (per-head scatter variants) | `rope_scatter_qh_batch`, `rope_scatter_qh_batch_h` |
| `kernel_rope_multi` (MROPE) | **MISSING (real, if MROPE model)** | no sectioned-dim MROPE kernel |
| `kernel_rope_vision` | **MISSING (real, if vision)** | no 2D vision RoPE |

### 5.5 norms

| llama.cpp | Camelid status | Mapped Camelid kernel |
|---|---|---|
| `kernel_rms_norm_fuse_impl` | **HAVE** | `rms_norm_f32`, `rms_norm_batch_*`, `rms_norm_per_head_f32`, `rms_norm_quantize_f32` |
| `kernel_norm_fuse_impl` (layer norm) | **MISSING (real, if model needs LN)** | Camelid has RMS norm only; no general layer-norm kernel |
| `kernel_l2_norm_impl` | **MISSING (real, narrow)** | no L2 norm |
| `kernel_group_norm_f32` | **MISSING (N/A)** | conv/group-norm not in LM decode scope |

### 5.6 softmax

| llama.cpp | Camelid status | Mapped Camelid kernel |
|---|---|---|
| `kernel_soft_max` / `_4` (ALiBi + scale) | **HAVE (subset)** | `softmax_causal_rows` (causal); standalone softmax is largely subsumed by fused online-softmax in the attention kernels. ALiBi-masked standalone softmax not separately present — relevant only for non-flash attention paths. |

### 5.7 dequant / get_rows / KV

| llama.cpp | Camelid status | Mapped Camelid kernel |
|---|---|---|
| `kernel_get_rows_q8_0` (embed gather) | **HAVE** | `embed_row_gather_q8_wire` |
| `kernel_get_rows_f{32,16}` | **HAVE (analog)** | `f32_to_f16`, embed gather path |
| `kernel_get_rows_q4_0` | **HAVE (N/A others)** | covered by wire path for supported quant |
| `kernel_get_rows_{q4_1,q5_*,q1_0,K-quant,iq*,mxfp4}` | **MISSING (N/A)** | unsupported quants |
| `kernel_cpy_q_f32` / `kernel_cpy_f32_q` (general (de)quant copy) | partial — **HAVE Q8_0** | `quantize_q8_0_f32` (f32→Q8_0); general multi-quant (de)quant copy MISSING (N/A) |
| KV write | **HAVE (explicit)** | `kv_scatter_f32`, `kv_scatter_kv16`, `kv_scatter_batch_f32`, `transpose_v16` (llama fuses KV write; Camelid has dedicated scatter) |

### 5.8 sampling

| llama.cpp | Camelid status | Mapped Camelid kernel |
|---|---|---|
| `kernel_argmax_f32` | **HAVE** | `argmax_f32_greedy` |
| `kernel_argsort_f32_i32`, `_merge` (top-k / general sampling) | **MISSING (real, narrow)** | no on-GPU bitonic argsort; Camelid is greedy-on-GPU only (general sampling presumably CPU-side / out of this Metal inventory) |

### 5.9 elementwise / activation / misc

| llama.cpp | Camelid status | Mapped Camelid kernel |
|---|---|---|
| `kernel_bin_fuse_impl` (add/sub/mul/div + broadcast) | **HAVE (subset)** | `residual_add_f32`/`_h`, `scale_f32`, `silu_mul_*`, `gelu_mul_f32`, `soft_cap_f32` |
| `kernel_unary_impl` (scale/clamp/relu/sigmoid/tanh/…) | **HAVE (subset)** | `scale_f32`, `soft_cap_f32`; full unary menu MISSING (N/A — most unused by Q8_0 LM decode) |
| `kernel_add_id`, `kernel_cumsum_add`, `kernel_diag_f32`, `kernel_repeat`, `kernel_concat` | **MISSING (N/A)** | not on the dense LM decode/prefill path |
| `kernel_im2col` / `_ext`, `kernel_ssm_*` | **MISSING (N/A)** | conv / SSM (Mamba) outside Camelid's transformer scope |

---

## 6. Caveats / constraints

**(a) Phase 1 fine profiling is BLOCKED on tooling.** This M4 host has only the Command Line Tools — **no Xcode.app**. Therefore Instruments, `xctrace`, and `xcrun -sdk macosx metal` GPU-capture tooling are unavailable (consistent with the standing GPU-profiler-access finding: hardware perf counters are entitlement-gated and the headless private-framework route is a dead end). What *is* still possible without Xcode:
- A coarse **tokens/s → GB/s roofline** (decode is bandwidth-bound; compare achieved against the ~120 GB/s M4 wall).
- **Per-kernel `MTLCommandBuffer.GPUStartTime` / `GPUEndTime`** timing (and Camelid's existing `CAMELID_RESIDENT_TRACE=1` per-token window: pre-GPU / encode / GPU-busy / kernel window).
- No occupancy / ALU-stall / limiter counters until Xcode.app is installed (deferred).

**(b) Decode matvec is already at the wall — do not chase it.** Prior M4 work established that **3B Q8_0 DECODE already ~ties llama.cpp/MLX at the ~120 GB/s memory-bandwidth wall** (batch-1 decode is bandwidth-bound, not compute-bound). The gap table reflects this: the Q8_0 decode matvec (`q8_0_block_linear_row_ksplit_f32y_wire`) is **HAVE**, not WORSE. The likely *real* opportunities are elsewhere: **prefill/GEMM**, **flash-attn at depth**, and **command-buffer / dispatch batching** — not raw decode matvec.

**(c) Losslessness contract is non-negotiable.** Bit-exact greedy parity is the campaign's hard gate. Any kernel that **re-associates reductions** — notably coalesced-KV / multi-workgroup flash-attn that splits the softmax+V accumulation across workers and merges (llama.cpp `kernel_flash_attn_ext_vec_reduce`, NWG>1 vec batching, and any coalesced-KV variant) — is **off-limits even if Phase 1 shows it faster**. Such llama.cpp kernels are flagged **faster-but-parity-unsafe**. Camelid's own split-K decode-attn (`attention_decode_splitk_*` + `_merge`) must hold bit-exact parity to remain in scope; if a future split introduces re-association, it is subject to the same bar.

**(d) Host GPU family (M4): the GEMM competitor is the simdgroup fallback, not the tensor path.** This M4 reports `has_tensor = false` — `[device supportsFamily:MTLGPUFamilyMetal4]` is false pre-M5/A19, and `llama-bench` prints *"ggml_metal_device_init: tensor API disabled for pre-M5 and pre-A19 devices"*. So llama.cpp's `kernel_mul_mm` runs its **`simdgroup_float8x8` fallback** (NR0=64 × NR1=32 × NK=32, ~8 KB SRAM) on this host; the cooperative-tensor `matmul2d` path (`GGML_METAL_HAS_TENSOR`, 128×64×32) is dead code here. **Consequence:** the Phase-1 GEMM comparison on M4 is simdgroup-matrix (llama) vs simdgroup-matrix (Camelid `steel_q8_mm`), and any GEMM result here does **not** generalize to M5+/A19+ (where llama gains the tensor-core path Camelid would also need). Re-baseline GEMM on M5+ separately.

---

## 7. Phase 1 entry point

Derived from the gap table, the roofline should attack these in order. None carry a speed claim — each is a **suspected** gap to be measured first.

1. **Q8_0 prefill GEMM** — `q8_0_block_wire_mm` / `steel_q8_mm{,_dual}` / `kernel_mul_mm_q8_0_f32` vs llama.cpp `kernel_mul_mm` (**on M4: the `simdgroup_float8x8` fallback — the tensor-core `matmul2d` is M5+/A19+ only, see §6d**). **Bottleneck class: compute-bound** (prefill GEMM is FLOP-bound at the ~3.4 TFLOPS Q8 GEMM envelope). *Suspected WORSE; measure GEMM achieved TFLOPS first.* This is the highest-value lever on M4 since decode is already at the wall.

2. **Command-buffer / dispatch batching** — Camelid's one-encoder-per-token + encode-ahead vs llama.cpp's 8+1 command-buffer split with per-op buffer binding and barrier-serialized concurrent encoders. **Bottleneck class: dispatch-overhead-bound.** Measure CPU encode time and inter-kernel GPU gaps via `GPUStartTime`/`GPUEndTime`; this is the most structurally divergent axis and the cheapest to instrument without Xcode.

3. **Flash-attn at depth (prefill + long-context decode)** — Camelid `attention_prefill_v3_f32` and `attention_decode_splitk_f32` at large position counts vs llama.cpp tiled `kernel_flash_attn_ext` / `_vec` with head-dim specialization and the `_blk` mask-skip classifier. **Bottleneck class: memory-bound at depth** (KV traffic dominates as context grows; the missing mask-block-skip classifier is the one candidate structural win — and only where masking yields skippable blocks). Any speedup must respect the §6c parity bar (no reduction re-association).

Out of scope for Phase 1 attack (confirmed HAVE / at-the-wall): batch-1 Q8_0 decode matvec (memory-bound, already ties), rope, RMS norm, residual/activation elementwise, greedy argmax. Out of scope as MISSING (N/A): all K-quant / IQ-* / MXFP4 / BF16 breadth, MoE (`mul_mv_id` / `mul_mm_id`), SSM, im2col — these are not real gaps under Camelid's Q8_0/Q4_0-only charter.