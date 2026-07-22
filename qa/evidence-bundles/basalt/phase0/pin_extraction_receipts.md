# NVFP4 wire-spec extraction — raw receipts

Pin: llama.cpp @ `<llama.cpp>`

## 0. Pin checkout verification

```
$ git -C <llama.cpp> log -1 --format='%H %ad %s'
acd79d603cb2e1c84c0886137b80f1ad649b6857 Sun Jun 14 15:07:31 2026 +0200 jinja : add count/d/e filter aliases (#24606)
$ git status --porcelain   # (empty — clean tree)
```

Matches the expected pin `acd79d603` (build 9632). Read-only; nothing edited on this checkout.

---

## 1. Type registration

### ggml/include/ggml.h:429-432 (enum ggml_type)
```
        GGML_TYPE_MXFP4   = 39, // MXFP4 (1 block)
        GGML_TYPE_NVFP4   = 40, // NVFP4 (4 blocks, E4M3 scale)
        GGML_TYPE_Q1_0    = 41,
        GGML_TYPE_COUNT   = 42,
```

### ggml/include/ggml.h:473-475 (enum ggml_ftype)
```
        GGML_FTYPE_MOSTLY_MXFP4   = 25, // except 1d tensors
        GGML_FTYPE_MOSTLY_NVFP4   = 26, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q1_0    = 27, // except 1d tensors
```

### include/llama.h:154-157 (enum llama_ftype)
```
        LLAMA_FTYPE_MOSTLY_MXFP4_MOE     = 38, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_NVFP4         = 39, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q1_0          = 40, // except 1d tensors
```

### ggml/src/ggml.c:744-751 (type_traits row)
```
    [GGML_TYPE_NVFP4] = {
        .type_name                = "nvfp4",
        .blck_size                = QK_NVFP4,          // 64
        .type_size                = sizeof(block_nvfp4), // 36
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_nvfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_row_nvfp4_ref,
    },
```

### ggml/src/ggml-cpu/ggml-cpu.c:283-288 (CPU type_traits_cpu row)
```
    [GGML_TYPE_NVFP4] = {
        .from_float               = quantize_row_nvfp4,
        .vec_dot                  = ggml_vec_dot_nvfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
```

### ggml/src/ggml.c:1416-1417 (ftype -> wtype)
```
        case GGML_FTYPE_MOSTLY_MXFP4:         wtype = GGML_TYPE_MXFP4; break;
        case GGML_FTYPE_MOSTLY_NVFP4:         wtype = GGML_TYPE_NVFP4; break;
```

### ggml/src/ggml.c:7741 (ggml_quantize_chunk dispatch)
```
        case GGML_TYPE_NVFP4:   result = quantize_nvfp4  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
```

### src/llama-model-loader.cpp:46 and :763
```
        case LLAMA_FTYPE_MOSTLY_NVFP4:    return "NVFP4";
...
            case GGML_TYPE_NVFP4:   ftype = LLAMA_FTYPE_MOSTLY_NVFP4;   break;
```

### gguf-py/gguf/constants.py:4415, 4469, 4594
```
    NVFP4   = 40                         # GGMLQuantizationType
    MOSTLY_NVFP4         = 39            # LlamaFileType, "except 1d tensors"
    GGMLQuantizationType.NVFP4:   (64, 4 + 32),   # GGML_QUANT_SIZES = (block_size, type_size=36)
```

---

## 2. Block layout & nibble packing

### ggml/src/ggml-common.h:211-217
```
#define QK_NVFP4 64
#define QK_NVFP4_SUB 16  // sub-block size for per-group scales
typedef struct {
    uint8_t d[QK_NVFP4/QK_NVFP4_SUB]; // UE4M3 scales (4 bytes, one per 16-element sub-block)
    uint8_t qs[QK_NVFP4/2];           // packed 4-bit E2M1 values (32 bytes)
} block_nvfp4;
static_assert(sizeof(block_nvfp4) == sizeof(uint8_t)*(QK_NVFP4/QK_NVFP4_SUB) + QK_NVFP4/2, "wrong nvfp4 block size/padding");
```
=> 36 bytes = 4 scale bytes (`d[0..3]`) followed by 32 qs bytes, one super-block per 64 elements. Field order: **d first, then qs**. Byte 0..3 = UE4M3 sub-block scales; byte 4..35 = packed nibbles.

### quantizer — ggml/src/ggml-quants.c:346-379 (quantize_row_nvfp4_ref)
```
void quantize_row_nvfp4_ref(const float * GGML_RESTRICT x, block_nvfp4 * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK_NVFP4;          // 64
    static const int qk_sub = QK_NVFP4_SUB;  // 16
    static const int n_sub = QK_NVFP4 / QK_NVFP4_SUB;  // 4
    assert(k % qk == 0);
    const int nb = k / qk;
    for (int i = 0; i < nb; i++) {
        for (int s = 0; s < n_sub; s++) {
            const float * xb = x + i*qk + s*qk_sub;
            float amax = 0.0f;
            for (int j = 0; j < qk_sub; j++) {
                if (amax < fabsf(xb[j])) { amax = fabsf(xb[j]); }
            }
            // UE4M3 scale: amax / 6.0 maps the max E2M1 value (6.0) to amax
            const uint8_t ue = ggml_fp32_to_ue4m3(amax / 6.0f);
            y[i].d[s] = ue;
            const float d = ggml_ue4m3_to_fp32(ue);
            for (int j = 0; j < qk_sub/2; ++j) {   // j = 0..7
                const uint8_t x0 = best_index_mxfp4(xb[0        + j], d);
                const uint8_t x1 = best_index_mxfp4(xb[qk_sub/2 + j], d);
                y[i].qs[s*(qk_sub/2) + j] = x0 | (x1 << 4);
            }
        }
    }
}
```
Packing order (per sub-block s, 16 elems -> 8 bytes at qs[s*8 .. s*8+7]):
- **low nibble** of qs byte j = element `xb[j]`   (j in 0..7, i.e. sub-block-local index 0..7)
- **high nibble** of qs byte j = element `xb[8+j]` (sub-block-local index 8..15)
So the 16 sub-block elements are split half/half: first 8 in low nibbles, second 8 in high nibbles of the same 8 bytes. NOTE: `amax/6.0` (NOT amax/max_value combined with `d`); the two nibbles use `best_index_mxfp4` = nearest-value search over the shared `kvalues_mxfp4` LUT with scale `d`.

### dequant — ggml/src/ggml-quants.c:531-554 (dequantize_row_nvfp4)
```
void dequantize_row_nvfp4(const block_nvfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    ... n_sub = 4 ...
    for (int i = 0; i < nb; i++) {
        for (int s = 0; s < n_sub; s++) {
            const float d = ggml_ue4m3_to_fp32(x[i].d[s]);
            float * yb = y + i*qk + s*qk_sub;
            for (int j = 0; j < qk_sub/2; ++j) {
                const int8_t v0 = kvalues_mxfp4[x[i].qs[s*(qk_sub/2) + j] & 0x0F];
                const int8_t v1 = kvalues_mxfp4[x[i].qs[s*(qk_sub/2) + j] >>   4];
                yb[j + 0       ] = v0*d;
                yb[j + qk_sub/2] = v1*d;
            }
        }
    }
}
```
Confirms low nibble -> yb[j] (local 0..7), high nibble -> yb[j+8] (local 8..15). Value = kvalues_mxfp4[nibble] * d, where d is the UE4M3-decoded sub-block scale (already includes the 0.5 convention factor — see §3).

---

## 3. Scale format — UE4M3

### E2M1 value LUT — ggml/src/ggml-common.h:1114-1118
```
// e2m1 values (doubled)
// ref: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
GGML_TABLE_BEGIN(int8_t, kvalues_mxfp4, 16)
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
GGML_TABLE_END()
```
Values are **2x the true E2M1 magnitudes** (true = 0,.5,1,1.5,2,3,4,6). The 0.5 factor lives inside `ggml_ue4m3_to_fp32` so `kvalues[i] * d` gives the correct product.

### UE4M3 decode — ggml/src/ggml-impl.h:500-515
```
// UE4M3: unsigned, 4 exp bits (bias=7), 3 mantissa bits
// Returns value * 0.5 to match kvalues_mxfp4 convention (kvalues = 2 * E2M1_float)
static inline float ggml_ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || x == 0x7F) {
        return 0.0f;
    }
    int   exp = (x >> 3) & 0xF;
    int   man = x & 0x7;
    float raw;
    if (exp == 0) {
        raw = ldexpf((float) man, -9);                 // subnormal: man * 2^-9
    } else {
        raw = ldexpf(1.0f + (float) man / 8.0f, exp - 7);  // normal: (1+man/8) * 2^(exp-7)
    }
    return raw * 0.5f;
}
```
- **Unsigned** (no sign bit; the 8th bit is unused/ignored on decode except in the NaN check). Byte layout: bit7 unused, bits6-3 = exp (4), bits2-0 = man (3).
- **bias = 7**.
- byte `0x00` -> 0.0 (zero). byte `0x7F` -> 0.0 (treated as NaN sentinel -> flushed to 0.0).
- Max finite: exp=14 (0xE), man=7 -> (1+7/8)*2^7 = 1.875*128 = 240; *0.5 = **120.0** decoded. Encoder clamps at code 0x7E (exp=15 is reserved: encoder never emits exp>=15, returns 0x7E). See encoder below.
- Note the block_nvfp4 comment calls it "E4M3 scale" but the runtime treats it as **unsigned** E4M3 (top bit stripped). Conversion strips the sign bit explicitly: `& 0x7F` (base.py:668).

### UE4M3 encode — ggml/src/ggml-impl.h:517-553
```
static inline uint8_t ggml_fp32_to_ue4m3(float x) {
    if (!(x > 0.0f)) { return 0; }         // <=0 or NaN -> 0
    if (x > 448.0f) { x = 448.0f; }        // clamp input domain
    uint32_t bits; memcpy(&bits, &x, 4);
    int fp32_exp  = ((bits >> 23) & 0xFF) - 127;
    int fp32_man  = (bits >> 20) & 0x7;    // top 3 mantissa bits
    int ue4m3_exp = fp32_exp + 7;
    if (ue4m3_exp <= 0) {                   // subnormal
        int man = (int) (x * 512.0f + 0.5f);   // round-half-up
        if (man > 7) { man = 7; }
        if (man < 1) { return 0; }
        return (uint8_t) man;
    }
    if (ue4m3_exp >= 15) { return 0x7E; }   // saturate to max finite code
    int round_bit = (bits >> 19) & 1;       // round-half-up on 4th mantissa bit
    int ue4m3_man = fp32_man + round_bit;
    if (ue4m3_man > 7) {
        ue4m3_man = 0; ue4m3_exp++;
        if (ue4m3_exp >= 15) { return 0x7E; }
    }
    return (uint8_t) ((ue4m3_exp << 3) | ue4m3_man);
}
```
- Rounding of the SCALE byte: round-half-up (`+round_bit` where round_bit = bit19, with carry into exponent). Subnormal path uses `+0.5` truncation (round-half-up).
- Domain clamp 448.0 on input; output saturates to `0x7E` (never emits exp==15 / 0x7F). So the encoder's max emitted decoded scale = decode(0x7E) = (1+6/8)*2^7*0.5 = 1.75*128*0.5 = 112.0.
- Input `amax/6.0` is what's fed (quantize_row_nvfp4_ref:367), so a sub-block whose amax = 6*112 = 672 would saturate.

### CUDA decode/encode mirror — ggml/src/ggml-cuda/common.cuh:830-869
`ggml_cuda_ue4m3_to_fp32` (830) uses hardware `__nv_fp8_e4m3` when FP8_AVAILABLE (divides by 2 to match the 0.5 convention), else the same scalar path as CPU; NaN bytes `0x7F`/`0xFF` -> 0.0. `ggml_cuda_fp32_to_ue4m3` (859) is Blackwell-only (BLACKWELL_MMA_AVAILABLE) and used only for activation sub-block scales.

### gguf-py mirror — gguf-py/gguf/quants.py:712-747 (NVFP4.ue4m3_to_fp32 / fp32_to_ue4m3) — same math, cross-checks the C.

### Validation — ggml/src/ggml-quants.c:5488-5493
```
        case GGML_TYPE_NVFP4:
            {
                // UE4M3 scales are uint8_t — all byte values are valid
                GGML_UNUSED(data);
                GGML_UNUSED(nb);
            } break;
```
No row-data validation for NVFP4 (every byte is a valid scale; contrast MXFP4 which validates E8M0).

---

## 4. E2M1 element codec + rounding

- Value table: `kvalues_mxfp4` (§3) — shared with MXFP4. 4-bit index: bit3 = sign, bits2-0 = magnitude index into {0,1,2,3,4,6,8,12} (doubled E2M1).
- Quantizer rounding: **LUT nearest-value search** via `best_index_mxfp4`, ggml/src/ggml-quants.c:299-310:
```
static inline int best_index_mxfp4(float x, float e) {
    int best_index = 0;
    float best_err = fabsf(kvalues_mxfp4[0]*e - x);
    for (int i = 1; i < 16; i++) {
        float err = fabsf(kvalues_mxfp4[i]*e - x);
        if (err < best_err) { best_index = i; best_err = err; }
    }
    return best_index;
}
```
Nearest by absolute error, first-wins on ties (strict `<`), scanning indices 0..15 in order. NOT round-nearest-even arithmetic; it is exhaustive LUT search including both signs.

- CUDA element encoder (Blackwell activation path) — ggml/src/ggml-cuda/common.cuh:871-891, `ggml_cuda_float_to_fp4_e2m1`: separate sign bit + positive LUT {0,.5,1,1.5,2,3,4,6} nearest search, `best_i | sign_bit`. Same nearest-search semantics but over the TRUE (un-doubled) magnitudes with an explicit inv-scale multiply.

---

## 5. Per-tensor scale — THE critical question

### Answer: YES, the pin implements a second-level per-tensor (and per-expert) F32 scale for NVFP4 — but as a SEPARATE sidecar tensor multiplied via an ggml_mul node, NOT an in-block field.

The in-block format (block_nvfp4) has **no** per-tensor factor: only 4 UE4M3 sub-block scales + nibbles. But the model carries an additional `.scale` (weight_scale_2) tensor per weight, plus an `.input_scale`, applied at graph-build time.

### Sidecar scale tensors created — src/llama-model.cpp:1317-1477
```
1317  // generic pass: load optional per-tensor/per-expert ".scale" tensors (e.g. NVFP4 scale2)
...
1324      layer.wq_s = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "scale", i), {1}, TENSOR_NOT_REQUIRED);
...  (wk_s, wv_s, wo_s, wqkv_s, ffn_gate_s, ffn_down_s, ffn_up_s, shexp variants — all shape {1})
1363      layer.ffn_gate_exps_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "scale", i), {n_expert}, ...); // per-expert
1366      layer.ffn_down_exps_s = ...{n_expert}...
1369      layer.ffn_up_exps_s   = ...{n_expert}...
1394      layer.wq_in_s = create_tensor(tn(LLM_TENSOR_ATTN_Q, "input_scale", i), {1}, ...);  // activation input scales
...
1459      if (output && output->type == GGML_TYPE_NVFP4) {
1462          output_s    = create_tensor(tn(LLM_TENSOR_OUTPUT, "scale"), {1}, TENSOR_NOT_REQUIRED);
1466          output_in_s = create_tensor(tn(LLM_TENSOR_OUTPUT, "input_scale"), {1}, TENSOR_NOT_REQUIRED);
1474  GGML_ASSERT(!(output && tok_embd && strcmp(output->name, tok_embd->name)==0 &&
                    output->type == GGML_TYPE_NVFP4 && (output_s || output_in_s)));
```
The `.scale` fields are all `TENSOR_NOT_REQUIRED` — a plain NVFP4 GGUF without them still loads (scales default to nullptr -> no ggml_mul). They exist so ModelOpt/compressed-tensors NVFP4 checkpoints (which DO carry a per-tensor weight_scale_2) reproduce the reference math.

### Application at inference — src/llama-graph.cpp:1085-1114 (build_lora_mm)
```
ggml_tensor * llm_graph_context::build_lora_mm(ggml_tensor * w, ggml_tensor * cur, ggml_tensor * w_s) const {
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);
    ... (lora adds) ...
    if (w_s) {
        res = ggml_mul(ctx0, res, w_s);   // <-- per-tensor F32 scale applied to the mul_mat output
    }
    return res;
}
```
Per-expert scale2 applied similarly at llama-graph.cpp:1638-1771 (up_exps_s/gate_exps_s/down_exps_s ggml_mul on MoE outputs). Model struct fields: src/llama-model.h:429-447 (`wq_in_s`..`ssm_beta_in_s`), :548-549 (`output_s`, `output_in_s`), :309/:547 comments naming them "NVFP4 per-tensor scale2 / input_scale".

### Where scale2 comes from (conversion) — conversion/base.py:653-686
```
654  def _nvfp4_pack(weight, scale):
656      """... Preserves original E4M3 scale bits as UE4M3 (strip sign bit).
657      The per-tensor scale2 factor is stored as a separate tensor and applied at inference time via ggml_mul()."""
668      d_ue = scale.view(torch.uint8).numpy()... & 0x7F     # sub-block UE4M3 = original E4M3 bits, sign stripped
...
678  def _repack_nvfp4(self, name, weight, scale, scale2, input_scale):
683      self.gguf_writer.add_tensor(new_name, raw, raw_dtype=NVFP4)
685      self._write_scale_tensor(new_name.replace(".weight", ".scale"), scale2)          # per-tensor F32
686      self._write_scale_tensor(new_name.replace(".weight", ".input_scale"), input_scale)
```
So the NVFP4 wire model = { NVFP4 weight block (4x UE4M3 + nibbles) } + optional { F32 `.scale` (scale2), F32 `.input_scale` } sidecar tensors. scale2 is NOT folded into the UE4M3 sub-block scales — the sub-block bits are copied verbatim from the source checkpoint (bit-preserving repack), and scale2 is applied downstream by ggml_mul.

### Contrast for the "naive NVIDIA recipe":
- The block is 16-elem sub-block scaled, but the ggml super-block bundles **4** sub-blocks (64 elems) into one struct.
- Sub-block scale is **unsigned** E4M3 (UE4M3, top bit dropped), NOT signed E4M3.
- The per-tensor F32 (weight_scale_2) DOES exist but lives in a separate sidecar tensor + ggml_mul, not in-block and not in GGUF KV metadata.

### Definitiveness: NVFP4 as a llama-quantize target does NOT exist.
`llama_ftype_get_default_type` (src/llama-quant.cpp:792-833) has **no** `LLAMA_FTYPE_MOSTLY_NVFP4` case -> falls to `default: return GGML_TYPE_COUNT` -> quantize entrypoint throws "invalid output file type" (src/llama-quant.cpp:866-868). Grep of `tools/` for NVFP4/nvfp4 => **no matches** (no quantize CLI option). Therefore NVFP4 GGUFs are produced ONLY by the Python conversion repack path (ModelOpt / compressed-tensors "nvfp4-pack-quantized"), never by `llama-quantize`. The C `quantize_row_nvfp4_ref` / `quantize_nvfp4` exist and are reachable via `ggml_quantize_chunk` (used by tests and any direct ggml caller), but no product ftype routes to them.

---

## 6. Quantizer per-tensor type rules (moot for NVFP4 target, documented for completeness)

Because NVFP4 is not a default_type (see §5), the NVFP4-specific branches of `llama_tensor_get_type_impl` never fire from a MOSTLY_NVFP4 quantize run. What governs the CONVERSION-side keep set:

- Norms, biases, gating, tiny tensors kept unquantized — `tensor_allows_quantization` (src/llama-quant.cpp:288-355): only tensors whose name ends in "weight", 2D+, excluding `_norm.weight`, `ffn_gate_inp.weight`, `output.weight` (unless requested), `altup`/`laurel`/`per_layer_model_proj`, pos-embd, token-types, ssm_conv1d, RWKV time_mix small weights, multimodal (position_embd/patch_embd/etc). In the conversion path the analogous effect: only tensors that had a 2D `weight_scale` in the source checkpoint are repacked to NVFP4 (conversion/base.py:697-711, `if scale.ndim < 2: continue`); everything else stays BF16/F16/F32 (or FP8 -> dequant -> Q8_0/F16).
- Token-embd / output tie handling: src/llama-model.cpp:1472-1477 asserts a tied NVFP4 output is only valid when NO sidecar LM-head scales are present; if `output_s`/`output_in_s` exist, output must be an independent tensor.
- **K % 64 fallback: there is NONE for NVFP4.** `tensor_type_fallback` (src/llama-quant.cpp:362-408) has cases only for IQ*/Q2_K/Q3_K/TQ*/Q4_K/Q5_K/Q6_K; the `default:` throws `"no tensor type fallback is defined for type %s"` (line 391). So an NVFP4 target with K not divisible by 64 would throw, not fall back — but this is unreachable because NVFP4 is never a quantize target. On the conversion side, `_nvfp4_pack` (base.py:672) does `n_super = n_blocks // 4` (n_blocks = K/16), silently requiring K % 64 == 0; the gguf-py generic path `quant_shape_to_byte_shape` (quants.py:15-18) raises ValueError if `shape[-1] % block_size(64) != 0`.

### gemma4 special-casing:
- **None in src/llama-quant.cpp** (grep gemma/gemma4/GEMMA => no matches). All gemma4 NVFP4 special handling is in the CONVERSION layer:
  - conversion/gemma.py:719-741 `_generate_nvfp4_tensors`: folds the per-layer `router.per_expert_scale` ([n_expert]) into each expert's `down_proj.weight_scale_2` (equivalent to a per-expert scalar on the down_proj output = where `ffn_down_exps_s` is applied at inference), then calls super().
  - conversion/gemma.py:743-752 `filter_tensors`: renames `per_dim_scale`/`layer_scalar` and expert aux tensors to end with `.weight` so they route correctly.
- Other arch transforms: conversion/llama.py:169-181 `_repack_nvfp4` notes the NVFP4 path bypasses the BF16 Q/K RoPE permutation site; conversion/qwen.py:378-467 `_transform_nvfp4_weight` applies head/column permutations for linear-attn projections before repack.

---

## 7. CPU compute path

- vec_dot name: `ggml_vec_dot_nvfp4_q8_0`, declared dot type `GGML_TYPE_Q8_0` (ggml-cpu.c:285-286). The NVFP4 super-block (64 elems) is dotted against **2** q8_0 blocks (each 32 elems).
- Scalar/generic: `ggml_vec_dot_nvfp4_q8_0_generic` — ggml/src/ggml-cpu/quants.c:278-312.
```
278  // NVFP4: super-block of 64 elements = 4 sub-blocks of 16 = 2 q8_0 blocks
279  void ggml_vec_dot_nvfp4_q8_0_generic(...) {
...
294      for (int ib = 0; ib < nb; ++ib) {
295          for (int s_idx = 0; s_idx < 4; ++s_idx) {
296              const float d = ggml_ue4m3_to_fp32(x[ib].d[s_idx]);
297              const int q8_block = s_idx / 2;               // sub-blocks 0,1 -> q8[2ib+0]; 2,3 -> q8[2ib+1]
298              const int q8_off   = (s_idx % 2) * QK_NVFP4_SUB;
299              const float dy = GGML_CPU_FP16_TO_FP32(y[2*ib + q8_block].d);
301              int sumi_lo = 0, sumi_hi = 0;
302              for (int j = 0; j < QK_NVFP4_SUB/2; ++j) {
303                  const uint8_t qv = x[ib].qs[s_idx*(QK_NVFP4_SUB/2) + j];
304                  sumi_lo += y[...].qs[q8_off + j + 0]              * kvalues_mxfp4[qv & 0xf];
305                  sumi_hi += y[...].qs[q8_off + j + QK_NVFP4_SUB/2] * kvalues_mxfp4[qv >> 4];
306              }
308              sumf += dy * d * (sumi_lo + sumi_hi);
```
- x86 (SSE/AVX2/AVX512): **no dedicated kernel** — arch-fallback.h:83-85 aliases `ggml_vec_dot_nvfp4_q8_0_generic` to `ggml_vec_dot_nvfp4_q8_0` under the x86 branch, i.e. x86 uses the SCALAR path. (Same fallback aliasing for PPC/LoongArch/RISCV/s390x/wasm.)
- ARM/aarch64: real NEON kernel in ggml/src/ggml-cpu/arch/arm/quants.c:736-844 (`ggml_vec_dot_nvfp4_q8_0`), with `__ARM_FEATURE_DOTPROD` and non-dotprod (`ggml_nvfp4_dot8`, defined ggml-cpu-impl.h:323) variants; falls through to an inline scalar loop (826-841) when NEON not present.
- No AVX2/AVX512 NVFP4 vec_dot exists in this pin.

---

## 8. CUDA paths

### Dequant / convert — ggml/src/ggml-cuda/convert.cu:620-658
`dequantize_block_nvfp4` (620) + `dequantize_row_nvfp4_cuda` (649). 32 threads/block, one super-block per block; `sub = tid/(16/2)=tid/8`, `j = tid%8`; `yy[y0]=cast(d*kvalues_mxfp4[q&0x0F])`, `yy[y1]=cast(d*kvalues_mxfp4[q>>4])`, y1=y0+8. Mirrors CPU exactly.

### MMVQ (vec-dot, decode/gemv) — PRESENT, all NVIDIA archs with dp4a
- `vec_dot_nvfp4_q8_1` — ggml/src/ggml-cuda/vecdotq.cuh:331-359, `VDR_NVFP4_Q8_1_MMVQ 4` (line 328). Uses `get_int_from_table_16(..., kvalues_mxfp4)` + `ggml_cuda_dp4a`; scale `ggml_cuda_ue4m3_to_fp32(bq4->d[is]) * __low2float(bq8->ds)`.
- Registered: mmvq.cu:19 (`vec_dot_nvfp4_q8_1`), :47 (VDR), :124 (get_vdr), :145 (nvfp4 mmvq max batch = 8), :232, :1027-1031 (`mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>`).

### MMQ (mul_mat_q) — PRESENT, two code paths
- Dispatch: mmq.cu:29-31 (`mul_mat_q_case<GGML_TYPE_NVFP4>`), :282 (supported-type list), :4149 (`extern DECL_MMQ_CASE(GGML_TYPE_NVFP4)`), template-instances/mmq-instance-nvfp4.cu.
- Type traits — mmq.cuh:3329-3340:
  - Blackwell (`BLACKWELL_MMA_AVAILABLE`): `load_tiles_nvfp4_nvfp4` + `vec_dot_fp4_fp4_mma<GGML_TYPE_NVFP4>` (native FP4 tensor-core MMA).
  - Non-Blackwell: `load_tiles_nvfp4` + `vec_dot_q8_0_16_q8_1_mma` (dequant-to-q8 MMA).
  - dp4a fallback (all): `vec_dot_q8_0_16_q8_1_dp4a`.
  So MMQ works on non-Blackwell via the q8_0_16 path; Blackwell gets native FP4.

### Blackwell / sm_120 gating
- `BLACKWELL_MMA_AVAILABLE` defined only for `__CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL && < GGML_CUDA_CC_RUBIN` (common.cuh:286-288); host predicate `blackwell_mma_available(cc)` (common.cuh:360-363).
- Native FP4 quantize of the activations: `quantize_mmq_fp4` kernel (quantize.cu:79-171) is `#if defined(BLACKWELL_MMA_AVAILABLE) ... #else NO_DEVICE_CODE`. Uses `ggml_cuda_fp32_to_ue4m3` (Blackwell-only) + +/-2 code search for min-error sub-block scale.
- Native MMA PTX — mma.cuh:1125-1154 `mma_block_scaled_fp4`: NVFP4 emits
  `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3` (MXFP4 uses `kind::mxf4 ... scale_vec::2X ... ue8m0`). scale_vec::4X = 4 UE4M3 scales/64-wide k = one per 16-elem sub-block.
- Native path gate at call site: mmq.cu:125 `use_native_fp4 = blackwell_mma_available(cc) && (src0->type==MXFP4 || NVFP4)`; `block_fp4_mmq` struct (mmq.cuh:49-54) = `uint32_t d4[4]; int8_t qs[128]` = 256 values, `sizeof==4*sizeof(block_q8_1)` (static_assert mmq.cu:137).
- MMQ activation-side interleave note mmq.cuh:51: "nvfp4 has block size 16, each int32 of d4 contains 4 ue4m3 scales".

### Op-support / force-to-q8 path
- ggml-cuda.cu:1694-1699: `use_fp16` explicitly excludes NVFP4 (`src0->type != GGML_TYPE_NVFP4`) — NVFP4 never routes through the F16 mul_mat path.
- ggml-cuda.cu:5211: NVFP4 listed as a supported mul_mat / mul_mat_id src0 type (returns true).

### Missing CUDA pieces for NVFP4
- No CUDA `from_float`/quantizer that writes the STORED block_nvfp4 layout (only the MMQ activation quantizer `quantize_mmq_fp4` which writes the transient block_fp4_mmq, Blackwell-only). Weight NVFP4 blocks are produced host-side (Python).
- Non-Blackwell has no native FP4 tensor cores; uses dequant+dp4a/q8 MMA (functional, slower).

### Other backends (noted, not the CPU/CUDA focus)
Vulkan: dequant_nvfp4.comp + types.glsl ue4m3_fp32_lut[128]. SYCL: dequantize.hpp:308/1493, vecdotq.hpp:915, mmvq.cpp. Metal: ggml-metal-device.m references. ARM: covered in §7.

---

## 9. Divergences from the naive NVIDIA-recipe & upstream #22042

(a) vs naive "16-elem block + 1 signed-E4M3 scale byte + per-tensor F32 scale":
1. Storage granularity: ggml bundles **4** 16-elem sub-blocks into a 64-elem super-block struct (block_nvfp4, 36 bytes). Scale granularity is still per-16 (4 UE4M3 bytes) — matches — but the container is 64-wide.
2. Sub-block scale is **UNSIGNED** E4M3 (UE4M3, bit7 stripped, `& 0x7F`), not signed E4M3. NaN sentinel `0x7F` (and `0xFF`) decode to 0.0; `0x00` = zero. bias 7, 3-bit mantissa, subnormals via man*2^-9.
3. The E2M1 element LUT is stored **doubled** (kvalues_mxfp4 = 2x true values); the 0.5 correction lives in the UE4M3 decode, so product `kvalues*d` is correct.
4. Per-tensor F32 scale (weight_scale_2) EXISTS but as a **separate sidecar tensor** (`.scale`, shape {1} or {n_expert}) applied by an `ggml_mul` node after mul_mat (llama-graph.cpp:1109-1110), plus a separate `.input_scale` for activations — NOT an in-block field and NOT GGUF KV metadata. A bare NVFP4 GGUF may omit them (TENSOR_NOT_REQUIRED).
5. Quantizer scale target is `amax/6.0` (maps the max E2M1 magnitude 6.0 to amax); no tensor-level factor folded into sub-block scales — sub-block bits are bit-copied from the source checkpoint on repack.
6. Element quantization is exhaustive nearest-LUT search (best_index_mxfp4), first-wins ties — not IEEE round-nearest-even arithmetic.
7. NVFP4 is **not a llama-quantize output type** (no default_type, no CLI option, no K%64 fallback). Born only from Python conversion of ModelOpt/compressed-tensors NVFP4 checkpoints.
8. Nibble packing: within an 8-byte sub-block, low nibbles hold local elems 0..7, high nibbles hold local elems 8..15 (half/half split), per sub-block laid out contiguously (sub 0 -> qs[0..7], sub 1 -> qs[8..15], ...). This is the MXFP4-style low/high split, NOT a linear 0,1 / 2,3 pairing.

(b) upstream discussion #22042: grep of the whole tree for "22042" => the ONLY hit is an unrelated data byte inside gguf-py/gguf/quants.py:1076 (IQ grid hex table). **No source comment in this pin references discussion #22042.** The only design citation in the NVFP4 code is the OCP MX v1.0 spec URL at ggml-common.h:1115.
