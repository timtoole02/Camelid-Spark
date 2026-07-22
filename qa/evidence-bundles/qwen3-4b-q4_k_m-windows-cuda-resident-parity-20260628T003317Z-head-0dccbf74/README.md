# Qwen3 4B Q4_K_M — Windows CUDA GPU-resident ChatML parity bundle

Exact row: `Qwen3 4B Instruct Q4_K_M` (`Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf`)
SHA256: `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5` (2,497,280,256 B)
Platform: Windows x86_64 (MSVC). GPU: RTX 3060 Laptop GPU (6 GB, compute 8.6, driver 576.83).
Source/runtime head: `0dccbf74` (release, lto=fat, codegen-units=1, CUDA default-on).

This is the **first Q4_K_M (mixed Q4_K + Q6_K) parity certification** for a mainstream LLM on
camelid — Phase 1 of the K-quant decode conductor. The hard kernels (`q4k_gemv`, `q6k_gemv`
with the 8-lane f32 parity anchor) already existed; this bundle is the end-to-end greedy
certificate that they are token-identical to llama.cpp on a full mixed model.

## Result

`all_pass = true` — the GPU-resident CUDA decode is token-AND-text-identical to the llama.cpp
`acd79d6` comparator at [1, 5, 50] generated tokens for all 3 chat probes (capital-of-France,
primary-color, say-hello), thinking-DISABLED ChatML, greedy. Cross-engine prompt-token parity
also passes. (Note: the "primary color" probe — a documented near-tie excluded from the Q8_0 4B
headline set — passes here at all token counts.)

The model's mix exercises **both** K-quant kernels in one run: `attn_v`, `ffn_down`, and the
tied `token_embd`/lm_head are Q6_K (`q6k_gemv`); q/k/o/gate/up are Q4_K (`q4k_gemv`).

## Comparator

llama.cpp 9632 (`acd79d603`) — confirmed: `local llama.cpp` HEAD == `acd79d6`.
Driven via `/completion` (ChatML specials parsed) with `-ngl 0 -ctk f32 -ctv f32 -fa off
--no-repack`, temperature 0, top_k 1, seed 0, `cache_prompt:false`, `return_tokens:true`.
**CPU-only**: this box's llama.cpp build ships no `ggml-cuda.dll`.

## Proof chain (differs from the Q8_0 bundles — read this)

Q8_0 CUDA-resident bundles prove `GPU == camelid cpu_reference == llama.cpp`. **The middle leg
does not exist for Q4_K_M**: K-quant linears load **wire-only** (`load_kquant_wire_linear`,
empty f32 `data`) for the resident GPU engine, and there is **no camelid CPU K-quant decode
path yet** (that is Phase 2). With CUDA hidden the model errors (`no-row-major-data ...
data_len=0`). So the proof here is **camelid GPU-resident CUDA decode == llama.cpp directly**.
That same CPU-path error is positive evidence the passing run was genuinely on the GPU.

## Disclosure-labeling gap (follow-up, not a defect)

camelid `/api/capabilities` mislabels this lane: `selected_backend=cpu_reference`,
`decode_path=safe_cpu_decode`, `quant_type=dense_or_other` ("non-validated row or quant;
failing closed to safe path") — the planner is Q8_0-centric and doesn't recognize Q4_K/Q6_K.
But `cuda_resident_active=true` and the runtime ran GPU-resident (36/36 layers in VRAM, coherent
output, CPU-path-errors-when-hidden). Output is parity-green; only the self-disclosure string is
wrong. Recorded in `capabilities.json` → `followups`.

## Speed (honest framing — NOT head-to-head)

- camelid Q4_K_M GPU-resident decode: **median 19.44 tok/s** (3 iters, warmup), peak **4.92 GB**
  (fully resident in 6 GiB).
- llama.cpp Q4_K_M **CPU**: tg128 **12.35 ± 0.10 tok/s**, pp512 66.87 tok/s (8 threads).

Different backends — do **not** read these as a ratio. A GPU-vs-GPU llama.cpp comparison needs a
CUDA llama build (absent). A same-model Q8 vs Q4_K_M GPU comparison needs Qwen3-4B-Q8_0 (not on
disk). On GPU the realized prize now is the VRAM footprint; the decode-speed prize is
bandwidth-bound and lands on the CPU lane in Phase 2. See `qwen3-4b-q4_k_m-cuda-resident-speed.json`.

## Artifacts

- `qwen3-4b-q4_k_m-windows-cuda-resident-parity.json` — the chat-parity result (schema
  `camelid.qwen3.chatml_chat_parity.v1`, variant `kquant_gpu_resident`).
- `qwen3-4b-q4_k_m-cuda-resident-speed.json` — speed receipt.
- `capabilities.json` — static planner output + observed runtime + follow-ups.
- `manifest.json` — full provenance.
- Harness: `scripts/chat-parity-qwen3-kquant.mjs` (K-quant variant; see its header for the one
  deliberate change vs `chat-parity-qwen3.mjs`).

## Reproduce

```
# GPU-resident camelid:
camelid serve --addr 127.0.0.1:8185 --model Qwen3-4B-Q4_K_M.gguf --no-open
# CPU llama.cpp reference:
llama-server -m Qwen3-4B-Q4_K_M.gguf --port 8090 -ngl 0 -ctk f32 -ctv f32 -fa off --no-repack -c 4096
# Harness:
node scripts/chat-parity-qwen3-kquant.mjs --camelid http://127.0.0.1:8185 \
  --llama http://127.0.0.1:8090 --model-id "Qwen3 4B Instruct Awq" \
  --row-id qwen3_4b_instruct_q4_k_m --display-name "Qwen3 4B Instruct Q4_K_M" --out <this>.json
```
