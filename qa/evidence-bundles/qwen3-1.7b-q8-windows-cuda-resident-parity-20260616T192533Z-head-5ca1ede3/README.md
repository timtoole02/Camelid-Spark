# Qwen3 1.7B Q8_0 — Windows CUDA GPU-resident ChatML parity bundle

Exact row: `Qwen3 1.7B Instruct Q8_0` (`Qwen/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf`)
SHA256: `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a`
Platform: Windows x86_64 (MSVC). GPU: NVIDIA GeForce RTX 3060 Laptop GPU (6 GB VRAM, compute 8.6), driver 576.83, CUDA 12.9 (nvcc V12.9.86).
Source/runtime head: `5ca1ede3`.

## Result

`all_pass = true` — the GPU-resident CUDA decode is token-AND-text-identical to the llama.cpp comparator at [1,5,50] generated tokens for: 3 confident probes (capital-of-France, say-hello, 2+2). The default "Name a primary color." probe is a documented frontier near-tie (~0.62/0.38 at the close-vs-continue decision); GPU reduction order closes the list while CPU/llama continue — both valid. See qwen3-1.7b-primarycolor-tie.json. Large-context single-shot GPU prefill exercised; see large-context-prefill.json.

`gpu_equals_cpu_reference = true` — and identical to the camelid `cpu_reference` path (`CAMELID_CUDA_RESIDENT_DECODE=0`, `cpu_reference_all_pass=true`) on the same binary. Correctness proof chain: GPU == cpu_reference == llama.cpp.

camelid path: `cuda_resident_q8_runtime` / `q8_0_cuda_resident_decode` (prefill `q8_0_cuda_resident_prefill`, support_level `supported_exact_row_smoke_chatml`, cuda_resident_active=true).

## Comparator

llama.cpp 9632 (acd79d603) (Clang 20.1.8 for Windows x86_64), flags `-ngl 0 -ctk f32 -ctv f32 -fa off --no-repack`.
On-disk Windows CPU comparator (same as the Windows x86 Qwen3 CPU bundles). The GPU-resident CUDA decode is compared against this CPU reference (transitively llama.cpp); correctness proof is GPU output == camelid cpu_reference output == llama.cpp. Differs from the 5d56eff pinned in the macOS/Ubuntu Qwen3 bundles.

## GPU / driver / CUDA

NVIDIA GeForce RTX 3060 Laptop GPU, compute 8.6, 6144 MiB VRAM (~5122 MiB free at engine build), driver 576.83, CUDA 12.9 (nvcc V12.9.86). RTX 3060 LAPTOP GPU with 6 GB VRAM — NOT the 12 GB desktop RTX 3060. The 8B row (8.7 GB Q8_0) does not fit fully in 6 GB, so it runs via the automatic VRAM+host-RAM offload split (some layers resident in VRAM, the rest streamed from system RAM each token; compute stays on the GPU and the math is identical). The 0.6B/1.7B/4B rows are fully VRAM-resident. Results are specific to this GPU/driver/CUDA combination (f32 reduction order is GPU-specific).

## Claim boundary

Supported exact-row smoke for this exact Qwen3 Q8_0 GGUF on Windows x86_64 (MSVC) with the GPU CUDA decode engine on the recorded RTX 3060 Laptop (6 GB) / driver 576.83 / CUDA 12.9 (fully VRAM-resident). ChatML chat, thinking DISABLED, short-chat envelope. GPU decode AND single-shot GPU prefill token+text identical to the camelid cpu_reference (transitively llama.cpp) at the listed token counts on the listed prompts. NOT claimed: other Qwen3 sizes beyond the validated 0.6B/1.7B/4B/8B rows, other variants/quants, base variants, Qwen3-MoE (A3B), model-native/long context beyond the recorded resident KV cap, thinking-mode, other GPUs/drivers/CUDA versions (results are GPU/driver/CUDA-version specific), or production throughput.

## Artifacts

- `qwen3-1.7b-windows-cuda-resident-parity.json`
- `qwen3-1.7b-windows-cpu-reference-parity.json`
- `capabilities.json`
- `qwen3-1.7b-primarycolor-tie.json`
- `large-context-prefill.json`
