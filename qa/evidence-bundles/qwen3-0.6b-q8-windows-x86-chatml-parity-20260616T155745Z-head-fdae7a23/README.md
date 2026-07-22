# Qwen3 0.6B Q8_0 — Windows x86_64 ChatML parity bundle

Exact row: `Qwen3 0.6B Instruct Q8_0` (`Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf`)
SHA256: `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`
Platform: Windows x86_64 (MSVC, rustc 1.95.0). Source/runtime head: `fdae7a23` (+ uncommitted Qwen3-Windows-support diff — see manifest `checkout_note`).

## Result

`all_pass = true` — token-AND-text-identical to the llama.cpp comparator at [1,5,50] generated tokens for: default 3 (capital-of-France, primary-color, say-hello)

camelid path: `cpu_q8_runtime_repack` / `x86_experimental_q8_0_avx2_rust` (support_level `supported_exact_row_smoke_chatml`, 16 threads). Parity holds on BOTH this AVX2 runtime-repack path and the `cpu_reference` scalar path (bit-identical).

## Comparator

llama.cpp 9632 (acd79d603) (Clang 20.1.8 for Windows x86_64), flags `-ngl 0 -ctk f32 -ctv f32 -fa off --no-repack`.
On-disk Windows comparator. NOTE: differs from the 5d56eff pinned in the macOS/Ubuntu Qwen3 bundles; recorded here as the Windows platform comparator, not as bit-exact continuity with those bundles.

## Claim boundary

Supported exact-row smoke for this exact Qwen3 Q8_0 GGUF on Windows x86_64 (MSVC), ChatML chat with thinking DISABLED, short-chat envelope. Token+text identical to the recorded llama.cpp comparator at 1/5/50 on the listed prompts, on the x86_q8 AVX2 path and the cpu_reference path. NOT claimed: other Qwen3 sizes/variants/quants, base variants, Qwen3-MoE (A3B), longer/model-native context, thinking-mode, WebUI smoke on Windows, production throughput, or broad Qwen-family support.

## Artifacts

- `qwen3-0.6b-windows-chatml-parity.json`
- `capabilities.json`
