# Qwen3 4B Q8_0 — Windows x86_64 ChatML parity bundle

Exact row: `Qwen3 4B Instruct Q8_0` (`Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q8_0.gguf`)
SHA256: `8c2f07f26af9747e41988551106f149b03eb9b5cb6df636027b6bf6278473300`
Platform: Windows x86_64 (MSVC, rustc 1.95.0). Source/runtime head: `fdae7a23` (+ uncommitted Qwen3-Windows-support diff — see manifest `checkout_note`).

## Result

`all_pass = true` — token-AND-text-identical to the llama.cpp comparator at [1,5,50] generated tokens for: 3 confident probes (capital-of-France, say-hello, 2+2). The default "Name a primary color." probe is a documented mac first-token near-tie; on this Windows 9632 comparator it also matched (both "Red") — see qwen3-4b-primarycolor-tie.json.

camelid path: `cpu_q8_runtime_repack` / `x86_experimental_q8_0_avx2_rust` (support_level `supported_exact_row_smoke_chatml`, 16 threads). Parity holds on BOTH this AVX2 runtime-repack path and the `cpu_reference` scalar path (bit-identical).

## Comparator

llama.cpp 9632 (acd79d603) (Clang 20.1.8 for Windows x86_64), flags `-ngl 0 -ctk f32 -ctv f32 -fa off --no-repack`.
On-disk Windows comparator. NOTE: differs from the 5d56eff pinned in the macOS/Ubuntu Qwen3 bundles; recorded here as the Windows platform comparator, not as bit-exact continuity with those bundles.

## Claim boundary

Supported exact-row smoke for this exact Qwen3 Q8_0 GGUF on Windows x86_64 (MSVC), ChatML chat with thinking DISABLED, short-chat envelope. Token+text identical to the recorded llama.cpp comparator at 1/5/50 on the listed prompts, on the x86_q8 AVX2 path and the cpu_reference path. NOT claimed: other Qwen3 sizes/variants/quants, base variants, Qwen3-MoE (A3B), longer/model-native context, thinking-mode, WebUI smoke on Windows, production throughput, or broad Qwen-family support.

## Artifacts

- `qwen3-4b-windows-chatml-parity.json`
- `capabilities.json`
- `qwen3-4b-primarycolor-tie.json`
