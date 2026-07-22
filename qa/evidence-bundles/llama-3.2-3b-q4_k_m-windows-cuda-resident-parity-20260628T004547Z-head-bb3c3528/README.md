# Llama 3.2 3B Instruct Q4_K_M — Windows CUDA GPU-resident raw-decode parity bundle

Exact row: `Llama 3.2 3B Instruct Q4_K_M` (`bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf`)
SHA256: `6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff` (2,019,377,696 B)
Platform: Windows x86_64 (MSVC). GPU: RTX 3060 Laptop GPU (6 GB, compute 8.6, driver 576.83).
Binary built at `0dccbf74` (release, lto=fat); doc head `bb3c3528`.

The **primary** model of the K-quant decode conductor (same family as the existing Llama-3.2-3B
Q8_0 perf rows). Mixed Q4_K + Q6_K: 168 Q4_K + 29 Q6_K tensors — `attn_v`, `ffn_down`, and the
tied `token_embd`/lm_head are Q6_K, so one run exercises **both** `q4k_gemv` and `q6k_gemv`.

## Result — confident probes token-identical; near-ties documented

Methodology: **raw-prompt token+text decode parity** (no chat template, no f32 diagnostics) over
8 prompts at 1/5/50 generated tokens, camelid GPU-resident vs llama.cpp `acd79d6` CPU.

- **5 / 8 probes token-AND-text-identical** at 1/5/50, including the hard ones:
  - code completion (`LRUCache` from an `OrderedDict`) — identical to depth 50
  - email extraction (structured)
  - the **long-context lighthouse logbook** (~3.5k-token prompt, 45 prior entries) continued
    token-identically to depth 50
  - capital-of-France chain, 60-word sequence
- **3 / 8 probes diverge at a benign greedy f32 near-tie** (camelid output coherent throughout):
  - **JSON** at token 0 — llama picks a markdown code-fence `` `` `` (logprob −0.7715), camelid
    picks `{\n` (−0.9547): a **0.18-logprob near-tie** (tighter than the 0.34-logit "primary
    color" near-tie documented for the Q8_0 4B row). camelid's bare `{` is *more* compliant with
    the "output only valid JSON, no prose" instruction.
  - **process-vs-thread** at token 26; **thunderstorm** at token 38 — deep-generation near-ties.

This is the **documented f32 reduction-order frontier**, identical in nature to the Qwen3
thinking-mode leading-trace frontier and the Q8_0 4B primary-color probe — NOT a kernel defect.
It matches the Q8_0 exact-row bar (token-identical on confident probes; near-ties documented).
See `llama-3.2-3b-q4_k_m-near-tie-analysis.json`.

## Why raw-decode (differs from the Qwen3-4B bundle's chat-parity)

The Llama-3 chat harness (`chat-parity-llama3.mjs`), like the Qwen one, reads camelid's
internally-rendered prompt tokens via `camelid_dense_diagnostics`, which runs the CPU f32 linear
and 503s on wire-only K-quant tensors. Rather than fork the more complex chat harness + wrestle
Llama-3 BOS/template edge cases, this bundle certifies the **decode kernels directly** with a
template-agnostic raw-prompt harness (`scripts/raw-decode-parity.mjs`): both engines greedily
complete the same raw prompt and must produce identical tokens. BOS alignment is verified
implicitly (identical first tokens on the France probe). This is arguably a cleaner proof of the
q4k/q6k decode kernels (no template confounds). The Llama-3.2-3B Q8_0 chat template is separately
established elsewhere.

## Proof chain

camelid GPU-resident CUDA decode (`q4k_gemv`/`q6k_gemv`, 28/28 layers VRAM-resident) == llama.cpp
`acd79d6` directly. The `GPU==cpu_reference` middle leg is N/A (no camelid CPU K-quant decode path
yet — Phase 2). Disclosure caveat: the static execution-plan mislabels the lane `cpu_reference`/
`safe_cpu_decode` though it ran GPU-resident — labeling follow-up, not a correctness defect.

## Speed (honest — NOT head-to-head)

camelid Q4_K_M GPU-resident **26.60 tok/s** @ 4.51 GB; llama.cpp Q4_K_M **CPU** tg128 **12.59 tok/s**.
Different backends — not a ratio. See `llama-3.2-3b-q4_k_m-cuda-resident-speed.json`.

## Artifacts

- `llama-3.2-3b-q4_k_m-windows-cuda-resident-parity.json` — raw-decode parity (schema
  `camelid.raw_decode_parity.v1`), 8 prompts × 1/5/50.
- `llama-3.2-3b-q4_k_m-near-tie-analysis.json` — confident-vs-near-tie classification + the
  measured JSON token-0 logprob gap.
- `llama-3.2-3b-q4_k_m-cuda-resident-speed.json`, `capabilities.json`, `manifest.json`.
