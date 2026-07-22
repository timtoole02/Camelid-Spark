# Llama 3.2 1B Instruct Q4_K_M — Windows CUDA GPU-resident raw-decode parity (MUSTER M-B1)

Row: `llama_3_2_1b_instruct_q4_k_m` — exact file `Llama-3.2-1B-Instruct-Q4_K_M.gguf`,
sha256 `6a74661014a3e2f139871f81e6cec852c489a627d169de503a3c0434a10c503d`, 807,693,984 bytes.
**Local file; upstream provenance unresolved** (matches no current upload of six surveyed
publishers; no producer metadata — see MUSTER_ACQUISITION.md). Mixed K-quant: 96 Q4_K +
17 Q6_K + 34 F32; attn_v/ffn_down are Q6_K in 8 of 16 layers each; TIED Q6_K
token_embd(lm_head) — one run drives both `q4k_gemv` and `q6k_gemv`.

## Method (MUSTER_CONDUCTOR.md 6.2, two-phase, engines never co-resident)

1. Gate pack committed BEFORE capture: `qa/prompt-packs/llama32-1b-q4km-raw-decode-pack-v1.json`
   (8 prompts = the harness France BOS probe + the seven qa/speed/prompts.json columns verbatim —
   the 3B Q4_K_M precedent set).
2. Determinism sanity: two independent CPU K-quant serve sessions over the full pack at depth 50 —
   byte-identical outputs (CAMELID_CUDA_RESIDENT_DECODE=0; separate serve processes to defeat
   the prompt cache).
3. Oracle captured ALONE: pinned llama.cpp llama-server, version 9632 (acd79d603), CPU backend,
   binary sha256 6c787bf07ac1d7e1bbaa1ee176c3ef0df58ea86494c8c1b1d2d9f4a9176b19ae, flags
   `-ngl 0 -ctk f32 -ctv f32 -fa off --no-repack -c 4096`, greedy `/completion`
   (temperature 0, top_k 1, seed 0, cache_prompt false) via
   `scripts/raw-decode-parity.mjs --reference-out`; the SAME solo session also captured
   `/tokenize` prompt tokens (add_special=true) and n_probs=5 logprob runs for every prompt.
   Server killed by PID before any Camelid process started.
4. Camelid leg: `camelid serve --model <exact gguf>` (default routing selects
   `cuda_resident_kquant_runtime`; serve log records all 16 layers VRAM-resident), compared via
   `--reference-in` at token depths 1/5/50; prompt tokens captured via
   `/api/models/tokenizer/encode` (add_special=true, parse_special=false).

## Results

- `all_pass=false` at the strict harness gate; **confident probes all pass**:
  5/8 prompts token-AND-text identical at every depth — France BOS probe, code_completion,
  structured_json, repetitive_extraction, and the 1,867-token longctx continuation to depth 50.
- **Prompt tokenization identical on 8/8 prompts** (prompt-token-parity.json), including the
  1,867-token long-context prompt.
- 3 open-ended flips, each probed and attributed (near-tie-analysis.json):
  | prompt | flip index | oracle top | camelid token | top-2 gap (nat) | camelid rank | CPU-lane control |
  |---|---|---|---|---|---|---|
  | normal_chat | 8 | " the" | " them" | 0.0744 | #2 | CPU lane emits the ORACLE token here |
  | creative_writing | 3 | " arrival" | " presence" | 0.1059 | #2 | CPU lane emits the ORACLE token here |
  | adversarial_lowaccept | 48 | "lama" | "umin" | 0.0135 | #2 | CPU lane branched earlier (no same-prefix control); near-coin-flip gap |
  All gaps are far inside the 0.33-nat soft-position threshold (Ornith Q4_K_M precedent);
  camelid's own two backends flip at two of the three positions, attributing the flips to
  fp reduction-order near-ties rather than decode defects.
- Verdict: **parity green under MUSTER_CONDUCTOR.md 6.3** — the same bar the promoted
  Llama-3.2-3B Q4_K_M row carries (confident probes all-pass + attributed open-ended near-ties).

## Environment

Head 052c4030 (source tree; branch muster/mb1-llama32-1b-q4km). RTX 3060 Laptop GPU 6144 MiB,
driver 576.83, CUDA 12.9. Windows 11 build 26220. Comparator pinned per the standing policy.

## Files

- llama-3.2-1b-q4_k_m-windows-cuda-resident-parity.json — harness report (1/5/50, all prompts)
- near-tie-analysis.json — flip attribution (oracle top-5 logprobs, gaps, controls)
- prompt-token-parity.json — camelid encode vs oracle /tokenize, 8/8 match
- oracle-nprobs-50tok.json — full oracle n_probs capture (solo session)
- api-webui/ — Phase 5 promotion smoke outputs (added after the contract row landed)
- manifest.json, SHA256SUMS
