# Phase 3 head-to-head — Camelid-spec vs llama.cpp-speculative (both lossless greedy, same prompts, acd79d6)

Both: Qwen3-0.6B->4B Q8, 128 tok, greedy, GPU. llama: -ngl 99 -ngld 99 --spec-draft-n-max 8 (draft GPU-resident). Camelid: best path per workload.

| workload | Camelid best (path,γ) | Camelid spec t/s | llama-spec t/s | **Camelid/llama** | Cam accept% | llama accept% | llama-spec vs llama-raw |
|---|---|---|---|---|---|---|---|
| code | ngram γ=2 | 50.2 | 97.8 | **0.51×** | 84% | 68% | 1.79× |
| json | ngram γ=4 | 47.5 | 108.7 | **0.44×** | 65% | 79% | 1.99× |
| extraction | ngram γ=2 | 32.5 | 76.2 | **0.43×** | 46% | 52% | 1.40× |
| chat | ngram γ=4 | 32.1 | 49.5 | **0.65×** | 50% | 28% | 0.91× |
| creative | ngram γ=2 | 31.7 | 37.7 | **0.84×** | 0% | 18% | 0.69× |
| adversarial | ngram γ=2 | 32.6 | 45.3 | **0.72×** | 54% | 25% | 0.83× |

llama raw 4B decode (llama-bench, prompt-independent): **54.506 t/s**. Camelid raw 4B ~31-40 t/s (per-workload plain baseline).
