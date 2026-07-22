# SPEC_RECHECK Phase 1 — fixed workloads

All cells: `max_tokens = 128`, greedy, `--warmup` (one unmeasured plain+spec pair first so the
CUDA graph build cost does not land on the measured plain baseline). Target Qwen3-4B-Q8_0,
GPU-resident on the RTX 3060 Laptop (6 GB). Prompts are raw continuations (no chat template),
each primed to induce its workload's output structure.

| Workload | File | Intent | Expected n-gram structure |
|---|---|---|---|
| code | prompts/code.txt | code completion | moderate (indentation, keywords, idioms recur) |
| json | prompts/json.txt | structured JSON array | high (repeated keys, braces, quoting) |
| extraction | prompts/extraction.txt | repetitive field extraction | very high (uniform row format) |
| chat | prompts/chat.txt | normal conversational reply | low (novel prose) |
| creative | prompts/creative.txt | literary short-story opening | low (novel prose) |
| adversarial | prompts/adversarial.txt | diverse unrelated facts | ~none (every line a new subject; drafts rejected) |

Metrics per cell (one JSON line in results/): accept_rate, mean_accepted_tokens_per_round,
draft_ms, verify_ms, f_draft, plain/spec tok/s, s_sync, first_divergent_generated_token_index
(lossless gate), gpu/cpu verify round split, peak RSS, GPU offload status.
