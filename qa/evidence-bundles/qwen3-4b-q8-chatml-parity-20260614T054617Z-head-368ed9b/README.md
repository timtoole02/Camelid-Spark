# Qwen3-4B Q8_0 — ChatML chat parity (exact row, with documented first-token frontier)

Tested on **mini2** (clean Mac mini M4, 16 GB) — camelid `serve` and the pinned
llama.cpp reference both run on mini2, driven over Thunderbolt SSH tunnels.

Qwen3-4B uses the **explicit head_dim** path (head_dim 128 ≠ embedding/head_count
2560/32 = 80; q_width = 32·128 = 4096), the same path proven token-exact on 0.6B.

Result (greedy, ChatML, thinking disabled):
- `What is the capital of France?` — token+text identical at 1/5/50. ✅
- `Say hello.` — token+text identical at 1/5/50. ✅
- `Name a primary color.` — **first-token near-tie**: llama.cpp ranks `The`
  (logprob −0.818) above `Red` (−1.156) by **0.34 logit** (the model is genuinely
  undecided between "The primary colors are…" and "Red."); camelid's f32 path
  flips that tie to `Red`. This is the documented f32-accumulation frontier on an
  uncertain prompt, not a correctness defect.

Forward correctness is independently confirmed: three additional confident probes
(including `What is 2 plus 2?` → "2 plus 2 equals 4.") were token-identical to the
reference at 50 tokens.

NOT claimed: other Qwen3 sizes/variants/quants, MoE (A3B), longer context,
thinking-mode, or broad Qwen-family support.
