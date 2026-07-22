# Qwen3-0.6B Q8_0 — ChatML chat parity evidence (exact row)

Token-AND-text-identical greedy parity between camelid's CPU reference and a
pinned llama.cpp reference for **Qwen3-0.6B Instruct Q8_0**, ChatML chat mode,
thinking **disabled**, at **1/5/50 tokens** for three fixed single-turn prompts
(`all_pass=true`, see `qwen3-0.6b-chatml-chat-parity.json`).

This is the first row exercising Qwen3's **explicit head_dim** path: head_dim is
128 but `embedding_length/head_count = 1024/16 = 64`, so the query projection is
*wider* than the embedding (`q_width = 16*128 = 2048`). camelid sources head_dim
from `attention.key_length` (`LlamaModelConfig.attention_key_length` → `DenseLlamaDims`).

Comparator: llama.cpp `1 (5d56eff)` with `-ngl 0 -ctk f32 -ctv f32 -fa off --no-repack`.

NOT claimed: other Qwen3 sizes/variants/quants, MoE (A3B), longer context, or
thinking-mode generation.
