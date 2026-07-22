# Qwen3-8B Q8_0 — ChatML chat parity evidence (exact row)

Token-AND-text-identical greedy parity vs the pinned llama.cpp reference at
**1/5/50 tokens** for three fixed ChatML thinking-disabled prompts
(`all_pass=true`, see `qwen3-8b-chatml-chat-parity.json`).

8B has head_dim 128 = embedding/head_count (square q) and **untied** embeddings
(a separate `output.weight`) — the first Qwen3 row exercising the untied output
projection. QK-norm + NEOX RoPE as for the rest of the family.

Tested on **mini2** (clean Mac mini M4, 16 GB): camelid `serve` and the pinned
llama.cpp both on mini2, driven over Thunderbolt SSH tunnels, two-phase (capture
the oracle alone, then compare camelid alone) to stay within 16 GB.

Operational note: on a 16 GB host run camelid with `CAMELID_METAL_NOCOPY=0`. The
default NOCOPY wire-page load (for a GPU-resident path that fails closed when
QK-norm is present) plus the CPU reference path doubles resident memory and
thrashes; NOCOPY=0 keeps ~9 GB resident with headroom.

Comparator: llama.cpp `1 (5d56eff)`, `-ngl 0 -ctk f32 -ctv f32 -fa off --no-repack`.

NOT claimed: other Qwen3 sizes/variants/quants, MoE (A3B), longer context,
thinking-mode, GPU-resident decode, or broad Qwen-family support.
