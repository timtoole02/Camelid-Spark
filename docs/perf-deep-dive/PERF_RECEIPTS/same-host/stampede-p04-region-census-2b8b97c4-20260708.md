# STAMPEDE P0.4 — rayon regions/token census + fork-join overhead gate (Phase 4 input)

Host: i7-11800H, Win11. Camelid 2b8b97c4 (release). Model basis: Llama-3.2-3B Q8_0,
batch=1 CPU decode, shipped win-x86_64 defaults (plan-applied: PARALLEL_LINEAR on, Q8 REPACK on,
decode-consumer/single-owner routes on, ATTENTION_DECODE_PARALLEL on, decode pool = physical cores).
Static census by code trace (verified against `select_x86_q8_plan`, `src/execution_plan.rs:504`).

## Measured cost/region (`bench-rayon-region`, 8 threads, 5000 iters)
receipt: `stampede-p04-rayon-region-2b8b97c4-20260708.jsonl`

| idle between regions | µs/region |
|---|---|
| 0 (hot, back-to-back) | **2.76** |
| 20 µs | 41.6 |
| 100 µs | 41.5 |
| 500 µs | 45.1 |

Windows park/unpark is a cliff: ANY gap ≥ 20 µs between regions costs the full ~42 µs wakeup.

## Regions/token (steady-state, KV pos ≥ 64, 28 layers)

- **True fork-join regions: 169/token** = 6/layer × 28 + 1 logits
  (attention-context, attn-out proj, attn residual add, fused gate+up, ffn down, ffn residual add; logits matvec ×1).
- Degenerate rayon entries (rows=1 ⇒ single chunk, no handoff): 57/token (2 norms/layer + output_norm).
- Serial (0 regions): **QKV projection** (see finding below), RoPE, KV write, SiLU, sampling, embedding.
- Attention-context region is gated on `position_count ≥ 64` — first ~63 steps are serial attention (subtract 28/token during warmup).

## P4.1 gate arithmetic

Token time at re-pinned baseline (7.79 tok/s) = 128.4 ms.

| scenario | overhead/token | % of token |
|---|---|---|
| all-hot (2.76 µs × 169) | 0.47 ms | **0.4%** |
| all-cold worst case (41.6 µs × 169) | 7.03 ms | **5.5%** |

Realistic value sits near the hot end: within a token the six per-layer regions are launched
back-to-back around large parallel matvecs (workers just ran, spin briefly, next region arrives in
µs); the long serial stretches (QKV) precede at most 1–2 region launches per layer. Honest
estimate ≈ 1–3% of token time.

**Phase 4 verdict: KILL per P4.1.** Even the all-cold upper bound (5.5%) barely grazes the 5%
threshold, and the realistic estimate is well under it. The ~29% non-matmul overhead from the
streaming role profile is NOT fork-join — see below for where it actually lives.

## Finding that supersedes Phase 4: QKV decode is SINGLE-THREADED for GQA models

The fused QKV decode-consumer triplet parallelizes only when `q_width == k_width == v_width`
(MHA). Llama-3.2-3B is GQA (Q=3072, K=V=1024), so QKV falls to the serial else-branch
(`src/inference.rs:13942–13967`) — zero rayon regions, one thread streams the QKV weights.

Scale of the leak (3B Q8_0): QKV ≈ 15.7M params/layer of 100.6M ⇒ ~13.7% of the per-token weight
stream (~468 MB of 3.4 GB) read at single-thread bandwidth (~⅓ of pool bandwidth). Rough model:
468 MB @ ~13 GB/s ≈ 36 ms of the 128 ms token; parallelized at stream rate (~32 GB/s) it becomes
~15 ms ⇒ **~107 ms/token ≈ 9.3 tok/s ≈ +20% decode, past llama.cpp b9918's 8.71** — from one fix,
before prefetch/multi-row. Parity-safe: output rows are independent; each row's block accumulation
order unchanged. This becomes the new opening lever of Phase 2 (utilization), replacing Phase 4.

Note: ALL Qwen3 rows are also GQA (4B: Q=4096/KV=1024 class) — the leak applies across the
supported matrix, not just Llama.
