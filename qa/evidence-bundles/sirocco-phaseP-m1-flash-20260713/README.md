# SIROCCO Phase P M1 — flash prefill attention (SHIPPED, opt-in)

**Machine:** RTX 3060 Laptop (GA106, CC 8.6, 30 SM), 6 GB, CUDA 12.9, WDDM, Win11
**Change:** a flash-tiled prefill attention (`flash_pref_scores` + `flash_pref_partial`, reusing `attn_sk_combine`), wired **prefill-only** and **opt-in** via `CAMELID_FLASH_PREFILL=1`. `src/cuda_resident.rs`.
**Result:** **2.2× @6.6k → 3.1× @8.8k → 11.5× @12.2k** prefill, **token-identical** to the byte-exact baseline (6/6 prompts), default path **byte-exact-unchanged**, verify untouched.

## The bottleneck it removes

`attention_batched` (the resident prefill attention) runs **one block per (query-token, head)**, each re-streaming the full prefix K/V — a **24–32× re-read** (`k_tokens × GQA-repeats`). It's O(n²) and 94–96 % of long-context prefill wall time (which is itself 99 % of a long-ctx request). M0a confirmed those re-reads are ~70 % real DRAM traffic; M0b proved a flash reassociation is token-parity-safe.

## The kernel

`flash_pref_scores` / `flash_pref_partial` = the split-K decode passes (`attn_sk_scores`/`partial`) **extended to the whole `k_tokens` query chunk**: each key's `K[p]`/`V[p]` is loaded and **dequantized once and reused across all `k` query rows** of the head (the query-reuse axis, ~`k`× less K/V DRAM traffic — the #443 uint4 load feeds all `k` dots). Grid stays `(n_heads, n_splits)` so SM-fill is unchanged. Buffers are laid out flat by `(t*n_heads+head)`, so **`attn_sk_combine` is reused unchanged** (Pass 3, called with `n_heads = k_tokens*n_heads`). Per-token causal mask (`score = p ≤ base+t ? dot : −inf`). **Scores live in global**, freeing the O(prefix) shared-scores buffer that capped `attention_batched` at ~11.2k ctx — a second win.

**Parity:** TOKEN-PARITY (the split-K reassociation uses the chunk's length for the split boundaries, differing per-token from `attention_batched`). Opt-in, prefill-only: `verify_batch` passes `flash_ok=false`, so `attention_batched` and its bit-identity contract are untouched.

## Performance (`flash-ab.txt`) — flash-off vs flash-on prefill wall

| ctx | OFF | FLASH | speedup |
|---|---|---|---|
| 2092 | 7.58 s | 6.25 s | 1.21× |
| 4072 | 23.0 s | 13.4 s | 1.72× |
| 6602 | 64.1 s | 29.4 s | 2.18× |
| 8802 | 132.6 s | 43.0 s | **3.08×** |
| 12211 | 728.6 s | 63.4 s | **11.5×** |

The win grows with context (attention's share and the re-read factor both rise). **Past ~11k the baseline additionally hits its shared-scores ceiling and degrades super-quadratically** (728 s at 12.2k); flash keeps scores in global and stays clean (63 s) — so the 11.5× is the bandwidth reuse *plus* the removed ceiling. Since prefill is ~99 % of long-ctx wall time, this ≈ the whole-request speedup. No-op below ctx 512 (uses `attention_batched`).

## Correctness (`gate-results.txt`, `m1-oracle.txt`)

- **Default path byte-exact** — device parity tests with flash OFF all pass: `splitk_spec_verify_bit_identical`, `prefill_then_decode_matches_sequential`, `verify_batch_matches_sequential`, `tree_linear_matches_verify_batch`.
- **Flash path token-parity** — 6-prompt oracle (flash-on vs flash-off, 64 greedy tokens), ctx 6.5k–13.6k: **6/6 token-identical**, including the near-tie repetition stressor. The flash reduction flips zero greedy tokens across 16 prefill layers.

## Files
- `flash-ab.txt` — the flash-off/on prefill A/B across ctx.
- `m1-oracle.txt` — the 6-prompt token-parity oracle.
- `gate-results.txt` — the device parity gates + design summary.
