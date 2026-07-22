# SPEED_CAMPAIGN.md Phase 4 — concurrent CPU-draft / GPU-verify: foundation + economics

Date (UTC): 2026-06-20
Machine: RTX 3060 Laptop (6 GB), i7-11800H (8C/16T), Windows, CUDA 12.9
Models: target Qwen3-4B-Q8_0 (GPU-resident) · draft Qwen3-0.6B-Q8_0 (CPU-only)
Camelid: feat/bactrian-experimental-fast, worktree dirty (uncommitted)

## What was built (Increment 1 — serialized draft-MODEL spec)

`main.rs generate_run`, gated by `CAMELID_SPEC_DRAFT_MODEL=<gguf>` (+ `CAMELID_SPEC_DRAFT_K`):
a small model runs **CPU-only** (`set_resident_paths_disabled(true)` + `set_is_drafter(true)`
→ the separate `resident_cuda_drafter_cache`, zero crosstalk with the GPU-resident target),
greedily drafts K tokens, and the existing **`verify_drafts_gpu`** confirms them in one batched
GPU forward. A self-healing re-sync keeps the drafter's KV mirroring the target's position
(rollback is free; a normal-decode fallthrough re-forwards ≤1 token).

This is the **producer/consumer foundation** of the flagship (SPEED_CAMPAIGN.md §3) — the
genuinely novel/hard part. The concurrency (Increment 2: thread the producer ahead of the
consumer) is NOT built; see the verdict.

## Lossless — PROVEN (Gate 5 / D1)

Under identical conditions the draft-model spec stream is **byte-identical to Camelid plain
greedy**:
- no-warmup: draft-spec `a3974dc7` == plain `a3974dc7` ✓
- with-warmup: draft-spec `9222ad0d` == plain `9222ad0d` ✓

(The no-warmup vs with-warmup difference is a pre-existing GPU resident-reuse near-tie flip in
plain decode too — not a draft-spec effect. The verify is authoritative, so draft quality only
affects speed.)

## Economics — the flagship does NOT win on this box (CPU-draft-bound)

`bench-generate` 4B + 0.6B draft, greedy, 64 tok, warmup-discarded:

| K | CPU draft/round | GPU verify/round | accept % | tokens/round | end-to-end |
|---|---|---|---|---|---|
| 2 | 230 ms | 75 ms | 51.6 | 2.03 | — |
| 4 | 502 ms | 82 ms | 38.0 | 2.52 | **4.2 tok/s** |
| 6 | 753 ms | 102 ms | 31.1 | 2.86 | — |
| plain greedy | — | — | — | — | **42.5 tok/s** |

**Blocker 1 — CPU draft ~10× too slow.** The 0.6B on the Windows scalar / ~2-thread CPU path
runs ~8 tok/s (≈117 ms/token) vs ~23 ms/token GPU decode. The break-even for a concurrent win
is `max(draft, verify) / tokens_per_round < plain_ms_per_token (~23 ms)`, i.e. the K-token draft
must finish under ~(tokens/round)×23 ms ≈ 58 ms (k=4) → the 0.6B CPU would need ~70 tok/s.
It is ~8 tok/s. **Even perfect overlap caps round time at max(502, 82) = 502 ms → ~5 tok/s**, so
Increment 2 (threading) cannot make this a win. Measured ~10× off the CPU memory-bandwidth
roofline (0.6 GB/token ÷ ~50 GB/s ≈ 12 ms/token ideal vs 117 ms actual) → the scalar kernel,
not bandwidth, is the limit.

**Blocker 2 — batched-verify overhead.** GPU verify of k=5 costs ~82 ms ≈ 3.5× a single 23 ms
decode, when a batched forward (one weight read) should be ~memory-bound-equal to one decode.
So even with FREE drafts, 82 ms for 2.52 tokens loses to plain-decoding them (~58 ms). (Same
overhead the Phase-2 n-gram spec lane showed.) The `verify_batch` path likely lacks the
resident decode's CUDA-graph / fast-path.

## Verdict (honest, per campaign rules)

- **Correctness foundation: DONE.** Lossless CPU-draft-model → GPU-verify producer/consumer.
- **Flagship performance win: NOT achievable on this hardware as-is.** Blocked by (1) CPU draft
  throughput and (2) batched-verify overhead. This is the campaign's documented "CPU too slow →
  low overlap" risk (§6), now quantified — reported, not hidden.
- **Do NOT build Increment 2 (concurrency) yet** — it can only reach `max(draft, verify)`, which
  is draft-bound and far slower than plain decode. Threading would be wasted until the drafter is
  fast.
- **What would unlock it:** (a) a fast CPU draft path (the CPU-perf-mission: enable the
  AVX2/VNNI kernels + more threads — ~5–10× headroom exists), AND (b) a leaner `verify_batch`
  (CUDA-graph / resident fast-path). Both are prerequisites, not Phase-4 wiring.

## Reproduce

```
CAMELID_SPEC_DRAFT_MODEL=<0.6B.gguf> CAMELID_SPEC_DRAFT_K=4 \
  camelid.exe bench-generate <4B.gguf> --prompt "…" --max-tokens 64 --warmup
# stderr: [draft-spec] SERIALIZED: … draft …ms/round (CPU) | verify …ms/round (GPU)
```

**Caveat.** Laptop GPU clocks unpinned (thermal drift). Draft/verify per-round times include
fixed per-round overhead; the verdict (draft ≫ verify by ~6×) is far outside any noise.
