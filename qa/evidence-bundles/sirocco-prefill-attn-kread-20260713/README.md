# SIROCCO Phase P — uint4 K-read in `attention_batched` (prefill)

**Machine:** RTX 3060 Laptop (GA106, CC 8.6, 30 SM), 6 GB, CUDA 12.9, WDDM, Win11
**Change:** apply the shipped byte-identical uint4 (128-bit, 8 keys/load) f16 K-cache read to `attention_batched` and `attention_tree_batched` — the resident-**prefill** and spec-verify attention kernels, which were still doing a scalar per-element f16 K read. `src/cuda_resident.rs`.
**Result:** **+17–19% prefill** wall time, **byte-identical** (bit-exact KV/output; greedy tokens unchanged).

## The reframe: prefill is 99% of long-context wall time

The Lane K campaign optimized **decode**. Direct measurement (`bench-generate`, Llama-3.2-1B Q8_0) shows decode is a rounding error of a long-context request:

| prompt ctx | prefill | decode (64 tok) | prefill share |
|---|---|---|---|
| 8802 | **152.8 s** | 1.2 s | **99.2%** |

Fitting `prefill_ms ≈ a·n + b·n²` over ctx {552, 2092, 4072, 8802} gives a ≈ 3.3 ms/token (per-chunk weight re-reads — prefill batches only in `MAX_VERIFY_K`=8-token chunks) + b ≈ 0.0016 ms/token² (the O(n²) attention). The quadratic dominates at high ctx, and within it `attention_batched` read the f16 KV cache **scalar** — the exact read win #1 replaced everywhere on the decode path but never on the prefill path.

## The change

Verbatim transplant of the #1 uint4 K-dot (from `attn_sk_scores`) into the two prefill/verify attention kernels' score loops. `--fmad=false` + identical d-order accumulation ⇒ **byte-identical**; head_dim=64 ⇒ every K row (incl. the tree's `slots[i]*head_dim` gather) is 16-byte-aligned; a `(head_dim & 7)==0` guard falls back to scalar otherwise. V-read left scalar on purpose (vectorizing it loses on occupancy — rejected experiments #5/#6).

## Measured (`prefill-ab.txt`) — interleaved OLD(scalar)/NEW(uint4), thermal-robust

| ctx | OLD prefill | NEW prefill | faster |
|---|---|---|---|
| 2092 | 9365 ms | 7546 ms | **+19.4%** |
| 4072 | 28306 ms | 22935 ms | **+19.0%** |
| 6602 | 76315 ms | 63606 ms | **+16.7%** |

Because prefill is ~99% of long-ctx wall time, this ≈ the whole-request speedup. No-op at ctx≈0 (no prefill).

## Correctness — byte-identical (`gate-results.txt`)

All existing device parity gates stay green with the kernels changed:
- `splitk_spec_verify_bit_identical` — decode ≡ the now-uint4 `attention_batched`/`attention_tree_batched` bit-identical.
- `prefill_then_decode_matches_sequential` — the batched (uint4) prefill's output matches the sequential reference.
- `verify_batch_matches_sequential`, `tree_linear_matches_verify_batch`, `tree_verify_multiround_lossless`, `tree_verify_forced_compaction_lossless`.
- End-to-end: NEW prefill greedy tokens == OLD prefill, 48/48 @ ctx 8802.

Byte-identity holds by the same argument as #1: a uint4 load returns the same 8 f16 values as 8 scalar loads, accumulated in the same order.

## Files
- `prefill-ab.txt` — the interleaved prefill A/B.
- `gate-results.txt` — the parity gate log.

_Same measurement protocol as the earlier wins: monitored-boost, mem clock stable at 6000 MHz; A/B interleaved OLD/NEW per ctx to cancel SM-clock drift._
