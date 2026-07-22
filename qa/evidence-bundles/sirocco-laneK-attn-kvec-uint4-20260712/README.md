# SIROCCO Lane K — byte-identical uint4 attention K-read — `camelid.speed-receipt/v1`

**Date (UTC):** 2026-07-12 · **Machine:** RTX 3060 Laptop (GA106), driver 576.83, CUDA 12.9, WDDM · Win11
**Camelid:** branch `sirocco-laneK-coalesced` off `main`@334653d8 · **Model:** Llama-3.2-1B-Instruct-Q8_0
(sha256 `3f87a880…c4c7ffd1`)

## The win

Widen the scalar f16 K-cache read in the decode attention to **uint4 (128-bit = 8 keys/load)**, accumulated
in the **same d-order** → **byte-identical** (the module compiles `--fmad=false`, so preserving operation
order preserves bits). Applied to three kernels: `attention_decode` (ctx≤512), `attention_decode_sw`
(gemma sliding), and `attn_sk_scores` (bit-identical split-K K-read, ctx>512). Alignment is exact
(head_dim=64 → every position row is 16-byte-aligned; `(head_dim&7)==0` guard routes odd head_dims to
scalar). 45-line diff, one file.

### Why this beats the coalesced kernel (and obsoletes it)

The previously-shipped opt-in `attn_sk_scores_coalesced` (warp-per-position + warp-shuffle) is **token-parity
only** — it re-associates the dot, so decode ≠ spec-verify and it **breaks lossless speculative decode at
ctx>512** (see the sibling receipt). This uint4 read keeps one-thread-per-position and the exact dot order,
so it is **byte-identical** — *and measured faster*, because it preserves cross-position parallelism (MLP)
that the warp-per-position coalesced kernel gives up.

## Correctness — byte-identical (the hard gate)

`splitk_spec_verify_bit_identical` (`cuda_resident/tests.rs:1111`), **coalesced off**, asserts the decode
path (`attention_decode` at pc=512/G=4; `attn_sk_scores` at pc 512→4096) is **bit-identical** (u32-bitcast
equality) to the unchanged spec-verify kernels (`attention_batched`/`attention_tree_batched`) across
`pc ∈ {512,513,768,769,1024,2000,3840,3841,4096}`:

```
test cuda_resident::tests::splitk_spec_verify_bit_identical ... ok   (1 passed, 1.44s — RAN, not skipped)
```

Because `attention_batched/tree` were **not** modified, this passing precisely proves the uint4 read is
byte-identical to the old scalar read. **Decode == spec-verify → lossless speculative decode preserved.**
Also: GPU-resident load + runtime parity gate green; ctx≈0 greedy output tokens unchanged
(`[11,358,2846,3411]`); nvrtc compiles the kernel (no CPU fallback). The G≥2 path the runtime probe can't
reach (adversarial reviewer's concern) is covered by the test's pc=512/G=4 assertion.

## Measurement (decode tok/s, interleaved, monitored-boost)

| ctx | scalar (old, this session) | coalesced (token-parity) | **uint4 (bit-identical)** |
|---|---|---|---|
| ≈0 | 126–128 | n/a (no-op) | **131** (no regression) |
| ~1542 (split-K) | ~84–86 | 90.2 / 88.0 / 83.5 | **94.7 / 93.9 / 91.6** |

At ctx~1542 the uint4 read is **+5.0% / +6.7% / +9.8% faster than coalesced** and **~+10% over the old
scalar** — while being byte-identical (coalesced is only token-parity). The gain scales with context (more
KV read). No ctx≈0 regression (uint4 is ~a no-op there — attention is a few positions).

## Verdict & recommendation

**Strictly dominant over the coalesced kernel: faster AND correct.** Ship the uint4 read (it is byte-identical,
so it needs no behavioral kill switch; a perf kill switch can be added if desired per I2). Recommend
**deprecating `CAMELID_ATTN_COALESCED`** — uint4 supersedes it on both axes. Promotion to `main` still wants
the parity gate + `splitk_spec_verify` green on every Windows CI target; both pass on the RTX 3060 dev box.

**Follow-ups (ranked):** EDIT 2 = uint4 the weighted-**V** read (byte-identical per the design workflow, but
its G-tiled index arithmetic needs a **mandatory G≥2 device test** — the runtime probe only exercises G=1;
do NOT rely on it, per the adversarial finding). Then the split-K **V** read (`attn_sk_partial`).

**Files:** `attn-kvec.diff` (the 45-line change), `splitk-test2.log` (bit-identical PASS),
`measure-uint4-out.txt` (A/B), `design-workflow-result.json` (10-agent design + adversarial review),
`speed-receipt.json`.
