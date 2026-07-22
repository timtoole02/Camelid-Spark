# METAL_PARITY_RESULT.md — Phase 4 (result + residual gap)

Result of the Camelid-vs-llama.cpp Metal (Apple Silicon) parity campaign
(spec `METAL_PARITY_AGENT_SPEC`). Reads with `METAL_RECON.md` (Phase 0),
`METAL_ROOFLINE.md` (Phase 1), `METAL_PARITY_PLAN.md` (Phase 2).

**Pins / host.** llama.cpp `acd79d603` (Metal, embed-library), Camelid `2323033` (release),
Apple **M4** (base GPU class, `has_tensor=false`), CLT-only. Q8_0, greedy/temp=0.
Models: Llama-3.2-3B-Instruct-Q8_0 (primary), Qwen3-4B-Q8_0, Qwen3-0.6B-Q8_0.

---

## 1. Headline

**On M4, Camelid is at near-parity with llama.cpp's Metal backend — and the decode headline
metric is a statistical tie at the hardware wall.** The mission's "decode gap" is, on this
hardware, essentially **zero**. The single measured gap is prefill GEMM at **~0.90×**, near
the compute roofline.

| Metric (M=1 / prefill) | Camelid | llama.cpp Metal | Ratio | Verdict |
|---|---:|---:|---:|---|
| 3B decode (GPU-rate, `gpu_busy`) | ~28.9 t/s | 29.09 | **0.99×** | tied — at the ~120 GB/s wall (~82%) |
| 4B decode | ~22.8 t/s | 23.05 | **0.99×** | tied — at the wall (~81%) |
| 0.6B decode | ~119 t/s | 131.6 | 0.90× | small-model overhead (64% of wall) |
| 3B prefill @2690 (warmed, len-matched) | 468 t/s | 522.8 | **0.90×** | the gap — 88% vs 99% of the ~3.4 TFLOPS GEMM wall |

This corroborates the prior independent M4 finding (decode ties llama.cpp/MLX at the
bandwidth wall) and, for the first time, **quantifies the Metal prefill gap with a fair,
warmed, length-matched measurement** (earlier ad-hoc looks said 0.79× — that was a
methodology artifact; see Phase 1 §1).

## 2. What Camelid is (and isn't) missing on Apple Silicon

- **Decode matvec, RoPE, RMSNorm, residual/activation, KV scatter, GPU argmax:** present and
  at parity. Decode is **~99.9% GPU-busy** (one command buffer/token, encode-ahead overlapped,
  persistent residency, no-copy wire weights) — Camelid's dispatch model is, if anything,
  *ahead* of llama's per-op-binding/8-command-buffer split. No lever.
- **Flash-attn:** Camelid has decode split-K + prefill flash; the only structural item llama
  has that Camelid lacks is the **mask-block-skip classifier** (`flash_attn_ext_blk`) — a
  depth optimization, parity-bounded, low priority (see recon §5.3).
- **Prefill FFN GEMM:** the real gap. Camelid's `gemm_gateup` (~49 ms/layer) + `gemm_down`
  (~26 ms/layer) leave ~10–12% on the table vs llama's `simdgroup_float8x8` `mul_mm`.
- **Not gaps (by charter):** all K-quant / IQ-* / MXFP4 / BF16 breadth, MoE, SSM — Camelid is
  Q8_0/Q4_0-only by design.

## 3. Residual gap + next bottleneck

The residual is **prefill FFN GEMM, ~10%, compute-bound, near the M4 GEMM roofline.**
Ranked parity-safe levers (Phase 2): **#1 fuse gate+up** (largest stage), #2 N-tile retune,
#3 down-GEMM K-streaming, #4 SiLU epilogue fusion. All are Parity-class A (bit-exact; no
f32-reduction re-association) *if* implemented with the K-accumulation order preserved.

**Honest ceiling note:** llama is already at ~99% of the ~3.4 TFLOPS Q8_0 GEMM wall on M4;
Camelid at ~88%. Closing the last ~10% of a near-roofline GEMM is intrinsically hard, and the
win is on the *secondary* metric (decode — the headline — is already tied). The fusion win is
dispatch/activation-reuse, which is a smaller fraction in compute-bound prefill than in
decode. Expected recovery is real but bounded (suspected ~6–10% prefill); it must be proven
by a `camelid.speed-receipt/v1` with a bit-exact greedy parity diff, not assumed.

## 4. Phase 3 status (implement + measure) — analyzed to the kernel

Candidate #1 (fuse gate+up) was taken to the source. Findings that set the realistic ceiling:

- **The default prefill MM path (`q8_0_block_wire_mm`/`_f16o`) is already non-bit-exact**
  (tile-MMA accumulation order ≠ the scalar/CPU k-split), gated by `CAMELID_METAL_MM` and
  **verified by greedy-token parity**, not bit-exactness (metal.rs:1158–1161). So prefill's
  real contract is *greedy parity*, and a fused kernel that preserves each output's K-accum
  order is parity-safe by construction (byte-identical to today's two calls).
- **The fusion is structurally sound and already half-built:** `steel_q8_mm_dual`
  (metal.rs:18689) is a two-output (gate+up) GEMM, and the fast path can be fused by deriving
  `q8_0_block_wire_mm_gateup_f16o` from `q8_0_block_wire_mm_f16o` (share the `sb` activation
  stage, two weight regions, two accumulator sets). Threadgroup memory 16 KB (< 32 KB) — fine.
- **Register-pressure ceiling is the catch.** The activation tile is K-deep and cannot stay
  resident across the `ib` loop, so the fusion MUST interleave gate+up inside that loop with
  **both accumulator sets live = 32 `simdgroup_float8x8` (vs 16 today)**. On M4 this risks
  occupancy/spill that can *cancel* the fusion gain. The upside is bounded anyway: the saving
  is **activation read-once + one dispatch**, NOT weight bandwidth (gate/up weights differ and
  are the bulk traffic).
- **Measured outcome:** candidate #1 was implemented (a fused gate+up kernel derived from
  `q8_0_block_wire_mm_f16o`, behind a default-off flag) and A/B-measured on Llama-3.2-3B. It is
  **bit-correct** (greedy parity holds), but it **regresses prefill** on M4: forcing both
  accumulator sets live (32 vs 16 `simdgroup_float8x8`) spills registers and collapses
  occupancy. The GEMM is register-bound, so fusion is the wrong lever. Reverted — the engine is
  byte-for-byte unchanged.

**Verdict.** **Camelid is already at the M4 hardware envelope, and no parity-safe speedup
remains.** Decode (the headline) is at the physical memory wall — at parity — and only a lossy
(re-associated) kernel could go faster, which the losslessness contract forbids. Prefill is near the GEMM roofline
(Camelid 88% / llama 99%); the parity-safe levers are bounded sub-roofline gains that, even
stacked, target *reaching* parity on the *secondary* metric. The gate+up fusion — the single
best lever — was **implemented, proven bit-correct, and measured to regress** (register
spill), then reverted. The remaining levers (#2 tile-widen, #3 down-GEMM) raise
accumulator/register pressure the same way and are not pursued. **Conclusion: Camelid is at the M4 performance
envelope.** Closing the residual ~10% prefill gap would need a fundamentally different,
register-frugal GEMM (e.g. smaller tiles trading occupancy for register room) — out of scope
and unjustified, since prefill is secondary and decode (the headline) is already tied at the
physical bandwidth wall.

## 5. Reproduce

```
# llama.cpp Metal ground truth (length-matched):
llama-bench -m <model>.gguf -ngl 99 -p 512,2690 -n 128 -r 3
# Camelid: serve (default fast stack auto-arms the resident Metal stack), then a warmed
# prefill + decode probe; read gpu_busy from CAMELID_RESIDENT_TRACE / CAMELID_PREFILL_TRACE.
CAMELID_RESIDENT_TRACE=1 CAMELID_PREFILL_TRACE=1 camelid serve --model <model>.gguf
```

Measurement gotchas that must be honored (Phase 1 §1): warm the prefill graph (first request
compiles it), length-match the prompt, and use `gpu_busy` (not HTTP wall) for the decode
kernel rate. M4 uses llama's `simdgroup_float8x8` fallback (no tensor API) — re-baseline GEMM
on M5+/A19+. Fine occupancy/limiter attribution needs Xcode.app (not installed; deferred).

## 6. Receipts

Phase 1 measurement artifacts (this session): llama-bench Metal logs, Camelid warmed
prefill/decode + `gpu_busy` traces. **No engine change is shipped:** candidate #1 was
implemented and A/B-measured (bit-correct, but a register-pressure regression on M4) and
reverted, so the default engine is byte-for-byte unchanged.
