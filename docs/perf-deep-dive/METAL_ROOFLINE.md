# METAL_ROOFLINE.md — Phase 1 (roofline + bottleneck classification)

Phase 1 deliverable of the Camelid-vs-llama.cpp Metal (Apple Silicon) parity campaign
(spec `METAL_PARITY_AGENT_SPEC`). Builds on `METAL_RECON.md` (Phase 0). Every number here
is measured on this host; nothing is a support claim — speed receipts come in Phase 3.

**Pins / host.** llama.cpp `acd79d603` (Metal, `-DGGML_METAL_EMBED_LIBRARY=ON`), Camelid
`2323033` (release, `lto=fat`), Apple **M4** (10-core CPU, base M4 GPU class; `has_tensor=false`
⇒ no Metal tensor API), macOS CLT-only (no Xcode). Models: Llama-3.2-3B-Instruct-Q8_0
(primary), Qwen3-4B-Q8_0, Qwen3-0.6B-Q8_0. Greedy / temp=0 / Q8_0.

**Roofline anchors (M4, prior-established).** Memory-bandwidth wall ≈ **120 GB/s**; Q8_0 GEMM
wall ≈ **3.4 TFLOPS** (llama.cpp brackets it here: 3.67 TFLOPS @pp512, 3.36 @pp2690).

---

## 1. Methodology (and the three traps corrected)

- **llama.cpp:** `llama-bench -ngl 99 -p <N> -n 128 -r 3` (in-process, no HTTP).
- **Camelid:** `serve` (default fast stack auto-arms `RESIDENT_DECODE/F32Y/WIRE/WIRE_NSG8/ATTN2/RESIDENT_PREFILL/MM`) + an OpenAI-compatible probe, plus the built-in `CAMELID_RESIDENT_TRACE` / `CAMELID_PREFILL_TRACE` GPU instrumentation (`MTLCommandBuffer.GPUStartTime/GPUEndTime` → `gpu_busy` per token; this is the no-Xcode roofline tool).
- **Trap 1 — first-request compile contamination.** Camelid compiles the prefill graph/PSOs on the *first* long prefill; an unwarmed prefill measured 451/341/114 tok/s (3B/0.6B/4B) — garbage for small models. Fix: warm the prefill graph, then measure. Warmed 3B prefill = **468** tok/s.
- **Trap 2 — prompt-length mismatch.** Prefill tok/s falls with length (attention O(n²)): llama 571 @512 → 523 @2690. Comparing Camelid@2690 to llama@512 understated Camelid (apparent 0.79×). Fix: **length-match** — both at 2690 → 0.90×.
- **Trap 3 — HTTP server overhead on decode.** Camelid's end-to-end (HTTP) decode is ~9–11% below its `gpu_busy` GPU-compute rate (3B: 26.5 HTTP vs ~29 GPU-rate). llama-bench has no HTTP. For a fair *kernel* roofline, decode uses Camelid's `gpu_busy` rate; the HTTP delta is a server-path note, not a Metal-kernel gap.

---

## 2. Decode roofline (M=1) — the headline metric

`gpu_busy` = median GPU command-buffer time per decode token (steady-state, ≥12 tokens).
Decode GB/s ≈ model_bytes × tok/s.

| Model | Camelid gpu_busy | Camelid GPU-rate | llama tg128 | ratio | Camelid GB/s | % of 120 wall |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.2-3B | ~34.6 ms | **~28.9 t/s** | 29.09 | **0.99×** | ~98.8 | ~82% |
| Qwen3-4B | ~43.8 ms | **~22.8 t/s** | 23.05 | **0.99×** | ~97.6 | ~81% |
| Qwen3-0.6B | ~8.4 ms | **~119 t/s** | 131.6 | 0.90× | ~76 | ~64% |

**Classification: memory-bound, AT THE WALL, tied.** 3B/4B decode is ~82% of the 120 GB/s
wall and within 1% of llama. Dispatch overhead is **negligible**: per token `gpu_busy ≈
commit_wait` and the next token's `encode` (~150–300 µs) is fully overlapped → decode is
**~99.9% GPU-busy**. There is no decode kernel lever here (confirms the prior M4 finding).

*Secondary:* Qwen3-0.6B decode is 0.90× / only 64% of the wall — small models are
under-bandwidth (fixed per-dispatch + large tied-vocab output-projection cost dominate the
tiny per-layer work). A real but lower-value lever (see plan #5).

---

## 3. Prefill roofline — the one real gap

Warmed, **length-matched at 2690 prompt tokens**. Prefill TFLOPS ≈ tok/s × 2 × N_params.

| Model | Camelid prefill | llama prefill | ratio | Camelid TFLOPS | llama TFLOPS | Camelid % of 3.4 wall |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.2-3B @2690 | **468 t/s** | 522.8 | **0.90×** | **3.00** | 3.36 | ~88% |
| (reference @512) | — | 571.4 | — | — | 3.67 | — |

**Classification: compute-bound, near the GEMM wall, ~0.90×.** llama is at ~99% of the M4
Q8_0 GEMM wall; Camelid at ~88%. The residual ≈ **10%**, and it is in the FFN GEMMs.

**Top-3 prefill cost centers** (3B, per-layer, from `CAMELID_PREFILL_TRACE`):

| rank | stage | ~time/layer | note |
|---|---|---:|---|
| 1 | `gemm_gateup` (FFN gate+up) | ~49 ms | dominant; two separate Q8_0 GEMMs over the same input |
| 2 | `gemm_down` (FFN down) | ~26 ms | K=8192 (wide-K) GEMM |
| 3 | `gemm_o` / attention proj | ~10 ms | |

(0.6B/4B prefill not length-matched-warmed here; decode is the headline and 3B is the
representative prefill measurement. Re-measure small-model prefill warmed before any claim.)

---

## 4. Bottleneck summary → Phase 2 entry

| Stage | Class | Camelid vs llama | Lever? |
|---|---|---|---|
| Decode matvec (3B/4B) | memory-bound, at wall | 0.99× tied | **No** — at the bandwidth wall |
| Decode (0.6B small) | dispatch/overhead-bound | 0.90×, 64% wall | secondary (plan #5) |
| **Prefill FFN GEMM** | **compute-bound, ~88% wall** | **0.90×** | **YES — the campaign's lever** (plan #1–#4) |
| Command-buffer/dispatch | n/a | Camelid already 1-encoder/token + encode-ahead + residency | No (Camelid ahead) |
| HTTP server path (decode) | software overhead | ~9–11% below GPU-rate | out of Metal scope (noted) |

**Handoff to `METAL_PARITY_PLAN.md`:** attack the prefill FFN GEMM (compute-bound). #1 =
fuse gate+up (largest stage, parity-class A); then N-tile retune, down-GEMM K-streaming,
SiLU epilogue fusion. Note the *measured* gap is ~10% (length-matched) — smaller than the
0.79× first seen — so candidate expected-wins should be read against a ~10% ceiling, and
closing the last 10% of a near-roofline GEMM is intrinsically hard.

---

## 5. Caveats / constraints

- **No Xcode ⇒ no occupancy/limiter/ALU-stall counters.** Classification rests on the
  achieved-vs-theoretical roofline (GB/s, TFLOPS) + `gpu_busy`/stage timing + controlled A/B.
  Fine "why" attribution on the prefill GEMM (occupancy, register pressure) needs Xcode.app
  (deferred). This is sufficient to classify *which* bottleneck; Phase 3 measures *whether* a
  change helps via tok/s + `gpu_busy` + bit-exact parity.
- **M4-specific:** GEMM numbers use llama's `simdgroup_float8x8` fallback (no tensor API);
  do not generalize to M5+/A19+ (re-baseline there).
- **Parity contract:** decode is at the wall and bit-exact; the prefill lever must stay
  bit-exact (int8 accumulation is associative; no f32-reduction re-ordering — see plan §C).
- All numbers measured this session; no support/perf claim ships without a Phase 3
  `camelid.speed-receipt/v1`.
