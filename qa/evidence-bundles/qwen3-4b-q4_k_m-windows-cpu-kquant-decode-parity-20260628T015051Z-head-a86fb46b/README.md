# Qwen3 4B Q4_K_M — Windows CPU K-quant decode parity bundle (Phase 2)

K-quant decode conductor **Phase 2**: certify the **CPU** K-quant decode lane and promote it
from a crash to a default-on, parity-correct path.

## What Phase 2 actually was

Recon found the CPU K-quant block-dot **already existed and was wired** at the matmul chokepoint
(`matmul_rhs_transposed_q4_k_block_dot` / `_q6_k_`), behind `CAMELID_X86_Q4K_DECODE` —
**default-off**. With it off, K-quant 2-D linears load wire-only (no f32 `data`) and the CPU
linear path has no consumer → it **errors** (`no-row-major-data`, `data_len=0`). So Phase 2 was
not "build the kernel" but **"flip the crash to correct output and certify it."**

- Q4_K decode already uses **AVX2** (`q4_k_dot_arm` → `q4_k_dot_avx2`, bit-identical to scalar).
- Q6_K decode uses the **8-lane scalar** `q6_k_wire_row_dot`. An AVX2 Q6_K sibling exists in
  refmath but uses a **different (single-accumulator) f32 order**, so swapping it in would change
  the bit pattern and risk parity — wiring an 8-lane-order AVX2 Q6_K is a documented follow-up.

`CAMELID_X86_Q4K_DECODE` is now **default-on** (opt out with `=0`). The GPU-resident lane never
reaches this CPU chokepoint (it runs `q4k_gemv`/`q6k_gemv` on-GPU), so default-on changes only
CPU-mode K-quant decode — turning a hard error into parity-correct output.

## Result

raw-prompt token+text decode parity, 3 confident probes at 1/5/50 tokens, camelid CPU vs
llama.cpp `acd79d6` CPU:

- **`The capital of France is`** — token-identical to **depth 50**.
- **`def fibonacci(n):`** — token-identical to **depth 50** (code).
- **`Q: What is 2+2? A:`** — token-identical to depth 5, then a benign f32 near-tie at depth
  (camelid continues coherently: " 4. But wait, what if I'm in a different base?...").

Same bar as the GPU primary bundle: token-identical on confident probes; near-ties at depth
documented (the f32 reduction-order frontier). The CPU block-dot lane (Q4_K AVX2 + Q6_K 8-lane
scalar) is parity-correct vs llama.cpp.

## Three findings (all logged as follow-ups)

1. **CPU K-quant was a crash, not a slow path** — fixed by the default-on flip (this bundle).
2. **serve CPU mode false-positives the f32 weight-materialization guard** on K-quant block-dot
   models (estimates ~16 GB f32 materialization the wire-only path never does, because
   `binding_runs_on_resident_gpu` is false on CPU). Bypassed here with
   `CAMELID_MAX_CPU_WEIGHT_MATERIALIZATION_BYTES`; `bench-generate` is unaffected. Fix: treat
   K-quant linears as non-materializing when the block-dot is enabled.
3. **Q6_K AVX2 (8-lane order)** not yet wired — perf follow-up; likely bandwidth-bound/null here.

## Speed (honest)

camelid CPU K-quant decode **~6.5–7.0 tok/s** vs llama.cpp Q4_K_M CPU tg128 **12.35 tok/s**
(~0.55×) — the known ~0.6× CPU tiled-GEMM gap; CPU decode is DRAM-bandwidth-bound on this box,
so the AVX2 Q4_K kernel and any future Q6_K AVX2 are not expected to close it (cf. the Phase 3
prefetch null and the Q8 SIMD nulls).

## Artifacts

- `qwen3-4b-q4_k_m-windows-cpu-kquant-decode-parity.json` — the parity result.
- `manifest.json` — provenance + findings + follow-ups.
- Harness: `scripts/raw-decode-parity.mjs`.

## Reproduce

```
camelid serve --addr 127.0.0.1:8185 --model Qwen3-4B-Q4_K_M.gguf --no-open
  # CPU mode: CUDA_VISIBLE_DEVICES=-1 CAMELID_CUDA_RESIDENT_DECODE=0 CAMELID_CUDA_RESIDENT_PREFILL=0
  # plus CAMELID_MAX_CPU_WEIGHT_MATERIALIZATION_BYTES=99999999999 to bypass the guard false-positive
llama-server -m Qwen3-4B-Q4_K_M.gguf --port 8090 -ngl 0 -ctk f32 -ctv f32 -fa off --no-repack -c 4096
node scripts/raw-decode-parity.mjs --camelid http://127.0.0.1:8185 --llama http://127.0.0.1:8090 \
  --model-id "Qwen3 4B Instruct Awq" --stop "151645,151643" --out <this>.json
```
