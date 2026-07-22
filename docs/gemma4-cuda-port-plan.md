# Gemma 4 (E4B-It) → CUDA decode lane — implementation plan

Branch: `feat/gemma4-cuda-lane`. Goal: a CUDA-resident decode lane for the **gemma4 serve runtime**
on Windows/RTX 3060 (6 GB), additive + off-by-default. Source specs: workflow `wf_5429ca2a-e36`
(full text in the run transcript). Parity ORACLE = CPU `Gemma4Runtime` (`src/gemma4_runtime.rs`).
Port TEMPLATE = macOS Metal gemma4 kernels (`src/metal.rs`, `#[cfg(target_os="macos")]`).
Scaffold to generalize = `CudaResidentDecode` (`src/cuda_resident.rs`, llama/qwen3-only today).

## Why this is real work (not a toggle)
There is **zero** CUDA gemma4 code today. `cuda_resident.rs` bakes in uniform Llama geometry and has
no Q4_0 GEMV; gemma4's GPU path exists only in Metal (Mac). So this is a kernel-port + engine
generalization, ~1–2 weeks, validated kernel-by-kernel against the CPU oracle.

## E4B-It exact config (src/model.rs:514-529; tests/gemma4_metadata.rs)
hidden 2560 · vocab 262144 · layers 42 · heads 8 · kv_heads 2 · head_dim sliding **256** / global **512**
· rope_dim 256/512 · ffn 10240 · sliding_window 512 · shared_kv_layers 18 · ple_dim 256 · eps 1e-6 ·
theta global **1e6** / sliding **1e4** · final_logit_softcap **30.0** · embedding scale sqrt(2560)≈50.596.
Global (full-attn) layers = {5,11,17,23,29,35,41}; all others sliding (5:1, last forced global).
KV: layers 0..24 OWN K/V; 24..41 SHARED (sliding→layer 22 cache, global→layer 23 cache).

## ⚠️ Mixed-quant blocker (run-target strategy)
`gemma-4-E4B-it-Q4_0.gguf` is a **mixed QAT export**: Q4_0 projections, **Q4_1** ffn_down (layers 0-4),
**Q4_K** tied head, **Q5_K** per_layer_token_embd, **BF16** per_layer_model_proj. The CPU oracle's
`WireFormat` = {Q8_0, Q4_0, Q6K} only (`gemma4_runtime.rs:37-88`) → it **cannot load this file today**.
Consequences:
- **Correctness oracle = the Q8_0 file** (uniform Q8_0 + Q6_K head; loads in CPU runtime). But Q8_0
  weights (~7 GB) do NOT fit 6 GB resident → use Q8_0 for parity (bigger box / partial offload), not the
  6 GB fit claim.
- **6 GB fit = Q4_0 file**, which additionally needs Q4_1 + Q5_K + Q4_K paths added to BOTH CPU oracle
  and CUDA (kept bit-identical), OR a uniform-Q4_0 export (cleanest — flag to operator).
- Foundation (geometry + Q4_0 GEMV + gemma kernels) is shared by both; build it first, resolve quant
  breadth at integration (Phase 5/6).

## CUDA kernels: have vs need (src/cuda_resident.rs)
HAVE (reuse): rms_norm_f32, rms_norm_per_head_f32 (=q/k-norm + weightless v-norm via use_weight=0),
quantize_q8_0/rms_norm_quantize, q8_gemv, q4k_gemv, q6k_gemv, rope_rotate, kv_scatter (f16),
attention_decode (GQA online softmax), residual_add, argmax/sample.
NEED (port from Metal + CPU oracle, each with a unit parity test in `cuda_resident/tests.rs`):
1. **q4_0_gemv** (MANDATORY) — 18B block: f16 scale + 16 nibble bytes, lo=(b&0xF)-8 / hi=(b>>4)-8,
   low half→y[0..16], high half→y[16..32], ×scale. Parity-safe f32 path (no dp4a; -8 bias breaks it).
   Port: metal.rs `q4_0_block_linear_row_ksplit_f32y_wire`; oracle `q4_0_wire_row_dot`. Add
   `ProjQuant::Q4_0` + a `repack_for_lane` Q4_0 arm (pass raw 18B through).
2. q4_1_gemv (for mixed Q4_0 file's ffn_down) — scale+min variant.
3. **GeGLU** gelu_pytorch_tanh(gate)*up (replaces silu_mul); constants 0.7978845608, 0.044715, clamp ±15.
   Port metal.rs `gelu_mul_f32`; oracle `inference/gemma4.rs:gelu_tanh`. Add gelu_mul_quantize[_q8k].
4. **soft_cap** final logits cap*tanh(x/cap), cap=30, before argmax. Port metal.rs `soft_cap_f32`.
5. **sliding-window mask** in attention: window_start=(pos+1).saturating_sub(512) on sliding layers.
6. **PLE injection** (per token, 7-step): pli compute (Q5_K table gather + f32 proj GEMV + rms_norm +
   gate/proj/post_norm/output_scale). v1: compute pli on CPU per token (simpler), inject on device.
7. embedding scale ×sqrt(hidden) (CPU-side at gather, like Metal).
8. hybrid head: Q6_K (Q8 file) on GPU via q6k_gemv+softcap, OR CPU head_on_cpu for Q4_K/Q6_K heads.

## Per-layer geometry generalization (the engine change)
Move head_dim/n_kv_heads/ffn_dim/q_width/kv_width/**scale** from uniform `CudaResidentDecode` fields onto
`ResidentLayer` (precedent: per-layer `quants: LayerQuants` already exists). `forward_pass` reads
`self.layers[li].*`. Scratch sized to per-layer maxima (model.rs max_ffn_length/max_kv_heads + max head_dim).
Per-layer scale (1/sqrt(head_dim); verify vs oracle query_pre_attn_scalar). Cross-layer KV: allocate caches
only for owning layers; shared layers index `cache[kv_source[l]]` and skip kv_scatter (mirror
metal.rs:6012-6017, plan model.rs:436-462).

## Serve wiring (src/api/mod.rs — additive, off by default)
- Add `Gemma4ServeRuntime::Cuda(Gemma4CudaRuntime)` (#[cfg(feature="cuda")]) + match arms in
  generate_greedy/_streaming (handlers unchanged — they take Arc<Gemma4ServeRuntime>).
- Gate `gemma4_cuda_enabled()` = cfg!(feature="cuda") && CAMELID_GEMMA4_CUDA in {1,true,yes}. Off default.
- In `load_gemma4_serve_runtime`: branch distributed → cuda → local; add "cuda" lane label; fail closed.
- /v1/health: report backend "gemma4-cuda" when active runtime is ::Cuda (store lane label on insert).
- Do NOT touch the capability/catalog ledger → no public support/perf claim changed.

## Parity harness (Phase 6)
Templates: `tests/gemma4_generation_parity.rs` (add a CAMELID_GEMMA4_CUDA branch beside the macOS GPU one),
`tests/cuda_cpu_parity.rs` + `src/cuda_parity.rs` (greedy token-id gate + per-step logits). Use
`ToleranceGate::argmax_stable` (attention reduction is FP-reassociated, token-parity not bit-exact) unless
the weighted-V reduction is serialized. Drive CPU `Gemma4Runtime::step` vs CUDA `forward_token_logits` on
the same greedy-fed sequence; assert max|Δlogit| ≤ atol+rtol·|cpu|.

## Q4_0 6 GB fit (verdict: fits comfortably)
Resident: Q4_0/Q4_1 layer weights ~2.19 GB + f32 PLE/proj ~0.33 GB + norms ≈ **2.52 GB**.
Keep Q4_K tied head (0.38 GB) and Q5_K per_layer_token_embd (1.94 GB) **CPU/mmap-gathered** (resident would
blow budget). KV f16 (24 owning layers, 20×256+4×512, kv=2, K+V) = 57,344 B/pos. Totals: 8192 pos ≈ 2.99 GB,
16384 ≈ 3.46 GB, 32768 ≈ 4.40 GB. **Default max_positions 8192, ceiling 32768** on this card.

## Build/dev notes
- camelid.exe is CUDA-enabled by default on Windows (build.rs forces cuda cfg; cudarc non-optional; NVRTC
  compiled at runtime, arch compute_61). Kernels live in the NVRTC `KERNELS` string in cuda_resident.rs.
- Compile NVRTC `--fmad=false`, mirror CPU reduction order for parity. Reuse f16 KV bit helpers.
- The running `camelid serve` locks `target/release/camelid.exe` — stop it before `cargo build` of the bin
  (cargo test of the lib is usually fine).

## Phase order (tasks #8–#13)
1 ✅ plan (this doc). 2 scaffold module + per-layer geometry. 3 kernels (q4_0_gemv first, unit-tested →
GeGLU → soft_cap → sliding mask → per-head norms wiring → dual-theta RoPE → PLE). 4 cross-layer KV. 5 serve
wiring + health. 6 parity vs CPU oracle (Q8_0 correctness; Q4_0 6 GB fit + tok/s).
