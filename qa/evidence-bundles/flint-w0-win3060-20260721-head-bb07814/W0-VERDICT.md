# W0 verdict — FLINT Windows conductor, RTX 3060 Laptop 6 GB (2026-07-21, HEAD bb07814)

**GATE W0: GREEN.** Tests green + banner says VRAM + one CUDA parity suite green on a real GGUF.

## Gate receipts
1. **Full suite**: `cargo build --release` exit 0; `cargo test --all-targets --release` exit 0
   (clean env verified first: no CAMELID_*/CUDA_VISIBLE_DEVICES in session or setx-persisted User/Machine).
2. **Banner** (banner-plan-offload-stderr.txt, vehicle `plan-offload --arch llama-8b --budget-mb 64`):
   `[hw] GPU: NVIDIA GeForce RTX 3060 Laptop GPU (x1) | compute 8.6 | tensor-cores yes | VRAM 5.0 GiB free / 6.0 GiB total`
   — the word VRAM, not "UNIFIED memory (shared with CPU)". First real-CUDA run of the S1
   CU_DEVICE_ATTRIBUTE_INTEGRATED probe: classified correctly. Heuristic arithmetic: 6.0 GiB VRAM vs
   [12.6, 18.8] GiB band on 15.7 GiB RAM — cannot false-positive here.
3. **CUDA parity on a real GGUF** (gemma4-e2b-cuda-parity-basic_v1.txt): gemma4_generation_parity,
   E2B Q8_0, `runtime = cuda-resident`, 5/5 prompts token- AND text-identical to the committed
   llama.cpp 5d56eff oracle, 10.66 s (pinned bundle 20260711/head-15bf42e3: 13.09 s, 2709-2741 MiB VRAM).

## Discrete-path regression receipts (S1/S2 must be no-ops here)
4. **Offload split preserved** (offload-forced-*.txt, offload-control-*.txt, Qwen3-0.6B Q8_0):
   forced: `[gpu] CAPACITY MODE: 2/28 layers offloaded to host RAM (26 resident), streaming 16 MiB/layer
   per forward over PCIe (9.4 GB/s H2D); ... split=forced` — CAMELID_OFFLOAD_FORCE_LAYERS fires on discrete.
   control: `[gpu] all 28 layers resident in VRAM (5122 MiB free) — full GPU speed`.
   NO "none-unified" / "NOT offloading (one physical pool" strings in any discrete run log (auto arm
   additionally proven by code-read: the bypass is reachable only via `else if unified_memory`,
   src/inference.rs:11482-11490).
5. **gemma4 defaults** (EXPECTED change, not regression): serve+CUDA gates default ON when CUDA present
   (api/mod.rs:4547-4571); KV cap discrete default 4096 by code-read (api/mod.rs:6924-6936 — the 32768
   arm requires cuda_unified_memory). Runtime kv_cap receipt deferred: nothing logs the resolved cap;
   a one-line log is added in the W2 PR (also needed by the Spark tester for the 32768 receipt).
6. **CUDA graphs default-off on Windows**: compile-time `_ => cfg!(target_os = "linux")`
   (cuda_resident.rs:5574-5578) — constant false here; single env-read site repo-wide.
   **World-check finding (graphs-optin-*.txt): CAPTURE NOW WORKS UNDER WDDM** — CAMELID_CUDA_GRAPHS=1 on
   Qwen3-0.6B ran clean (no `cuda graph <step>:` error) and produced BYTE-IDENTICAL 16 token IDs vs the
   default run (141.6 vs 139.9 tok/s). The 2026-07-03 "capture broken under WDDM" STATUS comment
   (cuda_resident.rs:7454-7462) is STALE on driver 576.83. Consequence: the W3 graphs=1 matrix row IS
   runnable; no conductor amendment needed.
7. **Kernel bit-exactness bonus** (cuda-kernel-receipts.txt): 42 ignored CUDA unit tests green, incl.
   cuda_q8_kernel_is_bit_identical_to_cpu_reference, verify_batch_matches_sequential,
   prefill_then_decode_matches_sequential, tree_verify_* lossless.

## Non-inert S1/S2 deltas on this discrete box (documented, deliberate, NOT regressions)
- `spec_gpu_enabled()` (api/mod.rs:1495-1501) now defaults ON here — NOT unified-gated, NOT in the
  conductor checklist, and the S2 commit message's "discrete GPUs unchanged" is wrong for it. Effect only
  when CAMELID_SPEC_DECODE is explicitly set: resident paths stay enabled during spec + GPU batched
  verify. Lossless contract holds by design. Live A/B receipt deferred (needs serve capture/compare rig).
- K-quant guard on batched GPU verify (inference.rs:2851-2855, 2959-2963): newly reachable via the above;
  falls back to CPU chunk verify cleanly. Deferred with the same rig.
- Gemma3-27B catalog row removed (universal, intentional).

## Deferred items (with rationale)
- Conductor-named qwen3/TinyLlama "windows-cuda-resident-parity" reruns: script harnesses needing
  camelid serve + llama.cpp comparator server simultaneously = violates the 1-GPU-process hard rule on
  this box. W3's hostreg 0-vs-1 matrix supersedes with fresher token-identical evidence on the same rows.
- E4B Q8_0 CUDA-resident: no pinned 6 GB receipt exists; not attempted (unknown OOM/offload behavior).
- ffn_decode_chain: known fast-box flake per campaign memory — was green in this run anyway.

## Known conductor doc errata found during recon
- NOCOPY gate anchor: actual location src/inference.rs:11878-11883 (conductor cites :11816).
- "Q8_0 passthrough lanes first" is unsatisfiable literally: every Q8_0 resident projection is
  widen(34→36)+SoA repacked (repack_for_lane, cuda_resident.rs:3275-3285). W2 takes the conductor's
  sanctioned alternative: register a repacked page-aligned host buffer.
