# WIN2METAL §A2 — Metal resident-decode reachability probe (GATES Phase 3)

Read-only investigation. Branch `win2metal/conductor` (base+phase0, HEAD `e1458a9`).
Host: Apple M4, 16 GB, macOS arm64. Frozen binary: `camelid v0.1.7-92-g28f224b`
(`/Volumes/Untitled/camelid-base-28f224b`). No source was edited.

---

## VERDICT

**YES-WITH-CHANGES.** The Metal resident decode *engine* is fully live and engaged on this
M4 (serve runs `metal_resident_q8_runtime` / `q8_0_metal_resident_decode`, GPU-per-token
telemetry confirmed). But the *speculative verify* path that Phase 3 targets is **unreachable
today**: `verify_drafts_gpu` / `verify_tree_gpu` / `generate_next_tokens_speculative` are
`#[cfg(feature = "cuda")]`-only, and the non-cuda build compiles **two-line stubs that
`return Ok(None)`**. `verify_batch` and `run_batched_layer_stack` exist **only** in
`src/cuda_resident.rs`; `metal::ResidentDecodeState` has **no batched/verify method at all**
(only single-token `forward_token` + prefill). So Phase 3 kernel work has a live engine to
attach to, but no `verify_batch` home and no Mac call path until that wiring is authored.

---

## The three facts that decide it

1. **The Metal resident engine IS engaged on this M4 (single-token decode).** Live serve probe
   on port 18090 with `Llama-3.2-1B-Instruct-Q8_0.gguf`:
   - `/api/capabilities` → `"selected_backend":"metal_resident_q8_runtime"`,
     `"decode_path":"q8_0_metal_resident_decode"`, `"prefill_path":"q8_0_metal_resident_prefill"`,
     `"cuda_resident_active":false`, reason `"Metal resident Q8_0 stack selected (Metal device
     present, resident decode enabled)…"`.
   - stderr: `[resident-dispatch] cuda_enabled=false metal_enabled=true`, then per-token
     `[resident] pos=N layers=16 … gpu_busy=~13500us kernel_window=…` telemetry.
   - `[camelid] CAMELID_METAL_NOCOPY: loading Q8_0 weights as page-aligned wire pages` — the
     NOCOPY default fired, which only happens when resident+wire+f32y are all on.
   - 200 tokens / 2.91 s wall ≈ 68 t/s e2e (the per-token `[resident] gpu_busy=` lines are the
     positive GPU evidence; throughput alone can't separate CPU/GPU for a bandwidth-bound 1B).

2. **The GPU verify path is CUDA-only; Mac gets a no-op stub.** `verify_drafts_gpu`
   (`src/inference.rs:2396`, `#[cfg(feature="cuda")]`) calls `slot.engine.verify_batch(...)`
   where `engine` is `cuda_resident::CudaResidentDecode`. The non-cuda twin
   (`src/inference.rs:2618`, `#[cfg(not(feature="cuda"))]`) is just `Ok(None)`. Same for
   `verify_tree_gpu` (`:2606`) and `generate_next_tokens_speculative` (`:2628`). The only
   `fn verify_batch` / `fn run_batched_layer_stack` in the tree are in `src/cuda_resident.rs`
   (`:4758`, `:4912`). `src/inference/metal_resident.rs` exposes only
   `try_metal_resident_prefill` and `try_resident_decode_forward_metal` (single-token) — no
   batched verify.

3. **Resident vs. speculative is NOT a hard mutual exclusion — it's a CUDA-shaped bridge that
   currently dead-ends on Mac.** The server pins:
   `session.set_resident_paths_disabled(speculative.is_some() && !spec_gpu_enabled())`
   (`src/api/mod.rs:6836`, mirrored `:8030`). With `CAMELID_SPEC_GPU=1` resident decode stays
   **enabled during speculation** (by design), and the request loop calls
   `verify_drafts_gpu` (`src/api/mod.rs:8128`). On CUDA that runs `verify_batch`; on Mac it
   hits the stub → `Ok(None)` → falls back to the CPU chunk verify. So speculation and
   resident decode are designed to **coexist** (CUDA), not to be mutually exclusive — only the
   `spec_gpu=off` default disables resident (CPU chunk-verify needs CPU-authoritative KV),
   which is what the `main.rs:3793` comment describes.

---

## 1. What turns the Metal resident path on, and why `=1` had no effect in bench

`apply_default_fast_stack()` (`src/main.rs:3722`) runs in the CLI entry (`run()`,
`src/main.rs:955`, for every non-deterministic subcommand) and sets — **only if unset** —
`CAMELID_METAL_RESIDENT_DECODE`, `CAMELID_METAL_F32Y`, `CAMELID_METAL_WIRE`,
`CAMELID_METAL_WIRE_NSG8`, `CAMELID_METAL_ATTN2`, `CAMELID_METAL_RESIDENT_PREFILL`,
`CAMELID_METAL_MM` all to `1`. **So resident decode is already ON by default in the CLI** —
explicitly exporting `CAMELID_METAL_RESIDENT_DECODE=1` is a no-op (the var is already `1`),
which is exactly why Phase 0 saw byte-identical output+timing on both bench paths. (To turn it
*off* you'd set `=0`, which the env-flag reader honors; library/embedder/test entry points
never call `apply_default_fast_stack`, so they default OFF.)

Runtime gate: `resident_decode_metal_enabled()` (`src/inference.rs:9821`) =
`!deterministic_mode_enabled() && CAMELID_METAL_RESIDENT_DECODE`. The dispatcher
`try_resident_decode_forward` (`src/inference.rs:2643`) routes to
`try_resident_decode_forward_metal` when CUDA is off. There is **no `cfg(target_os="macos")`
hard gate** on enablement — Metal is gated by the env flag + `resident_decode_eligible`
(`src/inference.rs:1904`) + `!resident_paths_disabled`. `metal::ResidentDecodeState` itself is
`cfg(target_os="macos")` (real) / non-macos stub returning `None` (`src/metal.rs:11974`).

The hard precondition the **bench** paths can miss is **weight residency**:
`resident_decode_eligible` requires every projection to be plain Q8_0 blocks **or** Q8_0 wire
pages *with the wire kernels active* (`q8_0_blocks.is_some() || (wire_mode_active() &&
q8_0_wire_pages.is_some())`, `src/inference.rs:1936`). The execution-plan selector only keeps
plain blocks when it picks the Metal plan, which requires
`CAMELID_METAL_RESIDENT_DECODE && !CAMELID_MAC_Q8_METAL_PLAN==0 && platform.metal_available`
(`src/execution_plan.rs:384`) → `metal_resident_q8_runtime`; otherwise the **rows4 CPU repack**
replaces the blocks (mutually exclusive storage, `:376-391`) and the resident gate bails. Serve
additionally arms NOCOPY wire pages via `apply_serve_nocopy_default` (`src/main.rs:3798`, macOS
+ resident+wire+f32y), which is the form `wire_mode_active()` consumes.

There is **no separate serve-only enable** — serve and bench share the same default fast stack.
The difference is (a) serve adds the NOCOPY wire loader and (b) **speculation flags** (see §3):
when CPU speculation is on, the session is pinned `resident_paths_disabled=true` and the
resident path is skipped regardless of the env flag.

## 2. Does `camelid serve` engage Metal resident decode here? — YES (probe evidence above)

Engine selection (`/api/capabilities`), dispatch trace
(`[resident-dispatch] … metal_enabled=true`), and per-token GPU telemetry
(`[resident] … gpu_busy=…`) all confirm it. The `[hw] GPU: none detected` banner is
CUDA/`nvml`-centric and Metal-blind, as warned — it was present while the Metal resident path
was demonstrably running.

## 3. Does SPECULATIVE decode via the server engage the resident path + `verify_drafts_gpu`?

- Server speculative toggles: `CAMELID_SPEC_DECODE` (`ngram`/`draft`,
  `src/api/mod.rs:65`/`:1292`), `CAMELID_SPEC_GPU` (`spec_gpu_enabled()`, `:1312`),
  `CAMELID_SPEC_TREE` (`src/main.rs:2973`, bench-side), `CAMELID_SPEC_NGRAM`
  (`src/main.rs:2535`, bench-side).
- With spec on and `CAMELID_SPEC_GPU=1`, the session stays resident
  (`src/api/mod.rs:6836`) and the loop calls `verify_drafts_gpu` (`:8128`). **On a non-cuda
  (Mac) build that method is the `Ok(None)` stub** (`src/inference.rs:2618`) → the code falls
  through to the CPU chunk verify (`src/api/mod.rs:8146+`). So enabling speculation **never**
  calls a Metal `verify_batch` today; it either (spec_gpu off) disables resident decode and CPU
  chunk-verifies, or (spec_gpu on) keeps resident decode for the *plain* single-token steps but
  still CPU chunk-verifies the drafts.
- Relationship is therefore **a portable design that currently has only a CUDA backend**: the
  `spec_gpu → keep resident + GPU verify` shape is already in place and is the right shape for
  Metal; the `main.rs:3793` "speculative disables resident decode" note is specifically the
  `spec_gpu=off` default (CPU-authoritative KV for rollback), not an unconditional law.

## 4. Which decode-attention kernel must Phase 3's verify match?

`CAMELID_METAL_ATTN_QUAD` / any "quad"/"position-quad" kernel is **absent from this tree**
(grep empty) — that was a separate uncommitted branch and is irrelevant to base+phase0.

Decode-attention dispatch is in `src/metal.rs` (`encode_decode_attention`, ~`:8540`). With the
default fast stack (`CAMELID_METAL_ATTN2=1`, split-K default-on via `splitk_attention_enabled()`
opt-out `CAMELID_METAL_ATTN_SPLITK=0`) and Llama-3.2-1B geometry (n_heads=32, n_kv=8,
head_dim=64, group=4):
- `v2 = attn2_enabled() && head_dim%32==0 && head_dim<=128` → true (tiled
  `attention_decode_v2_pipeline`).
- `splitk = v2 && !kv16 && splitk_enabled && group∈1..=4 && position_count>=128`
  (`src/metal.rs:8554`). **At context ≥ 128 the decode runs `attention_decode_splitk_*`**, with
  `n_splits = position_count.div_ceil(64).clamp(2,64)` (`:8560`); when half-mirrors exist it
  uses `attention_decode_splitk_kv16_direct` (head_dim==128) or `…_splitk_kv16` (head_dim 64),
  else the f32 `attention_decode_splitk_pipeline`. **Below 128 positions** it is the tiled
  `attention_decode_v2_pipeline`.

So a Metal `verify_batch` must reproduce, across its k draft rows with correct causal masking
per row, the **split-K flash math** (online softmax, GQA grouping, `1/sqrt(head_dim)` scale,
n_splits + merge) the single-token `forward_token` uses at depth — i.e. the kv16/f32 split-K
kernel — and the tiled v2 kernel for the shallow rows. (The CUDA verify achieves this via the
two position-aware kernels in `run_batched_layer_stack`, `src/cuda_resident.rs:5219`; the recent
`62ecfce`/`255e479` "split-K spec-verify parity" commits are CUDA-side.)

---

## Phase 3 on-ramp (what must change so serve/bench exercise a Metal `verify_batch`)

The call sites already exist and are **not** CUDA-cfg-gated, so they reach whatever
`verify_*` resolves to on Mac:
`generate_next_tokens_speculative` (`src/main.rs:2559`, bench-speculative),
`verify_tree_gpu` (`src/main.rs:3132`), `verify_drafts_gpu` (`src/main.rs:3259`,
`src/api/mod.rs:8128`). Today they resolve to the `Ok(None)` stubs.

Concretely:
1. **Author `metal::ResidentDecodeState::verify_batch(...)`** (+ a batched layer-stack analog
   of `run_batched_layer_stack`) in `src/metal.rs` / `src/inference/metal_resident.rs`: a
   k-token batched forward over the resident wire weights, causally masked per draft row,
   matching the split-K decode-attention kernel above and the existing resident QKV/FFN/GEMV
   kernels. The per-session engine `self.resident_decode` (`src/inference.rs:1599`) is the
   instance to extend; it is already materialized to `position` by the decode path.
2. **Give `verify_drafts_gpu` (and optionally `verify_tree_gpu` /
   `generate_next_tokens_speculative`) a real macOS body** — i.e. replace the
   `#[cfg(not(feature="cuda"))]` `Ok(None)` stubs (`src/inference.rs:2606-2638`) with a
   `cfg(target_os="macos")` implementation that checks `resident_decode_metal_enabled()` +
   `!resident_paths_disabled` + `resident_decode_eligible(true)` + `engine.filled()==position`,
   then calls the new Metal `verify_batch` and applies the existing longest-accepted-prefix
   logic (identical to the CUDA arm `src/inference.rs:2479-2491`).
3. **No call-site or flag plumbing needed beyond the existing switches.** End-to-end test on
   Mac = `serve` with `CAMELID_SPEC_DECODE=ngram CAMELID_SPEC_GPU=1` (keeps resident on,
   `src/api/mod.rs:6836`, and routes to `verify_drafts_gpu`), or `bench-speculative`
   `CAMELID_SPEC_NGRAM=<γ>` (routes via `generate_next_tokens_speculative`). The existing
   Mac spec-verify parity harness from phase0 is the correctness gate.

Until 1+2 land, Phase 3's Metal kernel work has **no reachable home** on Mac: every spec path
short-circuits to `Ok(None)` and CPU chunk-verify.
