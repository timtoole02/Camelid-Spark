# FLINT_WINDOWS_CONDUCTOR.md — validate the Spark speed work on the RTX 3060 (TEST BUILD)

> **STATUS — RUN COMPLETE 2026-07-22 (PR #8 → main 9ae323a; arm64+cuda CI green).** W0–W4 all
> sealed on this box; only the Spark tester's leg remains (see "Done when"). Execution deltas vs
> the plan as written below:
>
> - **Premise corrections (W1 recon):** `WirePages` is a page-aligned HEAP allocation, not an
>   mmap (the `GgufWireMmap`/`q8_0_wire_mmap` plumbing is dormant, never populated), and NO Q8_0
>   resident lane is passthrough — `q8_gemv`/`q8_gemm_batched` read a widened f32-scale SoA
>   layout, so the lane registers **repacked** page-aligned buffers (this doc's own sanctioned
>   alternative in §W2). The NOCOPY gate anchor is `inference.rs:11878`, not `:11816`.
> - **1× streaming build (W2):** the first wire-pages variant held ~2× host RAM (retained wire +
>   registered SoA) — a 70B build would have peaked ~151 GB of the Spark's 128 GB and never
>   loaded. Shipped instead: Q8_0 projections load file-backed only and the builder streams each
>   tensor disk → transient → its registered SoA buffer (ONE steady copy; the token embedding
>   alone keeps wire pages for the CPU's per-token row reads). 70B projects ~78.2 GB SoA /
>   ~80.5 GB peak. Verified on-box: hostreg peak 1.247 GB vs 1.256 GB flag-off control.
> - **W3:** token-identical across the full matrix (TinyLlama + qwen3 0.6B/1.7B × plain /
>   batched-prefill / graphs / spec / CPU-fallback; 197-155-113 zero-copy, 0 uploads anywhere) —
>   `qa/evidence-bundles/flint-w3-win3060-20260722-head-c50aa93/W3-PARITY-VERDICT.md`. Bonus W0
>   finds: **CUDA graph capture now WORKS under WDDM** on driver 576.83 (the 2026-07-03 broken-
>   capture STATUS comment near `cuda_resident.rs:7454` is stale), and S2's `spec_gpu_enabled`
>   flip is NOT unified-gated (deliberate; disclosed in the W0 verdict).
> - **W4 + pre-merge adversarial review (17 findings, 9 confirmed, all fixed pre-merge):** the
>   unset default requires the INTEGRATED attribute **or a ≥ 96 GiB pool** — NOT bare
>   `cuda_unified_memory`, whose VRAM≈RAM heuristic can misfire on discrete 8/8–24/24 boxes; MoE
>   models keep the historical loaders (resident admission always declines them); the loader gate
>   mirrors the decode gate, with a loud once-per-process warning if hostreg-loaded weights ever
>   hit the CPU file-reader path; dist-worker/dist-master pin the flag off; and the fit advisor
>   deliberately KEEPS the 2× unified margin (it is quant/arch-blind, so the 70B row honestly
>   reads *unknown* — the load path is the authority and admits it).

**Goal:** you are on the engine's original CUDA reference machine — Windows, RTX 3060 Laptop
6 GB, driver 576.83, CUDA 12.9 — the exact card every existing `windows-cuda-resident-parity`
evidence bundle was captured on. Two jobs, in order:

1. **Regression-gate** the Spark changes (PRs #6/#7, `SPARK_SPEED.md`) on the DISCRETE path —
   they were written for unified memory and must be no-ops here.
2. **Build the zero-copy weight lane** (`CAMELID_CUDA_HOSTREG`) and prove it **bit-exact** on
   this card. Discrete GPUs read host-registered memory over PCIe — slow, but *correct*, which
   is precisely what this box can prove and the Mac (no CUDA device) could not. The DGX Spark
   then only has to answer the perf question. This lane is what unlocks 70B Q8_0 on the Spark
   (see `SPARK_SPEED.md` §"Stage S2-deep").

This is a TEST BUILD repo (`github.com/timtoole02/Camelid-Spark`). Everything lands here as
PRs against `main` — do NOT touch the upstream Camelid repo. House rules: no kernel changes
(parity charter — the kernels must stay byte-identical); every new behavior gets an env escape
hatch; run `cargo test --all-targets` before any push (lib/bin runs miss `tests/`); commit
messages stay plain (no co-author trailers, no comparative-engine talk); CI here is
arm64-Linux — your Windows build is a local signal only, and the release build ≠ `cargo check`
(fullfp16-class issues only surface in release codegen).

## Phase W0 — build + regression gate [AGENT]

```powershell
git clone https://github.com/timtoole02/Camelid-Spark
cd Camelid-Spark
cargo build --release          # CUDA is default-on for Windows via build.rs
cargo test --all-targets --release
```

Then verify the Spark changes are inert on discrete, with receipts:

* `.\target\release\camelid.exe serve` startup banner: `[hw] GPU: ... | VRAM ...` — must say
  **VRAM**, NOT "UNIFIED memory" (`HardwareProfile.cuda_unified_memory` must be `false` on a
  6 GB card next to ≥16 GB RAM; the probe is `src/capability.rs::detect`, attribute read in
  `src/cuda.rs::probe_capability` — `CU_DEVICE_ATTRIBUTE_INTEGRATED`, S1 added it and this box
  is its first run on real CUDA).
* Offload split still available on discrete: the auto-split branch in `src/inference.rs`
  (search `"none-unified"`) must only bypass on unified. `CAMELID_OFFLOAD_FORCE_LAYERS=2` on a
  small model should still offload (test hook fires everywhere).
* gemma4 defaults: `CAMELID_GEMMA4_SERVE`/`CAMELID_GEMMA4_CUDA` now default **ON when CUDA is
  present** — that includes this box (previously opt-in). The E2B row is parity-validated on
  this exact card (`tests/gemma4_generation_parity.rs`, CUDA branch): if the E2B Q8_0 GGUF is
  on disk, run it. KV cap must stay **4096** here (the 32768 default is unified-only; env
  `CAMELID_GEMMA4_KV_CAP` overrides). If this default flip is unwanted on this box, `=0` both.
* CUDA graphs must still be **opt-in** on Windows (the Linux-only default flip is in
  `src/cuda_resident.rs::cuda_graphs_enabled`).
* Re-run whichever CUDA parity suites have their model files on this box (the qwen3
  0.6B/1.7B/4B rows and TinyLlama are the historical ones). Any regression vs. the pinned
  evidence bundles is a STOP — report before proceeding.

**Gate W0:** tests green, banner says VRAM, at least one CUDA parity suite green on a real GGUF.

## Phase W1 — recon the zero-copy surface [AGENT, blind]

Confirm with file:line before writing code (anchors verified at `d6dc58c`):

* The Metal fast-load gate to twin: `src/inference.rs:11816`
  `fn metal_nocopy_fast_load_enabled() -> bool { cfg!(target_os = "macos") && env_flag_enabled("CAMELID_METAL_NOCOPY") }`
  and its load-path consumer near `src/inference.rs:464` ("loading Q8_0 weights as page-aligned
  wire pages"). This is the pattern: Q8_0 linears keep their PAGE-ALIGNED wire bytes mmap'd
  instead of materializing blocks.
* The backing infra (all reusable as-is): `src/wire_mmap.rs` (`GgufWireMmap::map`, page_size,
  advise_*), `CpuTensor.q8_0_wire_pages: Option<Arc<WirePages>>` (`src/tensor/mod.rs:1149`),
  the `load_q8_0_wire_pages_linear` loader, and the tied-embedding page sharing.
* The upload sites to bypass: `src/cuda_resident.rs` `build_...` upload closures —
  `clone_htod` at ~5830/5840 (projections, with `repack_for_lane`!), ~6023/6025 (final norm,
  output weight). **Key question for W1:** which projections go through `repack_for_lane`
  non-trivially? A repacked lane CANNOT zero-copy the raw wire bytes — those must either keep
  the upload path or register a repacked page-aligned host buffer instead. Map exactly which
  lanes are passthrough (wire bytes usable in place) vs repacked.
* cudarc 0.19 host-register surface: check `cudarc::driver` for safe wrappers; else use
  `cudarc::driver::sys` raw (`cuMemHostRegister_v2`, flag `CU_MEMHOSTREGISTER_DEVICEMAP`,
  `cuMemHostGetDevicePointer_v2`, `cuMemHostUnregister`). The crate is pinned
  `0.19`/`cuda-12090`/`fallback-dynamic-loading`.
* How engine weight fields are typed (CudaSlice?) — the mapped-host path needs a second
  representation (owned `CudaSlice` vs borrowed device pointer + registration guard). Find the
  narrowest seam: ideally the projection-slice handle, not every kernel call site.

## Phase W2 — implement `CAMELID_CUDA_HOSTREG` [AGENT]

* Env gate, **default OFF in this phase**: `CAMELID_CUDA_HOSTREG` (1/true/on/yes ⇒ on;
  0/false/off/no ⇒ off; unset ⇒ off for now — the unified-default flip is Phase W4).
* When on (and the model/lane qualifies): load Q8_0 linears via the wire-pages path (the
  NOCOPY loaders — lift the `cfg!(macos)` gate at `inference.rs:11816` into a
  platform-appropriate disjunction rather than duplicating the loader), `cuMemHostRegister`
  the page-aligned regions with DEVICEMAP, fetch device pointers, and hand the resident
  engine those instead of `clone_htod` copies. Registration lifetime = engine lifetime
  (unregister on drop; mind Drop order vs the mmap Arc).
* Scope discipline: **Q8_0 passthrough lanes first.** Repacked lanes and K-quant wire come
  later or register a repacked host-side page-aligned buffer (still saves the device copy) —
  do NOT contort the kernels. Anything that can't zero-copy falls back to the upload path
  silently per-tensor, and the engine records which path each projection took (one log line
  at build: `N zero-copy / M uploaded`).
* Fit interplay: when HOSTREG will drive the load, the fit advisor's unified 2× multiplier is
  wrong (footprint becomes ~1×) — leave the advisor alone in this PR, note it in
  SPARK_SPEED.md for the flip PR.

**Gate W2:** `cargo test --all-targets --release` green with the flag off AND on (CPU-only
suites unaffected); a small Q8_0 model decodes under `CAMELID_CUDA_HOSTREG=1`.

## Phase W3 — parity proof on the 3060 [AGENT]

The whole point. On at least TinyLlama Q8_0 + one qwen3 row (0.6B or 1.7B):

1. Greedy 50+ tokens, `CAMELID_CUDA_HOSTREG=0` vs `=1`: **token-identical** (the repo's parity
   bar). Byte-compare logits if a dump harness is handy; token-IDs at depth 50 otherwise.
2. Cross the matrix: batched prefill on/off (`CAMELID_CUDA_RESIDENT_PREFILL_BATCHED`),
   spec verify (`CAMELID_SPEC_GPU=1` + ngram), CUDA graphs `=1`. All token-identical.
3. Run the existing cuda_resident parity test suite with the env set, if it inherits env.
4. Record decode tok/s both ways and DON'T gate on it: hostreg on discrete goes over PCIe and
   is expected slower — note the numbers in the PR as evidence the lane exercised the mapped
   path (if tok/s is identical, suspect the flag silently didn't engage).

**Gate W3:** parity table in the PR body (model × mode × 0/1 → identical), with the
zero-copy/uploaded projection counts from the build log.

## Phase W4 — default-on-unified + ship [AGENT]

* Flip the unset-default to: on when `HardwareProfile::cached().cuda_unified_memory`, off
  otherwise (exact shape of the S1/S2 gates). Discrete boxes keep upload unless opted in.
* Adjust the fit advisor's unified multiplier 2×→1× when hostreg will apply (or leave 2× and
  note it — DECIDE in the PR, either is defensible for a test build; say which and why).
* Update `SPARK_SPEED.md` (move the S2-deep row to "shipped — perf pending Spark") and
  `FLINT_HANDOFF.md` (70B Q8_0: expected to load fully resident now; ask the Spark tester for
  the `zero-copy/uploaded` log line + tok/s).
* PR to `main`; the arm64 CI must be green (it compile-gates the Linux side of your cfg work);
  squash-merge on green.

## Done when

1. ✅ W0 regression receipts recorded — `qa/evidence-bundles/flint-w0-win3060-20260721-head-bb07814/`
   (W0-VERDICT.md: suites green, banner VRAM, gemma4 E2B CUDA parity 5/5 oracle-identical). [AGENT]
2. ✅ `CAMELID_CUDA_HOSTREG` implemented — default off → INTEGRATED-or-≥96GiB-on, per-tensor
   silent fallback, `[cuda] hostreg: N zero-copy / M uploaded` engagement receipt. [AGENT]
3. ✅ W3 parity table token-identical across the matrix on this card
   (`qa/evidence-bundles/flint-w3-win3060-20260722-head-c50aa93/W3-PARITY-VERDICT.md`). [AGENT]
4. ✅ PR #8 squash-merged, arm64+cuda CI green (main 9ae323a); SPARK_SPEED.md (S2-deep → SHIPPED,
   perf pending) + FLINT_HANDOFF.md updated. [AGENT]
5. ⏳ The Spark tester reports, per the FLINT_HANDOFF.md 70B section: [TESTER — the other machine]
   - the 70B Q8_0 load-time stderr lines `[cuda] hostreg attrs: …` and
     `[cuda] hostreg: N zero-copy / M uploaded` (expect N = 561 = 80×7 + head; any M > 0 means
     the per-tensor fallback engaged — send the line either way);
   - greedy decode tok/s on the 70B Q8_0;
   - an upload-vs-hostreg A/B **on the 14B Q8_0 row** (`CAMELID_CUDA_HOSTREG=0` vs `=1`, short
     greedy — token streams must be identical). A `=0` leg on the **70B** lands on the CPU lane
     (the pre-hostreg posture; ~2× cannot fit the pool and unified refuses offload) — still
     expected token-identical, just minutes-slow;
   - watch item: one unreproduced >10-min `spec_draft_rollback` stall right after heavy
     register/unregister churn on the 3060 (W3 verdict, anomaly 2) — report if it reproduces.

Known traps, from this campaign's history: `cargo check` ≠ release build on new
targets; `with_extension` mangles dotted stems (append suffixes instead); test literals of
changed structs live in `src/api/mod.rs`/`src/fit.rs` test mods; the T7-era lesson applies to
you too — always `--all-targets`. If `cuMemHostRegister` fights WDDM on some region, register
smaller aligned windows per tensor rather than the whole mmap; the parity gate is the truth.
