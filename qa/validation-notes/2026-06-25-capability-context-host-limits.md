# Capability conductor — `context.full_length` host limits + a KV predict-and-abort gap (Windows CPU)

Date: 2026-06-25 · Platform: Windows x86_64 (MSVC), CPU backend · Oracle: llama.cpp `acd79d6`
Scope: the `context.full_length` and `context.rope_scaling` lanes of `MODEL_CAPABILITY_COVERAGE_CONDUCTOR.md`, validated on this development host.

This note records why two `context.full_length` cells (Llama 3.2 1B and 3B) are **NOT** promoted to `done` on this box, and surfaces a real safety gap. It is the honest counterpart to the receipts that *did* land. Per conductor §9 (predict-and-abort) and the runnable-lane memory policy, a length the model genuinely supports but that this host cannot materialize is a **host limit**, not a model limit or a refusal — and it must be documented, never discovered by crashing.

## Host facts

- Total RAM: **15.74 GiB** (16,905,969,664 bytes); ~7 GiB free at measurement.
- Resident Q8_0 weights (default lazy file-backed): TinyLlama ~1.17 GB, Llama 3.2 1B ~1.32 GB, Llama 3.2 3B ~3.42 GB.
- CPU KV cache is **f32** (`src/inference/kv_cache.rs`), shape `[layers, seq, kv_heads, head_dim]` for each of K and V. Per-token bytes = `2 · n_layers · n_kv_heads · head_dim · 4`.

| Row | n_layers | n_kv_heads | head_dim | KV bytes/token | trained ctx | KV at trained ctx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TinyLlama 1.1B | 22 | 4 | 64 | 45,056 (44 KiB) | 2048 | **88 MiB** ✅ |
| Llama 3.2 1B | 16 | 8 | 64 | 65,536 (64 KiB) | 131072 | **8.0 GiB** ⚠ no headroom |
| Llama 3.2 3B | 28 | 8 | 128 | 229,376 (224 KiB) | 131072 | **28.0 GiB** ❌ infeasible |

(The 3B figure corrects an in-flight reader error that had claimed 1.75 MiB/token; the `inspect`-measured head counts give 224 KiB/token, confirmed by recompute. This is load-bearing: it is why 3B rope-scaling at >8192 positions IS feasible here, while 3B *full* 131072 is not.)

## What was validated (class D, bit-exact vs llama.cpp `acd79d6`)

Harness `qa/capability/context_parity.mjs`: a long prompt is decoded greedily; Camelid emits a server-sealed `camelid.parity-receipt/v1`; `camelid verify-receipt` feeds Camelid's exact prompt token ids to llama.cpp `/completion` and asserts the continuation matches bit-exact (prompt PINNED → proves KV + rope correctness across the whole context, token-for-token). The harness itself projects KV bytes and **aborts before requesting an unsafe context** — the host is never discovered by crashing it.

- **TinyLlama `context.full_length` → done.** Bit-exact at **1953 tokens (95% of the trained 2048)**; full-length KV ≈ 88 MiB, fully reachable. Receipt: `capability-receipt.context.full_length.tinyllama-1.1b-chat-q8_0.json`.
- **Llama 3.2 1B `context.rope_scaling` → done.** Bit-exact at **8511 tokens (> 8192, the llama3-scaled regime; full self+reference verify)**. Receipt: `capability-receipt.context.rope_scaling.llama-3.2-1b-instruct-q8_0.json`.
- **Llama 3.2 3B `context.rope_scaling` → done.** Bit-exact at **8304 tokens (> 8192; reference-only verify — 1.9 GiB KV)**. Receipt: `capability-receipt.context.rope_scaling.llama-3.2-3b-instruct-q8_0.json`.

## What stays `wip` (host-limited, NOT a model limit)

- **Llama 3.2 1B `context.full_length`** — validated bit-exact across a long context (feasible frontier, **8511 tokens**; also an earlier 7958-token run), but the **trained 131072** materializes an 8.0 GiB f32 KV cache, which fits total RAM only with no safe headroom (and exceeds free memory). Not validated at the trained length on this box. Needs a larger-RAM host.
- **Llama 3.2 3B `context.full_length`** — validated bit-exact across a long context (feasible frontier, **8304 tokens**), but the **trained 131072** needs a **28 GiB** f32 KV cache — ~12 GiB beyond this box's entire RAM. Decisively infeasible here.

These two cells remain `wip` with this host-limit annotation. They are NOT `n/a` (the models genuinely support 131072) and NOT `done` (unvalidated at the trained length here).

## Safety gap surfaced (conductor §9)

Camelid has **no pre-flight KV predict-and-abort**. The context cap is the GGUF trained length (`model.rs:141` → `kv_cache.rs:23`), but the KV cache grows **lazily/incrementally** (`kv_cache.rs:135-136`, `keys.resize`/`values.resize`, chunks of `CAMELID_KV_CACHE_GROW_TOKENS`=256). A request approaching 131072 on a host that cannot hold the projected KV would **OOM mid-generation**, not fail closed. The existing weight-materialization guard (`CAMELID_MAX_CPU_WEIGHT_MATERIALIZATION_BYTES`, default 6 GiB, `api/mod.rs:6533`) covers **weights only** and estimates ≈0 for lazy-Q8 file-backed linears, so it never fires for KV.

This is exactly the rail conductor §9 calls load-bearing on Windows ("never discover the ceiling by crashing the host"). The `context_parity.mjs` harness modelled the missing guard externally (projects KV, aborts before request).

**Update — the in-engine guard is now implemented** (`src/inference/kv_cache.rs`): `ensure_position_capacity` projects the bytes of the actual (post-rounding) KV growth and refuses with a typed `RuntimeShapeMismatch` *before* the `resize`, so an over-budget context fails closed instead of OOMing mid-generation. The per-session budget is `CAMELID_MAX_KV_CACHE_BYTES` (explicit override, bytes) or, absent that, **80% of available physical RAM** (Windows `GlobalMemoryStatusEx` via `crate::gait::host_ram_status`); it is unbounded only where neither is known (e.g. off Windows), where the env override remains the gate. The budget is host-derived operational config and is excluded from KV-cache state equality. Unit tests in `kv_cache.rs` cover the per-token byte math, the refuse-before-allocate behavior, the budget policy, and the equality exclusion. This does **not** make the trained 131072 context reachable on this box — it makes the host limit a clean refusal rather than a crash.

## macOS — auto-budget now engages (WIN2METAL Phase 1, Bucket B)

Date: 2026-06-26 · Platform: macOS arm64 (Apple M4, 16 GiB), CPU reference + Metal-resident backends · Host: development Mac mini.

Before this change `gait::host_ram_status()` returned `None` on every non-Windows target, so the KV auto-budget was **inert on Mac** — with no `CAMELID_MAX_KV_CACHE_BYTES` set, the budget resolved to `u64::MAX` (unbounded) and only the env override gated. macOS now has a real `host_ram_status()` (`src/gait/mod.rs`): **total** from `sysctl hw.memsize`, **available** from the Mach VM statistics as `(free + inactive)` resident pages × the VM page size (counting only `free` would badly understate headroom under the memory compressor, which keeps the free pool small; the cold, reclaimable `inactive` pages are the rest of the working set, and the 80% factor adds further headroom). `host_ram_status` returns `None` only on query failure. The remaining unixes keep `None`.

| Row | n_layers | n_kv_heads | head_dim | KV bytes/token | host avail (sampled) | 80% auto-budget | tokens at budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama 3.2 1B | 16 | 8 | 64 | 65,536 (64 KiB) | ~5.3 GiB | ~4.25 GiB | ~69,500 |

So on this 16 GiB box the no-env auto-budget now refuses a context well below the trained 131072 (whose 8.0 GiB f32 KV exceeds 80% of available) — a clean typed refusal, not a crash. Validated 2026-06-26:

- **Unit (this host).** `gait::gait_safety::host_ram_status_reports_live_physical_ram` asserts `Some` with `total>0 && available>0 && available<=total`; `inference::kv_cache::tests::macos_ram_branch_engages_auto_budget` runs the real `resolve_kv_cache_budget_bytes()` with no env and asserts a bounded (`< u64::MAX`) budget equal to 80% of available — the off-platform unbounded fallback is gone on Mac. Both pass.
- **e2e (a) — typed error, not a crash.** `serve` on the CPU reference path with a tiny `CAMELID_MAX_KV_CACHE_BYTES=100000`: a `/v1/completions` request returns **HTTP 503** `runtime_unavailable / generation_step_failed`, body `"KV cache growth to 256 positions needs 16777216 bytes of f32 K+V, above the 100000 byte budget for this host; reduce the prompt/context length or set CAMELID_MAX_KV_CACHE_BYTES deliberately ..."`. The server **survives** — subsequent `/v1/health` and `/v1/models` return 200 and a normal request still generates.
- **e2e (b) — auto-budget live without env.** With no override, normal requests serve fine (the ~4.25 GiB budget is generous), and the resolver-level proof above confirms the budget is the RAM-derived value, not `u64::MAX`. A purely behavioral large-context refusal was not forced on this box: the prefill **activations** for a ~90k-token prompt (several GiB) thrash the 16 GiB host before the KV-byte projection is reached, and the unoptimized reference prefill over tens of thousands of tokens is prohibitively slow — an orthogonal limit, not a guard failure.

**Scope / honest caveat.** The guard lives in the CPU `LlamaKvCache` (`src/inference.rs` prefill/decode → `ensure_position_capacity`). That is the path exercised above (`cpu_reference` / `cpu_q8_runtime_repack` with resident paths disabled). The **default** macOS runtime is `metal_resident_q8_runtime`, which keeps the KV cache **on the GPU** and advances `position` without growing the CPU `LlamaKvCache`, so it does **not** pass through this guard — a resident-path KV ceiling is a separate concern (a later bucket), not addressed here. Bucket B's deliverable is the real `host_ram_status()` reading itself, which makes the auto-budget finite and active on Mac wherever the CPU KV cache is the authority.
