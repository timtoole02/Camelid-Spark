# P0-R1 recon — additive serve lanes vs `generation_lock` (Camelid @ ffada00f)

Question (from the engine-inversion mission brief): the gemma4 / runnable / dg serve
lanes short-circuit out of `chat_completions` BEFORE the generation lock is acquired
(`src/api/mod.rs:7160-7196`). Does each lane either own its own serialization or not
share mutable GPU state with the main engine?

## Answers (file:line evidence at the pin)

Shared facts: lane runtimes live in their own `AppState` maps (`gemma4_runtimes` :88,
`runnable_runtimes` :93, `dg_runtimes` :98); the map `RwLock` guards only the map —
`resolve_*` clones the inner `Arc` out (:4345, :4879, :5299), so the map lock never
serializes generation. All CUDA users share the device PRIMARY context via cudarc
`CudaContext::new` (main engine `src/cuda_resident.rs:2603`, dg
`src/diffusion_gemma/cuda.rs:1220`, gemma4-Cuda via `CudaResidentKernels::new`), but
each owns separate streams and separate device allocations — cross-lane exposure is
VRAM/SM contention, not memory corruption.

| Lane | Backend (default) | Own-request serialization | Shared w/ MAIN engine | Verdict |
|---|---|---|---|---|
| gemma4 Local/Distributed | CPU (rayon) / CPU+TCP | none; per-call KV caches (`src/gemma4_runtime.rs:1699-1700`, `&self`) | none (CPU) | UNSERIALIZED-BUT-ISOLATED |
| gemma4 Cuda | GPU, per-runtime resident (`mod.rs:2635`-region state, own KV `:2671-2677`) | `std::sync::Mutex` held across the WHOLE decode on the blocking thread (`mod.rs:4276-4296`) | separate ctx/buffers; contention only | OWN-SERIALIZED |
| runnable CPU (default) | CPU (rayon), call-local caches (`src/runnable/model.rs:728,1162,1224,1380`) | none; per-call state | none (CPU) | UNSERIALIZED-BUT-ISOLATED |
| runnable CUDA (opt-in `CAMELID_QWEN35_CUDA=1`) | GPU, per-runtime `cuda: Mutex<Option<CudaResidentDecode>>` (`model.rs:364-365`) | guard held across whole prefill+decode (`model.rs:1421-1508`) | separate instance/buffers | OWN-SERIALIZED |
| **dg (CUDA build — default-on win-x86_64 via build.rs)** | GPU through a **process-global** `static ENGINE: OnceLock<Mutex<Option<Engine>>>` (`src/diffusion_gemma/cuda.rs:1160`) | **NONE at any level** — runtime is a bare `Arc<DgServeRuntime>` with no lock (`mod.rs:5275-5285`, handlers :5388/:5471 take no guard); the global engine Mutex is per-KERNEL-OP only (lock sites :1341,:1422,:1528,:1863,:2447,:2547,:2643) | separate singleton from main engine (no main corruption) but **shared across all dg requests**: `Engine::last_logits` handed between ops within a step (`:2646`), `dg_generation_reset` mutates it per block (`:2619-2627`) | **SHARED-AND-UNSERIALIZED (dg-vs-dg)** |
| dg (CPU build) | CPU, call-local buffers | none; per-call state | none | UNSERIALIZED-BUT-ISOLATED |

Extra cross-lane side effect (not corruption): `Gemma4CudaResident::load` calls
`kernels.ctx.disable_event_tracking()` (`src/gemma4_runtime.rs:2737`) on the shared
primary context — a process-wide cudarc bookkeeping toggle worth knowing about.

## Hold-then-detach hazard in the lanes

None. gemma4-Cuda and runnable-CUDA acquire their whole-decode Mutex guards INSIDE the
`spawn_blocking` thread, so a client disconnect that drops the async frame cannot free
the guard out from under an in-flight decode (the opposite failure mode of the main
path: the lane blocking thread simply runs to completion unobserved). Cancellation for
lanes is a Phase-1+ nicety, not a corruption fix.

## Escalation disposition

The dg finding is NOT the brief's strict escalation case (no mutable GPU state shared
with the MAIN engine — the main-engine mission is unaffected and proceeds). It is a
real dg-lane-internal correctness hazard: two concurrent `CAMELID_DG_SERVE=1` requests
on a CUDA build interleave per-kernel-op locks over the shared global `Engine`
(resident canvas / `last_logits`), logically corrupting each other's generations.
Practical exposure is bounded: the lane is env-gated opt-in and a dg block takes
minutes, but the fix (a whole-generation lock, mirroring gemma4-Cuda's
per-runtime Mutex) is small and belongs in its own change, not this mission.
Filed as a follow-up task outside this mission's scope.

Caveats recorded by the recon (honest limits): cudarc internals were not exhaustively
audited for further process-global toggles beyond `disable_event_tracking`; the dg
kernel-op lock pattern was sampled at seven sites, not every helper.
