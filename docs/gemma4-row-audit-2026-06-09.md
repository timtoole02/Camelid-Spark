# Gemma 4 row audit — 2026-06-09

Scope: establish the current evidence-backed truth for every practically available
Gemma 4 text-generation GGUF row before any new support claim is made. This audit
was produced on branch `feature/gemma4-performance-support` (cut from `main` at
53bbb6a, which includes the merged `feat/gemma4-engine-support` work).

## 1. What is true today (pre-branch)

### Code

- `general.architecture = "gemma4"` is detected and bound (`src/model.rs`):
  per-layer-type head dims (sliding/global), dual RoPE bases and rotary dims,
  sliding-window schedule (5:1 + final-global), cross-layer shared KV,
  Per-Layer-Embeddings (PLE) for E-series, final logit soft-capping, QK-norm.
- `Gemma4Binding` validates **dense FFN only** (`ffn_gate`/`ffn_up`/`ffn_down`).
  There is no router/expert binding: a Gemma 4 MoE GGUF (26B A4B) fails with a
  generic missing-tensor error, not a typed blocker.
- CPU runtime (`src/gemma4_runtime.rs`): wire-backed Q8_0 mmap load, greedy
  decode (deterministic argmax), streaming via cumulative decode.
- GPU runtime (`Gemma4GpuRuntime` + `metal::Gemma4ResidentModel`): full resident
  decode graph, one command buffer per token, KV on GPU, nocopy wire pages.
- API: `/v1/chat/completions` (non-streaming + SSE) behind `CAMELID_GEMMA4_SERVE=1`.
  No `/v1/completions` route. No typed rejection of multimodal content parts.
- Frontend: one catalog row (`gemma4_e4b_it_q8_0`); chat unlock requires
  runtime-loaded + generation-ready + contract-supported (fail-closed gate works).

### Evidence

| Row | Load | CPU greedy parity vs llama.cpp | GPU parity | API smoke | Evidence bundle |
|---|---|---|---|---|---|
| E4B-it Q8_0 | yes | yes — `The capital of France is` → ids `[9079, 236761, 108, 1018, 14977, 53121, 2900, 563, 506, 5279, 529, 7001]` | yes (token-identical to CPU) | chat (stream + non-stream), manual | **none committed** |
| every other row | never attempted | none | none | none | none |

- `qa/evidence-bundles/` contains **zero** gemma4 bundles (120 dirs, none gemma).
- The only committed parity artifact is the env-gated `tests/gemma4_forward.rs`
  prefill/teacher-forced check plus prose in `docs/gemma4-engine-status.md`.
- Perf on M4 16GB (E4B): CPU ~6.75 tok/s warm (NEON sdot), GPU ~11.2 tok/s
  (~118 GB/s, at the ~120 GB/s bandwidth wall; ceiling for 8 GB/token ≈ 13).

### Docs

- README lists "Gemma 4 (E4B) — Experimental". COMPATIBILITY/SUPPORT_MATRIX/
  ROADMAP still only speak of **Gemma 2 9B** as a "planned exact-row candidate".
  No doc claims Gemma 4 support; no contradiction found, but the support
  surfaces have not caught up with the engine work either.

## 2. Practically available Gemma 4 text rows (checked 2026-06-09, HF API)

| Row | Repo | Q8_0 size | Fits 16 GB Mac | Fits 2×16 GB | Status here |
|---|---|---|---|---|---|
| E2B-it | unsloth/gemma-4-E2B-it-GGUF | 5.05 GB | yes | — | downloaded this session |
| E4B-it | unsloth/gemma-4-E4B-it-GGUF | 8.19 GB | yes (tight; embeddings must stay file-backed) | — | on disk, validated |
| 12B-it | unsloth/gemma-4-12b-it-GGUF | 12.67 GB | **no** (GPU-resident decode would thrash) | yes (~6.3 GB weights/node) | downloading; two-Mac target |
| 26B A4B-it (MoE) | unsloth/gemma-4-26B-A4B-it-GGUF | 26.86 GB | no | **no** (~13.4 GB weights/node + KV + OS on 16 GB boxes → thrash) | blocked: memory + no MoE binding/runtime |
| 31B-it (dense) | unsloth/gemma-4-31B-it-GGUF | 32.64 GB | no | **no** (~16.3 GB weights/node alone exceeds budget) | blocked: memory |
| base (non-it) rows | google/* and unsloth non-it repos | — | — | — | gated (HTTP 401); not practically available |
| MTP/assistant rows | `MTP/gemma-4-*-Q8_0-MTP.gguf` | 0.10–0.51 GB | yes | — | blocked: runtime semantics (see below) |

### MTP/assistant row tensor reality (inspected `gemma-4-E2B-it-Q8_0-MTP.gguf`,
sha256 `9eba8199…1919`)

- `general.architecture = "gemma4-assistant"` (distinct arch, 4 layers).
- `nextn_predict_layers = 4`, `embedding_length_out = 1536` (couples the head
  to the host model's hidden width), `embedding_length = 256`.
- Per-layer `layer_output_scale` scalar tensors; `attn_q`/`attn_q_norm` only —
  **no K/V projections in the file**; `shared_kv_layers = 4` of 4, i.e. KV is
  sourced from the host model under an undocumented contract.
- `nextn.pre_projection [3072, 256]`, `nextn.post_projection [256, 1536]`.
- Verdict: tensor map is parseable, but the generation semantics (host hidden
  handoff, KV sourcing, acceptance rule) are not documented anywhere we can
  verify against. Per repo policy this stays **fail-closed** with a typed
  blocker until an oracle exists to prove lossless speculative decode.

## 3. Gaps this branch must close (smallest honest set)

1. **Fail-closed guards**: typed unsupported errors for (a) multimodal content
   parts on gemma4 chat routes, (b) gemma4 MoE tensors (router/experts), and
   (c) `gemma4-assistant` MTP rows — each naming exactly what is missing.
2. **Row-aware tests**: `gemma4_metadata`, `gemma4_binding`,
   `gemma4_generation_parity` (env-gated per exact row), `gemma4_api_smoke`,
   `gemma4_capabilities`; prompt packs under `qa/gemma4/prompt_packs/`.
3. **E2B-it Q8_0 promotion**: load, CPU+GPU greedy parity vs llama.cpp, API
   smoke, capabilities/frontend row, evidence bundle.
4. **E4B-it Q8_0 bundle**: the parity that already exists must become a
   committed evidence bundle or it does not count.
5. **CPU decode perf**: per-token allocation waste and the 262K-vocab output
   projection are the measured hot spots; any change must preserve greedy ids.
6. **Two-Mac distributed lane**: gemma4 is single-node only today. The honest
   claim to build toward is *distributed layer sharding* (never "shared
   memory"): E4B split first to prove parity, then 12B-it as the row that a
   single 16 GB Mac cannot serve.
7. **Docs**: support surfaces updated only after bundles exist, with exact-row
   wording and explicit non-claims (no multimodal, no family-wide support).

## 4. Claim boundaries (unchanged until evidence lands)

- "Camelid supports Gemma 4" — **not claimable** (only exact rows ever will be).
- Multimodal Gemma 4 — **not claimable**; inputs must fail closed.
- 26B A4B / 31B — **blocked**, with memory math recorded above.
- Long context — nothing beyond the tested prompt envelope is claimable;
  bounded context packs (512→8192) come before any 32K+ roadmap entry.
