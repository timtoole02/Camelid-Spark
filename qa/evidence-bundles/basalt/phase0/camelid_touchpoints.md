# BASALT Phase 0 — Camelid-side integration touchpoints (raw receipts)

Recon target: `<camelid>` at `main` HEAD `4f9603f0` (Merge PR #465), read-only.
Every claim below carries a file:line receipt against that tree. Collected 2026-07-16.

---

## 1. GGUF reader enum + fail-closed path

### 1.1 `GgufTensorType` enum
`src/gguf/reader.rs:35-62` — derives `Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash` (line 35). Variants (36-61):
`F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2K, Q3K, Q4K, Q5K, Q6K, Q8K, IQ4NL, IQ4XS, Tq1_0, Tq2_0, I8, I16, I32, I64, F64, BF16, Unknown(i32)`.

### 1.2 `from_id` (ggml wire ids)
`src/gguf/reader.rs:65-93` — explicit map: 0=F32, 1=F16, 2=Q4_0, 3=Q4_1, 6=Q5_0, 7=Q5_1, 8=Q8_0, 9=Q8_1, 10=Q2K, 11=Q3K, 12=Q4K, 13=Q5K, 14=Q6K, 15=Q8K, 20=IQ4NL, 23=IQ4XS, 34=Tq1_0, 35=Tq2_0, 24=I8, 25=I16, 26=I32, 27=I64, 28=F64, 30=BF16; `other => Self::Unknown(other)` at reader.rs:91. **NVFP4 (id 40) today → `Unknown(40)`.**

### 1.3 `layout()` — (block_size, type_size)
`src/gguf/reader.rs:95-125`; `Self::Unknown(_) => None` at reader.rs:123. For BASALT: NVFP4 layout would be `(64, 36)` per ggml-common.h (QK_NVFP4=64, block=36 bytes).

### 1.4 Fail-closed hop chain for an unknown tensor type
1. Parse: `read_metadata` tensor-index loop calls `tensor_nbytes` at `src/gguf/reader.rs:442`.
2. `tensor_nbytes` (`reader.rs:530-535`): `tensor_type.layout().ok_or_else(|| BackendError::UnsupportedGguf(format!("tensor {name} has unknown or removed GGML type {tensor_type:?}")))` — **an NVFP4 file fails AT PARSE**, before admission ever runs. Error text would read `... unknown or removed GGML type Unknown(40)`.
3. If the type IS parseable but not covered: runnable admission `src/runnable/admit.rs:119-128` (`admit`) → `check_quants` `admit.rs:192-210` → `is_covered_quant` `admit.rs:177-190` (covered: F32, F16, Q8_0, Q6K, Q5K, Q4K, Q3K, Q4_0, IQ4XS). Reject is a structured `AdmissionReject { axis, offending_value, tensor, message }` (`admit.rs:75-86`), `offending_value: format!("{:?}", tensor.tensor_type)` at `admit.rs:198`, message lists the covered set at `admit.rs:200-204`. `From<AdmissionReject> for BackendError` → `UnsupportedGguf` at `admit.rs:99-103`.
4. API surface of the refusal: `GET /api/models/local` per-file `admitted`/`admission_reason` (`src/api/mod.rs:17678-17681`); calls `crate::runnable::admit(&gguf)` at `mod.rs:17808`; a parse failure (the NVFP4 case today) lands as `admission_reason = Some(format!("GGUF parse failed: {err}"))` at `mod.rs:17828`.
5. Dequant fails closed independently even if admission were bypassed: `src/runnable/dequant.rs:39-43` `other => Err(BackendError::UnsupportedTensorType(...))`; module doc `dequant.rs:10-12` "admission should already have rejected it, but dequant fails closed regardless".
6. gemma4 wire lane fails closed at load: `WireQuant::new` `src/gemma4_runtime.rs:87-99` — any type outside `WireFormat` → `BackendError::UnsupportedTensorType("tensor {name} is {other:?}; gemma4 wire load supports Q8_0, Q4_0, Q4_1, Q4_K, Q5_K, and Q6_K")` (doc at `gemma4_runtime.rs:76`).

### 1.5 Consumers that serialize/compare tensor-type NAMES (receipt-disturbance surface)
- `src/receipt/mod.rs:201-216` `quantization_label`: prefers `general.file_type` ftype label (`declared_file_type_label`, mod.rs:221-225; ftype map mod.rs:235-256 — **has NO entry for ftype 26/NVFP4**, returns None), else falls back to the dominant tensor type's **Debug name** `format!("{:?}", tensor.tensor_type)` at `mod.rs:208`. This string lands in `lane.quantization` of every parity receipt (RECEIPTS.md schema) and is digested into `receipt_id` — existing receipts are untouched by ADDING a variant, but a new NVFP4 file emitted BEFORE the enum variant exists would label as `Unknown(40)`; after, as the new variant's Debug name. Choose the variant name once (e.g. `Nvfp4` → label "Nvfp4") and never rename: renames change future receipt bodies, not old ones (old receipts verify against their own recorded strings).
- `src/runnable/smoke.rs:157-171` `headline_quant`: `format!("{tt:?}")` → smoke receipt `quantization` (e.g. `qa/runnable/smoke/Qwen3-0.6B-Q8_0.json` has `"quantization": "Q8_0"`).
- `src/runnable/admit.rs:198,202` — Debug name in `AdmissionReject.offending_value` + covered-set list in the message (message text also asserted by tests `admit.rs:298-370`; `tests/runnable_admission.rs` exists).
- `GgufTensorType` derives `Serialize`/`Deserialize` (`reader.rs:35`) — serde external tagging serializes unit variants as their name strings; `AdmissionOk.quants: BTreeSet<GgufTensorType>` (`admit.rs:108-112`) is `Serialize`. `GgufTensorDescriptor` (contains `tensor_type`) is `Serialize` (`reader.rs:128-136`) and is dumped verbatim by `camelid inspect` (`src/main.rs:1409-1412`, `serde_json::to_string_pretty(&gguf)`).
- `src/receipt/verify.rs` itself does NOT re-derive/compare the quantization name (only test fixture at verify.rs:1019 uses `"Q4_K"`); verify checks are: self-digest, reproducibility gate, lane identity via `lane.gguf_sha256` (`verify.rs:542-560`), Camelid re-run token compare (`compare_results` verify.rs:563), reference re-run (`reference_rerun` verify.rs:665). Quality-tier `format`/`baseline_format` are free-form `String`s (`src/receipt/mod.rs:462-472`) checked only for lossy same-format honesty (`verify.rs:101-128`).
- `src/execution_plan.rs:307-309, 1049-1064` — matches on enum values (Q8_0/Q4K/Q6K presence) to pick plan labels; a new variant simply doesn't match (falls to default plan), no name serialization.

---

## 2. Conventions to mirror

### 2.1 `decode_q4_k_tensor` (crate::tensor)
`src/tensor/mod.rs:5350-5364`:
```rust
pub(crate) fn decode_q4_k_tensor(
    name: &str,
    bytes: &[u8],
    expected_elements: usize,
) -> Result<Vec<f32>> {
    let blocks = decode_q4_k_blocks(bytes)
        .map_err(|e| BackendError::InvalidTensorData(format!("{name}: {e}")))?;
    ...
}
```
Error posture: decode error → `BackendError::InvalidTensorData` with the tensor name prefixed; siblings `decode_q3_k_tensor` (5334-5348) and `decode_q5_k_tensor` (5366-) are shape-identical. Re-exported to the runnable lane via `crate::tensor` (imported in `src/runnable/dequant.rs:16-19`).

### 2.2 `q4_k_wire_row_dot` (crate::inference)
`src/inference.rs:19236`:
```rust
pub(crate) fn q4_k_wire_row_dot(weight_wire: &[u8], input: &[Q8KBlock]) -> f32
```
- **No per-tensor scale parameter** — confirmed: only wire bytes + pre-quantized activation; the per-superblock `d`/`dmin` f16 scales are decoded inline from the wire bytes (`inference.rs:19242-19243`). A BASALT `nvfp4_wire_row_dot` carrying UE4M3 sub-block scales inside the 36-byte block would fit the same shape; anything needing a per-TENSOR scale is a signature departure to flag.
- Doc block `inference.rs:19223-19234`: mirrors `ggml_vec_dot_q4_K_q8_K_generic` numeric shape exactly; scalar correctness-first, no SIMD; marked `#[allow(dead_code, ...)]` at 19235 (the DG lane note).
- CUDA twins: `src/cuda_resident.rs:428` ("Bit-identical reproduction of the validated CPU oracle q4_k_wire_row_dot"); GPU-vs-CPU-oracle parity tests at `src/cuda_resident/tests.rs:2160-2241` (fills expected with `crate::inference::q4_k_wire_row_dot` on the SAME bytes, asserts worst rel divergence).

### 2.3 Rayon-over-rows pattern
`src/gemma4_runtime.rs:176-199` (`matvec_q`) and `:303-323` (`matvec_q8k`): `const ROW_CHUNK: usize = 64; out.par_chunks_mut(ROW_CHUNK).enumerate().for_each(...)` — fixed chunks, each row's dot serial and landing at a fixed index, "bit-identical to the per-row version (greedy parity safe)" (comment 171-175). Determinism rationale independently documented in `qa/determinism/determinism-baseline-20260614T063455Z.md:28-39` (rayon partitions OUTPUT space; no cross-thread float combine; `--threads 1 == --threads 10` byte-identical).

### 2.4 Runnable-lane single-dispatch principle + the covered set
- Principle: `src/runnable/dequant.rs:1-12` — "Breadth comes from one small dispatch over per-format routines, not a per-format kernel matrix (`RUNNABLE_LANE_SPEC.md`, principle #3)"; f32-only, no Metal/CUDA fast paths; anchored externally against ggml reference fixtures (Gate 2).
- The dispatch: `dequant.rs:23-44` `pub fn dequantize(tensor_type, bytes, n_elements, tensor_name) -> Result<Vec<f32>>` — one match; each covered quant routes to the crate's existing validated block decoder.
- Covered set declared (three axes), authoritative in code at `src/runnable/admit.rs`:
  - Architectures: `COVERED_ARCHITECTURES = ["llama","qwen2","qwen3","qwen35","gemma2","gemma3","phi3"]` — admit.rs:32-34.
  - Tokenizers: `SPM_TOKENIZERS = ["llama","gemma","gemma4"]`, `BPE_TOKENIZERS = ["gpt2"]` — admit.rs:38-39.
  - Quants: `is_covered_quant` — admit.rs:177-190 (F32, F16, Q8_0, Q6K, Q5K, Q4K, Q3K, Q4_0, IQ4XS).
- Smoke guardrail (distinct, narrower): `src/runnable/smoke.rs:1-16` — smoke-admission runs ONLY on oracle-qualified combos; `is_oracle_qualified` smoke.rs:42 (llama/Q8_0/SPM, qwen3/Q8_0/BPE, gemma3/Q8_0/SPM, phi3/Q8_0/SPM per smoke.rs:95-96 + tests 333-338); anything else refused "combo not yet anchored" (smoke.rs:12-13). Pass emits `execution_lane = Runnable` receipt with `parity: not_compared` (smoke.rs:15-16).
- Module-level statement of refusal-as-load-bearing: `src/runnable/mod.rs:1-10` (principle #2 cite at mod.rs:7).

---

## 3. src/gemma4_runtime.rs

### 3.1 Module doc
`gemma4_runtime.rs:1-11`: from-scratch gemma4 runtime; forward math validated bit-for-bit vs llama.cpp in `tests/gemma4_forward.rs` ("The capital of France is" → " Paris..."); incremental KV cache, one token per `step`; weights stay quantized in memory, matmuls dequantize on the fly.

### 3.2 Residency structure (CPU lane)
- No eager decode, no resident copy: `WireQuant { mmap: Arc<GgufWireMmap>, offset, element_count, format }` — `gemma4_runtime.rs:77-82`, doc 70-76 ("mmap pages fault in on first touch... Any tensor type outside WireFormat fails closed at load").
- `WireFormat` enum (the covered wire set for this engine): `Q8_0, Q4_0, Q4_1, Q4K, Q5K, Q6K` — `gemma4_runtime.rs:38-45`; `values_per_block`/`bytes_per_block` at 49-67. **An NVFP4 lane plugs in here: new `WireFormat` variant (64 vals / 36 B) + arm in `WireQuant::new` (87-99) + a row-dot fn in the matvec dispatches.**
- Per-layer holder: `struct LayerWeights` — `gemma4_runtime.rs:530-553` (attn_q/k/v/output, ffn_gate/up/down all `WireQuant`; norms are `Vec<f32>`; optional MoE).
- Top-level: `pub struct Gemma4Runtime` — `gemma4_runtime.rs:749-767` (`layers: Vec<LayerWeights>`, `token_embd: WireQuant`, `per_layer_token_embd: Option<WireQuant>`, `output_norm: Vec<f32>`; **no separate output/lm_head field**).
- GPU lanes: `pub struct Gemma4GpuRuntime` (Metal) `gemma4_runtime.rs:1934` — layer weights as anonymous GPU `WirePages`, embeddings stay file-backed mmap (1938-1945, 2005-2010, `WirePages::read_from_file` 2033-2035); `pub struct Gemma4CudaResident` `gemma4_runtime.rs:2635` — tied Q6_K head on GPU when resident else CPU (2630, 2661).

### 3.3 Matvec dispatch seam
`WireQuant::matvec` `gemma4_runtime.rs:147-164` — routes Q8_0/Q4_0/Q4_1 → `matvec_q` (quantize activation to Q8_0 blocks once), Q4K/Q6K → `matvec_q8k` (Q8_K activations), Q5K = gather-only (162). Row-dot fn tables: `matvec_q` 180-187, `matvec_q_rows` 210-217, `matmul_q` 266-273 (`q8_0_wire_row_dot`/`q4_0_wire_row_dot`/`q4_1_wire_row_dot`), `matvec_q8k` 307-311 and `matmul_q8k` 336-340 (`q6_k_wire_row_dot`/`q4_k_wire_row_dot`). All imported from `crate::inference` at 15-19. Batched `matmul_q`/`matmul_q8k` (one weight read reused across K activations — the spec-decode bandwidth win) at 251-299 / 325-360.

### 3.4 Tied lm_head — YES
- Binding fallback: `Gemma4Binding::bind` `src/model.rs:1216-1219` — `match find_tensor(gguf, "output.weight") { Some(desc) => (desc, false), None => (token_embedding.clone(), true) }` → `output_is_tied_embedding`. (Same pattern for llama: model.rs:772-775; struct fields model.rs:1156-1167.)
- Runtime: logits are ALWAYS `self.token_embd.matvec(hidden, vocab, &last)` — `gemma4_runtime.rs:1671` (comment 1668-1670: "token_embd is vocab-major... the tied logits are a single block-wise Q8 matvec"); batched verify path `logits_rows` via `token_embd.matmul_q8k`/`matmul_q` at 1261-1270. Final logit soft-cap applied when configured (1672-1674).
- QAT rows: tied head is Q6_K; Q8_0-row tied head is Q8_0 (WireFormat doc 34-36; GPU head selection 1953, 1977-1999, 2757).

### 3.5 KV path — f32, NOT f16
- `gemma4_runtime.rs:489-495`: "Camelid's gemma4 KV cache is f32. The reference's DEFAULT cache is f16 (+ flash attention with an f16-rounded Q path), which flips near-tie argmax..." — parity oracles are captured with `-ctk f32 -ctv f32 -fa off` to match (e.g. E4B manifest comparator line, and the M-B1 bundle README:20).
- `pub type Gemma4KvCache = Vec<Vec<Vec<f32>>>` — `gemma4_runtime.rs:776-778`. Cross-layer KV sharing `first_kv_shared` (907, 1110-1160).

---

## 4. Pilot provenance — Gemma 4 E4B GGUF (file currently ABSENT from models/)

- Exact filename: `gemma-4-E4B-it-Q8_0.gguf`.
- sha256 `a2232a649523c36bf530f1dc3614eb8c800645c4227390381c8b05d4d6eee05a`, size 8,192,951,456 bytes — recorded in TWO committed manifests:
  - `qa/evidence-bundles/gemma4-e4b-it-q8-0-20260610T103400Z-head-96a75007b156/manifest.json:8-13` (also: oracle = llama.cpp 5d56eff plain-f32 GEMV `--no-repack -fa off -ctk f32 -ctv f32 -ub 1`, prompt pack `qa/gemma4/prompt_packs/basic_v1.json`, oracle artifact `qa/gemma4/oracle/gemma-4-E4B-it-Q8_0.basic_v1.json`).
  - `qa/evidence-bundles/gemma4-e2b-e4b-context-512-8192-20260706T190805Z-head-0b0e4709188f/manifest.json` models[] entry (same sha + size; bounded-context 512-8192 `passed=true`).
- HF repo: `unsloth/gemma-4-E4B-it-GGUF` — catalog literal `src/api/mod.rs:17567-17579` (`catalog_id: "gemma4_e4b_it_q8_0"`, repo_id + filename + size_bytes 8192951456), mirrored in `frontend/src/lib/supportedModels.js:91` and `docs/gemma4-row-audit-2026-06-09.md:53` ("E4B-it | unsloth/gemma-4-E4B-it-GGUF | 8.19 GB").
- E2B sibling (5,048,350,848 B, sha `0a8488b1...`) in the same context manifest; E2B CUDA bundle records `"source": "unsloth/gemma-4-E2B-it-GGUF"` (`qa/evidence-bundles/gemma4-e2b-q8-cuda-resident-parity-20260711T014910Z-head-15bf42e3/manifest.json:13`).
- Ledger row `gemma4_e4b_it_q8_0` exists (`ledger/camelid-ledger.json`, model_rows; status `supported_exact_row_smoke`; identity has NO sha256 field for this row — sha lives in the bundles). Contract evidence text = `src/api/mod.rs:3414`.
- Support-surface receipts: COMPATIBILITY.md:66 (E4B row), STATUS.md:42, SUPPORT_MATRIX_v0.1.md:36.
- models/ dir at recon time: no gemma-4-E4B file present (only `gemma-4-26B_q4_0-it.gguf` among gemma4 rows).
- QAT de-risk sibling (different file, do not confuse): `gemma-4-E4B_q4_0-it.gguf` (docs/gemma4-cuda-q4_0-plan.md:4; mixed Q4_0/Q4_1/Q6_K per docs/gemma4-cuda-port-plan.md:22).

---

## 5. RECEIPTS.md + DECISIONS.md

### 5.1 RECEIPTS.md (root) — parity-receipt conventions
- Two governing rules: receipt = ONE request, never a support promotion (RECEIPTS.md:12-17); receipts only meaningful for deterministic/greedy runs, sampled runs stamped `reproducible: false` (19-22).
- Schema `camelid.parity-receipt/v1`, defined by `ParityReceipt` in `src/receipt/mod.rs` (RECEIPTS.md:106-109); `receipt_id` = SHA-256 over canonical body (sorted keys, compact, receipt_id excluded; single sealing implementation = `camelid seal-receipt`, RECEIPTS.md:161-169). `lane` block carries `gguf_sha256`, `gguf_filename`, `quantization`, `architecture`, `tokenizer_kind`, `tokenizer_sha256`, versions (111-126).
- Verifier steps + exit codes (RECEIPTS.md:60-87): self-digest, reproducibility gate (exit 2), lane identity, Camelid re-run, reference re-run; divergence record = exit 3; `--self-only` = PARTIALLY VERIFIED.
- Optional execution-trace rollup on the deterministic CPU lane (40-50), ISA-specific.
- Emit paths (89-104): parity harness `--emit-receipt` (`compared_against_reference: true`) vs server opt-in `"camelid_receipt": true` (`compared_against_reference: false`, null match fields).
- **NOTE: the phrase "HARNESS vs CERT classes" appears NOWHERE in RECEIPTS.md or the repo** (grep across *.md: only incidental hits, docs/perf-deep-dive/KQUANT_RECON.md:59 and qa/ornith/G-AGENT-qwen35.md:5). The real distinctions are: (a) harness-emitted vs server-claim receipts above; (b) receipt families by schema id — `camelid.parity-receipt/v1`, `camelid.capability-receipt/v1` (with `oracle_class` letter grades, e.g. qa/capability/macos/receipts/*.json), `camelid.agent_eval/v1`, `camelid.quality-tier/v1` (src/receipt/mod.rs:436), `camelid.raw_decode_parity.v1` (bundle READMEs); (c) runnable smoke receipts (`execution_lane = Runnable`, parity `not_compared`, smoke.rs:15-16).

### 5.2 Evidence-bundle conventions (qa/evidence-bundles/README.md)
- Dir = durable, reviewable manifests + checksums; commit only sanitized content; raw/private staging stays out of git (README.md:57-62); privacy audit script `scripts/audit-evidence-bundle-privacy.mjs` before citing/refreshing (line 62).
- Bundle id scheme (observed across the directory): `<row-or-topic-slug>-<UTC stamp YYYYMMDDTHHMMSSZ>-head-<git-head-hex>/` (e.g. `gemma4-e4b-it-q8-0-20260610T103400Z-head-96a75007b156/`, `llama-3.2-1b-q4_k_m-windows-cuda-resident-parity-20260716T151718Z-head-052c4030/`).
- Contents: `manifest.json` (schema-tagged, e.g. `camelid.gemma4_exact_row_public_evidence.v1` with per-log sha256 entries — E4B manifest lines 2, 19-55) + `SHA256SUMS`; verify all via `bash scripts/check-evidence-bundle-checksums.sh` (README.md:54). STATUS.md is the canonical durable-anchor index; COMPATIBILITY.md anchors section must stay a pointer (enforced by drift check E, scripts/check-ledger-drift.mjs:199-214).

### 5.3 DECISIONS.md
- Heading format: `## D<N> — <Title> (<YYYY-MM-DD>)` (em-dash; D13/D15 use plain `-`). First: `## D1 — Topology: ... (2026-06-13)` at DECISIONS.md:6.
- **Last number: D16 — CONFIRMED** — `## D16 — API engine inversion: one worker thread owns all decode compute (2026-07-09)` at DECISIONS.md:694, and it is the final heading in the file. So BASALT's decision entry = D17. Caveat: the historical numbering has duplicates (two D6 at :102/:172, two D7 at :122/:232, two D11 at :487/:546) — do not renumber, just append D17.

---

## 6. RUNNABLE_LANE_SPEC.md — location

- **The file `RUNNABLE_LANE_SPEC.md` does NOT exist in the repo** (git ls-files + filesystem search: absent). It was the external conductor spec; code and docs cite it by name: `src/runnable/dequant.rs:4` (principle #3, single dispatch), `src/runnable/admit.rs:13` (principle #2, machine-readable refusal), `src/runnable/mod.rs:7`, `BACKEND_ASKS.md:4`.
- The in-repo authority is (a) the code covered-set in `src/runnable/admit.rs` ("The covered-set here is **authoritative for the runnable lane** and is taken verbatim from the spec", admit.rs:15-16) and (b) `RUNNABLE_LANE_RECON.md` (repo root, 152 lines) — the Gate-0 recon that quotes the spec's scope ("generic, f32-only, breadth-first path whose correctness becomes the promotion oracle for the supported lane", RUNNABLE_LANE_RECON.md:6-8) and its v1 sets (§1 "Gap → admission gate": three-axis covered-set, machine-readable refusal naming {axis, offending value, tensor}; §2 quant set).
- Rules relevant to adding a new tensor type: new quant enters via (1) `GgufTensorType` variant + `from_id` + `layout` (reader.rs), (2) `is_covered_quant` (admit.rs:177-190) ONLY once a dequant-to-f32 routine exists ("covered iff the runnable lane has a dequant-to-f32 routine for it", admit.rs:174-175), (3) the one dispatch arm in `dequant.rs:29-44`, (4) external anchoring against reference fixtures (dequant.rs:6-8: "Correctness is anchored externally against ggml reference fixtures (Gate 2), not trusted from the internal paths it reuses"; fixtures at `tests/fixtures/dequant/*.json`, e.g. Q5_K.json; test `tests/runnable_dequant.rs`). Smoke stays refused until the (arch, quant, tokenizer) combo is oracle-qualified (smoke.rs:10-13, 42-56).

---

## 7. Existing eval assets to reuse

### 7.1 Prompt sets
- **Five-prompt gemma4 gate**: `qa/gemma4/prompt_packs/basic_v1.json` + pinned oracles `qa/gemma4/oracle/gemma-4-E4B-it-Q8_0.basic_v1.json` (E4B manifest:16-17; COMPATIBILITY.md:33); bounded-context packs `qa/gemma4/prompt_packs/context_{512,1024,2048,4096,8192}_v1.json` + `deep_v1.json`; chat-template lock `qa/gemma4/template_shapes_v1.json`. In-tree harness: `tests/gemma4_generation_parity.rs` (cited COMPATIBILITY.md:66; CUDA branch per api/mod.rs:3456 evidence text).
- **MUSTER raw packs** (8 prompts): `qa/prompt-packs/llama32-1b-q4km-raw-decode-pack-v1.json`, `qa/prompt-packs/phi3-mini-raw-decode-pack-v1.json` — schema `camelid.raw_decode_prompt_pack/v1`; derivation rule = MUSTER_CONDUCTOR.md Amendment A-3 (MUSTER_CONDUCTOR.md:137): prompt 0 = the harness France BOS probe + the seven `qa/speed/prompts.json` columns verbatim, committed BEFORE oracle capture. Harness: `scripts/raw-decode-parity.mjs` — default depths `--token-counts 1,5,50` (raw-decode-parity.mjs:39); two-phase `--reference-out`/`--reference-in` so engines are never co-resident (:45-50); **`--stop`/`--variant`/`--proof-chain` defaults are Llama-3/K-quant specific — non-Llama rows MUST pass all three** (A-3, MUSTER_CONDUCTOR.md:137).
- **"8 prompts × 128 greedy tokens"**: the 128 figure is the SPEED pack's per-column budget, not the parity depth — `qa/speed/prompts.json` columns: code_completion/structured_json/repetitive_extraction/normal_chat/creative_writing/adversarial_lowaccept all `n_gen: 128`, longctx_splitk `n_gen: 96` (schema `camelid.speed-prompts/v1`). Parity gates run those 8 prompts at depths 1/5/50 (harness default; e.g. M-B1 bundle README:27 "compared via --reference-in at token depths 1/5/50"). MUSTER gotcha (memory + A-3): SPM rows can't consume `qa/speed/prompts.json` directly ({columns:[…]} crashes the harness) — hence the committed array-form packs.
- Other five-prompt packs: `qa/prompt-packs/tinyllama-broader-5prompt.json`, `mistral-broader-50tok-5prompt.json`, `gemma3-chat-gate-pack-v1.json` (5 prompts, 1/5/50 — api/mod.rs:3067 evidence).

### 7.2 Determinism + CUDA-vs-CPU tolerance conventions
- CPU determinism baseline: `qa/determinism/determinism-baseline-20260614T063455Z.md` — reduction sites audit; `--threads 1` vs default byte-identical (:36-39); output-partitioned rayon, no cross-thread float combine (:28-34).
- **Cross-backend (CUDA-vs-CPU-oracle) tolerance policy — the rule**: `MUSTER_CONDUCTOR.md:93`: "A non-identical position is admissible ONLY if probed and attributed under the established cross-backend tolerance discipline (precedent: Ornith Q4_K_M — every flip probed to a ≤0.33-nat soft position where the oracle's own backends also flip...). The bundle must contain the probe artifacts, the nat/logprob gap, and the oracle-side control. 'Looks coherent' is not attribution. Widening a tolerance, shrinking a pack, or swapping a prompt to convert FAIL→PASS is a campaign-failing act."
- Precedent artifacts: `qa/ornith/constrained-vram/RECEIPT_ITEM2_qwen35_parity_cuda.json` (:746 result line), probe tooling `qa/ornith/constrained-vram/probe_divergence_cuda.mjs` (llama.cpp `/completion` with `n_probs: 10`, :33), oracle-side control `compare_oracle_backends.mjs` + `oracle_backend_variance.json`. Worked example table (gaps 0.0135-0.1059 nat, camelid rank #2, CPU-lane controls): M-B1 bundle README:37-47. Weakest-accepted precedent: gemma3 M-A1 DISCLOSED 0.3416-nat near-tie above the 0.33 line (api/mod.rs:3067).
- Kernel-level CUDA tolerances (unit scope, not token gates): `src/cuda_resident/tests.rs:2160-2241` etc. — GEMV output vs `*_wire_row_dot` CPU oracle on SAME bytes, worst-rel asserted; IQ4_XS kernel "numerically matches ... validated to 1e-4" (`src/cuda_resident.rs:900`).

### 7.3 Perplexity corpus/slice (Ornith Item 4)
- In-Rust PPL instrument: `src/quality/mod.rs:1-40` — `Perplexity` accumulator (f64 NLL), `PerplexityConvention` pinned to mirror `llama-perplexity` (window n_ctx, stride n_ctx/2, BOS once at corpus start, second-half scoring; :18-40).
- Corpus (tracked in-repo): `qa/ornith/constrained-vram/heldout_coding.txt` (165 KB repo Rust + llama.cpp C++, disjoint from calibration; sha256 computed this recon = `460634c23b5a6ddeeaa325b4a461c44c569e753f4064a2af20290f36f35aaedf` — NOT recorded in any committed SHA256SUMS, only referenced by name in QUANT_QUALITY_TABLE.md:10 and RECEIPT_ITEM4_residency.json:21). Calibration: `TRACES_agentic_20.txt` (sha256 computed = `0d60d251bc1089caff2a06e76874c7969dadeba3410f96160b8e7bbcba66cf9e`) + `imatrix_ornith_agentic.gguf`, both tracked.
- Method receipt: `qa/ornith/constrained-vram/QUANT_QUALITY_TABLE.md` — `llama-perplexity` on heldout_coding.txt, c=2048, PPL ± err per quant (Q6_K 2.3636 baseline; deltas quoted %).
- Pinned llama-perplexity binary: `<llama.cpp>/build/bin/llama-perplexity.exe` (task-given pin, build 9632 / acd79d603).

### 7.4 KL-divergence / logit-dump tooling
- **No KL tool exists** — confirmed: `MUSTER_RECON.md:96` "no KL probe exists"; `docs/architecture/IQUANT_IQ4XS_PLAN.md:97,132` only PROPOSES a "KL-divergence / perplexity band". Nothing in src/scripts computes KL.
- **Logit-dump surfaces DO exist** (BASALT can build KL on these):
  - Camelid API OpenAI logprobs: chat `logprobs`/`top_logprobs` (`src/api/mod.rs:462-463`, response `ChatLogprobs` :1183-1200) and legacy completions `logprobs: N` (:529, `CompletionLogprobs` parallel arrays :1256-1266).
  - Camelid diagnostics extension: `GenerationDiagnostics.top_logits: Vec<LogitDiagnostic>` + per-step `step_top_logits` (`src/api/mod.rs:1031-1044`); `LogitDiagnostic { token_id, logit, probability, rank, selected, text }` (:1103-1111); full dense forward diagnostics behind `camelid_dense_diagnostics=true`, dumped by `scripts/extract-forward-trace.mjs` (schema `camelid.forward-trace.v1`, includes per-layer states + logits).
  - Oracle side: llama.cpp `/completion` `n_probs` captures (probe_divergence_cuda.mjs:33; committed full capture `oracle-nprobs-50tok.json` in the M-B1 bundle, README:59).
  - Harness precedent combining both: `scripts/chat-parity-tinyllama.mjs:93-107,134` (llama `top_logprobs: 20` vs `backendChat.camelid.top_logits`).

---

## 8. Ledger + drift checker

### 8.1 `ledger/camelid-ledger.json` schema
Top-level: `{ ledger_version: "camelid.ledger/v1", provenance: { source_head, note }, capabilities: {...}, model_rows: [...] }` (28 rows at HEAD).
- `provenance.note`: derived from the static CapabilitiesResponse literal in `src/api/mod.rs` by `scripts/extract-capabilities-to-ledger.mjs`; per CAIRN Amendment 1 the CODE is the source of truth; drift check re-derives and fails CI on disagreement (provenance excluded).
- `capabilities` keys: engine, gguf_metadata, tensor_loading, tokenization, inference, streaming, model_downloads, hf_catalog_install, execution_plan, support_contract, **supported_quantization**, **planned_quantization**, supported_model_families, planned_model_families, api_features, notes.
- Quantization entries: `{ id, status, notes }` — statuses in use: `supported`, `supported_current_gate`, `supported_named_exact_rows_only`, `planned_beyond_named_certified_rows`.
- `model_rows[]`: `{ identity: { id, family, quantization, gguf_filename, [sha256] }, contract: {...} }`; contract fields include id, tool_capable, family, quantization, status (e.g. `supported_exact_row_smoke`), support_scope, full_support_status/blockers, per-check strings (metadata_parses, tokenizer_works, tensors_load, generation_runs, parity_audited, performance_measured), frontend gate strings, bounded_context_*_pack fields per bucket, evidence (long prose), next_step. Row `gemma4_e4b_it_q8_0` present (family `gemma4_ple_matformer_decoder`).
- **How a BASALT "watch item" fits**: the established pattern is a `planned_quantization` entry (e.g. `{"id":"Q4_0/Q5_0","status":"planned_beyond_named_certified_rows","notes":"...engine facts exist, per-row certification remains planned"}`) — an NVFP4 entry would go there (edit the src/api/mod.rs literal, re-run the extractor). Since the ledger is derived, a watch item MUST be added in code first; hand-editing the JSON makes check A fail.

### 8.2 `scripts/check-ledger-drift.mjs` (runs in public-scrub, JOB=6) — what it enforces
- Check A — freshness: re-derives via `buildLedger()` from `extract-capabilities-to-ledger.mjs`, deep-diff vs committed (provenance stripped) → "ledger is STALE vs the code contract" (:81-88).
- Check B — supported-table non-contradiction: README `| Model row | Quant |` table + COMPATIBILITY `| Exact row | Public claim |` table; any support-claim row that maps to a ledger row whose `contract.status` is not `supported*` fails; unmapped rows are LOGGED never failed (:90-140). **Known gotcha (memory, iq4xs lane): MISSING doc rows are not caught — only contradictions.**
- Check C — frontend catalog: every `catalog_id` in `frontend/src/lib/supportedModels.js` must have a contract row (:142-157).
- Check D — sha256 cross-surface agreement: for ledger-anchored `(gguf_filename, identity.sha256)` pairs, any line in README/COMPATIBILITY/STATUS/src/api/mod.rs naming the file AND stating a full sha must state the ledger's sha (:159-197). Only rows WITH identity.sha256 are anchored.
- Check E — anchors single home: COMPATIBILITY.md "Durable evidence anchors" must not re-list qa/evidence-bundles paths; index lives in STATUS.md (:199-214).

---

## 9. CLI / env surface for a refusal receipt + pilot runs

- **Refusal-side (NVFP4 file today, pre-enum)**:
  - `camelid inspect <path.gguf>` (`src/main.rs:538-539`, arm :1409-1412) → `read_metadata` fails: `UnsupportedGguf("tensor <name> has unknown or removed GGML type Unknown(40)")` (reader.rs:530-535). Cheapest deterministic refusal artifact.
  - `camelid runnable-smoke <path.gguf>` (`src/main.rs:540-544`, arm :1444-1467) → prints `smoke-admission REFUSED/FAILED: <err>` to stderr, exits 1; on covered-but-unanchored combos the machine-readable "combo not yet anchored" (smoke.rs:95-96).
  - Server-side: `camelid serve` then `GET /api/models/local` → per-file `admitted:false` + `admission_reason` ("GGUF parse failed: ..." for unknown-type files, mod.rs:17828); `POST /api/models/runnable-smoke {filename}` (mod.rs:17904-17939).
- **Pilot gemma4 runs (post-enum)**:
  - Direct CLI, no server: `camelid gemma4-generate <path> --prompt "The capital of France is" --max-tokens 24` (`src/main.rs:598-605`, arm :1536+, loads `Gemma4Runtime::load`). CUDA variant `gemma4-cuda-generate` (:608), Metal `gemma4-generate-gpu` (:616).
  - Serve lane: `CAMELID_GEMMA4_SERVE=1 camelid serve --model <path>` — gate fn `gemma4_serve_enabled` `src/api/mod.rs:4445-4452` (accepts 1/true/yes); routing check `mod.rs:6762`. CUDA opt-in `CAMELID_GEMMA4_CUDA=1` (+ `--features cuda`) `mod.rs:4454-4463`.
  - Load endpoint fit preflight bypass: `CAMELID_SKIP_FIT_CHECK=1` (EXACTLY trimmed "1") — `fit_check_skipped` `src/api/mod.rs:4220-4222`, guard `fit_preload_guard` :4224-4234; 422 message text :4195-4205. Relevant here because the 8.2 GB E4B on this 16 GB box previously tripped the fit check (demo-video memory).
  - Tests env pin: `CAMELID_GEMMA4_GGUF=<path>` drives `tests/gemma4_forward.rs:9`, `tests/gemma4_load.rs:9`, `tests/gemma4_spec_decode_parity.rs:12`.
  - Model path semantics: `--model` absolute, or relative resolved against `CAMELID_MODELS_DIR` / `<exe dir>/models` / `./models` (`src/main.rs:326-334`).
- **Memory-safety rules that bind any BASALT run on this box** (bench-memory-safety memory): free-RAM check (models + 3 GB) before any load; single-engine capture/compare (raw-decode-parity two-phase `--reference-out`/`--reference-in` exists exactly for this); kill orphans by PID, never blanket-taskkill.

---

## Surprises / deviations worth the conductor's attention

1. **NVFP4 files fail at PARSE, not at admission** — `tensor_nbytes` (reader.rs:530-535) rejects `Unknown(40)` before `runnable::admit` runs, so today's refusal reason is `UnsupportedGguf: unknown or removed GGML type`, surfaced via /api/models/local as "GGUF parse failed: ...". Adding the enum variant + layout MOVES the refusal point from parse to the admission/dequant gates — the refusal receipt text changes class. Capture the baseline refusal BEFORE touching the enum.
2. **RUNNABLE_LANE_SPEC.md is not in the repo** — code cites it, but the only in-repo articulations are admit.rs/dequant.rs module docs and RUNNABLE_LANE_RECON.md.
3. **"HARNESS vs CERT classes" does not exist as a convention** — closest real distinctions documented in §5.1.
4. **`quantization_label` ftype map (receipt/mod.rs:235-256) has no entry for NVFP4's ftype 26** — an NVFP4 GGUF whose `general.file_type=26` falls back to the dominant-tensor Debug name; pick the enum variant Debug name deliberately (it becomes the receipt-visible label) and consider adding ftype 26 to the map in the same change.
5. **DECISIONS.md has duplicate numbers** (two D6/D7/D11) — append D17, never renumber.
6. **The ledger is derived** — a watch item goes into the `src/api/mod.rs` capabilities literal (planned_quantization is the established slot) + re-run `scripts/extract-capabilities-to-ledger.mjs`, else drift check A fails CI.
7. **E4B GGUF absent from models/** — must be re-downloaded from `unsloth/gemma-4-E4B-it-GGUF` and hash-verified against `a2232a64...` before any pilot leg (WRAITH deleted target/debug, but the 8B WRAITH pair ~17 GB still occupies models/ — check free disk + the free-RAM rule).
8. **gemma4 parity oracles are pinned to llama.cpp `5d56eff` plain-f32 GEMV flags** (`--no-repack -fa off -ctk f32 -ctv f32 -ub 1`), NOT the repo-wide acd79d603 pin used by the K-quant/MUSTER lanes — BASALT must choose and state its comparator pin explicitly.
