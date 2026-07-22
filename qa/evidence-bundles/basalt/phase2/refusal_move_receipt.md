# BASALT Phase 2 — Refusal-point-move receipt (parse → admission)

Campaign: BASALT. Phase 2 (GGUF load + runnable-lane admission surface). Date: 2026-07-16.
Branch: `basalt/phase2-load-interop` off `main` `939710de` (Phase 1 NVFP4 format core merged).
Host: Windows 11 Home 10.0.26220 dev laptop (RTX 3060 Laptop 6 GB, 16 GB RAM).

Test artifact (unchanged from Phase 0, hash re-verified this session):

| File | SHA256 | Size (bytes) |
|---|---|---|
| `<camelid>/models/qwen3-0.6b-NVFP4-basalt-refusal.gguf` | `7337b616141b2436f839b353fb40dc2f77023989316ea7d83624f4f45e2a9146` | 341,454,496 |

Provenance: pin `acd79d603` (build 9632) `llama-quantize --tensor-type '.*=nvfp4'` from
Qwen3-0.6B Q8_0; 197 tensors nvfp4, 113 f32; deterministic (byte-identical re-run);
pin-validated by `llama-completion` greedy generation. Full provenance:
`qa/evidence-bundles/basalt/phase0/refusal_receipt.md` §ii–v.

## 1. BEFORE — the committed Phase 0 baseline (parse-time refusal)

Cited verbatim from `qa/evidence-bundles/basalt/phase0/refusal_receipt.md` §vi, captured
against `main` `4f9603f0` (pre-enum, `GgufTensorType::Unknown(40)`), both **exit code 1**:

```
$ <camelid>/target/release/camelid.exe inspect <camelid>/models/qwen3-0.6b-NVFP4-basalt-refusal.gguf
Error: unsupported GGUF feature: tensor token_embd.weight has unknown or removed GGML type Unknown(40)
```

```
$ <camelid>/target/release/camelid.exe runnable-smoke <camelid>/models/qwen3-0.6b-NVFP4-basalt-refusal.gguf
smoke-admission REFUSED/FAILED: unsupported GGUF feature: tensor token_embd.weight has unknown or removed GGML type Unknown(40)
```

Refusal point: **GGUF parse** — `tensor_nbytes` (`src/gguf/reader.rs`) has no layout for
`Unknown(40)`, so the file never reaches the admission gate. The refusal names a GGML
type hole, not a policy.

## 2. AFTER — Phase 2 build (admission-scope refusal)

Build: `cargo build --release` on `basalt/phase2-load-interop` (this bundle's commit;
exe `<cam-basalt>/target/release/camelid.exe`). Captures below are verbatim.

### 2a. `camelid inspect` — now parses. **Exit code 0.**

```
$ <cam-basalt>/target/release/camelid.exe inspect <camelid>/models/qwen3-0.6b-NVFP4-basalt-refusal.gguf
```

stdout is the full 7,374,850-byte pretty-printed `GgufFile` JSON (310 tensor
descriptors); stderr empty. Head, verbatim:

```json
{
  "path": "<camelid>/models/qwen3-0.6b-NVFP4-basalt-refusal.gguf",
  "version": 3,
  "tensor_count": 310,
  "metadata_count": 28,
  "alignment": 32,
  "data_start_offset": 5951136,
  "metadata": {
```

The tensor that named the BEFORE parse error now serializes as a first-class NVFP4
descriptor (verbatim from the same dump):

```json
    {
      "name": "token_embd.weight",
      "dimensions": [
        1024,
        151936
      ],
      "tensor_type": "NVFP4",
      "relative_offset": 4096,
```

Tensor-type census derived from the captured dump (`summarize_inspect.mjs` over the
verbatim stdout):

```json
{
  "version": 3,
  "tensor_count": 310,
  "architecture": "qwen3",
  "file_type": 39,
  "tensor_type_counts": {
    "F32": 113,
    "NVFP4": 197
  },
  "first_tensor": {
    "name": "output_norm.weight",
    "tensor_type": "F32"
  }
}
```

**197 NVFP4 / 113 F32** — identical to the pin quantizer's per-tensor log
(`phase0/quantize_nvfp4.txt`) and the pin loader's own census (`type nvfp4: 197
tensors`). `general.file_type = 39` (`LLAMA_FTYPE_MOSTLY_NVFP4`) round-trips, and the
receipt ftype map now labels it `NVFP4`.

### 2b. `camelid runnable-smoke` — refuses at ADMISSION. **Exit code 1.**

Verbatim, complete stderr (stdout empty):

```
$ <cam-basalt>/target/release/camelid.exe runnable-smoke <camelid>/models/qwen3-0.6b-NVFP4-basalt-refusal.gguf
smoke-admission REFUSED/FAILED: unsupported GGUF feature: unsupported quant NVFP4 in tensor token_embd.weight for architecture "qwen3": NVFP4 is pilot-scoped to gemma4 until Gate G3 (BASALT D-B3)
```

This is the D-B3 pilot-scope reject (`AdmissionReject { axis: quant, offending_value:
"NVFP4", tensor: token_embd.weight }` surfaced through `BackendError::UnsupportedGguf`)
— a policy refusal naming its decision record, not a parser hole.

## 3. What moved, exactly

| | BEFORE (main 4f9603f0) | AFTER (Phase 2) |
|---|---|---|
| `inspect` | exit 1, `UnsupportedGguf ... Unknown(40)` at parse | exit 0 — metadata parses; 197 tensors report type `NVFP4` |
| `runnable-smoke` | exit 1, same parse error | exit 1, **admission** reject: quant axis, `NVFP4 is pilot-scoped to gemma4 until Gate G3 (BASALT D-B3)` |
| refusal class | GGML-type hole (accidental shape) | machine-readable policy reject (`AdmissionReject { axis: quant, offending_value: "NVFP4", tensor, message }`) |
| NaN-sentinel check (D17/T5) | unreachable (parse fails first) | decode-time, in `decode_nvfp4_tensor` via `runnable::dequant` (admission is metadata-only and cannot scan wire bytes) |

The refusal REMAINS fail-closed at every point past admission: dequant of any
non-covered type refuses, and smoke stays gated on oracle-qualified combos
(gemma4+NVFP4 refuses as `combo not yet anchored` until Gate G3).
