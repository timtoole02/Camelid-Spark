# BASALT Phase 2 — Pilot NVFP4 row inspect receipt (metadata-only)

Date: 2026-07-16. Branch `basalt/phase2-load-interop`. Exe: `<cam-basalt>/target/release/camelid.exe`
(Phase 2 build; same binary as `refusal_move_receipt.md` §2).

Scope discipline: **metadata/inspect only** — no model loads, no generation (free-RAM rule;
the pilot rows are 4.6–8.2 GB on a 16 GB box). `camelid inspect` reads the GGUF header only.

## 1. gemma-4-E4B-it-NVFP4-mm.gguf (the produced pilot row)

Produced this session by the concurrent quantize agent (pin `acd79d603` `llama-quantize`
per D-B5); file observed stable (mtime 16:05:35, unchanged through 17:08+; no
llama-quantize process running at capture time).

| File | SHA256 | Size (bytes) |
|---|---|---|
| `<camelid>/models/gemma-4-E4B-it-NVFP4-mm.gguf` | `eb293344972e2b292a043b8e7649b9788dca915b034e5c2721cfc531cf9863d9` | 6,058,607,776 |

### 1a. `camelid inspect` — parses clean. **Exit code 0.**

Tensor-type census derived from the verbatim stdout dump (720 descriptors):

```json
{
  "version": 3,
  "tensor_count": 720,
  "architecture": "gemma4",
  "file_type": 39,
  "tensor_type_counts": {
    "F32": 339,
    "BF16": 1,
    "Q8_0": 86,
    "NVFP4": 294
  },
  "first_tensor": {
    "name": "output_norm.weight",
    "tensor_type": "F32"
  }
}
```

Matches the expected produced-row shape exactly: **294 NVFP4** matmul weights,
**86 Q8_0** (the 2 kept embeddings `token_embd` + `per_layer_token_embd`, plus the 84
2-D F32 `inp_gate`/`proj` tensors the quantizer converts to Q8_0), **339 F32**
(423 − 84), **1 BF16** (`per_layer_model_proj.weight`). `general.file_type = 39`.

Sidecar assertion (D-B2, recon §10 Phase 2): scan of all 720 tensor names for
`.scale` / `.input_scale` suffixes → **zero sidecar tensors**. The produced pilot row
is sidecar-free, as `llama-quantize` output must be.

### 1b. `camelid runnable-smoke` — refused. **Exit code 1.** Verbatim stderr:

```
$ <cam-basalt>/target/release/camelid.exe runnable-smoke <camelid>/models/gemma-4-E4B-it-NVFP4-mm.gguf
smoke-admission REFUSED/FAILED: unsupported GGUF feature: unsupported quant BF16 in tensor per_layer_model_proj.weight; runnable v1 covers F32, F16, Q8_0, Q4_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS (NVFP4: gemma4 pilot only until Gate G3)
```

**Honest outcome, recorded as-is:** the real E4B pilot carries one BF16 tensor
(`per_layer_model_proj.weight`, 55.1 MB), and BF16 has never been in the runnable
covered set — so runnable admission of the full pilot FILE refuses on BF16 grounds
before the NVFP4 pilot scope or the smoke anchoring gate are ever consulted. This is
not a failure of the Phase 2 wiring: the pilot's execution home is the gemma4
supported lane (`gemma4_runtime`, Phase 3), not the generic runnable runtime. The
D-B3 admission semantics themselves are proven by (a) the qwen3 live capture above
(NVFP4-specific pilot-scope reject) and (b) the synthetic-fixture unit tests
(`admits_gemma4_nvfp4_pilot`, `rejects_nvfp4_outside_pilot_arch`,
`gemma4_nvfp4_smoke_refusal_is_not_yet_anchored` — the last asserting that a
BF16-free gemma4+NVFP4 combo refuses at the SMOKE gate as
`combo not yet anchored: gemma4/NVFP4/Spm`, not at admission).

## 2. Other produced protocol rows

At capture time the models dir contained no other produced BASALT rows (no `Q4K-mm`,
`Q4_K_M-df`, `Q4_K_M-im` yet — the quantize agent had produced only the `NVFP4-mm` row,
plus the re-downloaded sources `gemma-4-E4B-it-Q8_0.gguf` (8,192,951,456 B, mtime 15:35)
and `gemma-4-E4B-it-BF16.gguf` (15,053,095,840 B, mtime 16:02)). Their receipts belong to
the quantize leg of the G2 assembly, not this bundle.

## 3. Wild interop candidate (optional Phase 2 leg, BASALT_RECON.md §8/§10)

`wild-FreedomAISVR-gemma-4-E4B-it-NVFP4.gguf` — HF `FreedomAISVR/Gemma-4-E4B-it-NVFP4-GGUF`,
**unknown provenance, load-or-refuse-cleanly test only, zero claims**. Download completed
and size-stabilized during this session (two stats 60 s apart, unchanged; final mtime
17:58:55):

| File | SHA256 | Size (bytes) |
|---|---|---|
| `<camelid>/models/wild-FreedomAISVR-gemma-4-E4B-it-NVFP4.gguf` | `ea8cac5b184e19c09583fa2df691e15db1e0fb990be8d8a7ea8123ea50dad8cf` | 5,185,929,952 |

### 3a. `camelid inspect` — parses clean. **Exit code 0.** Census from the verbatim dump:

```json
{
  "version": 3,
  "tensor_count": 720,
  "architecture": "gemma4",
  "file_type": 39,
  "tensor_type_counts": {
    "F32": 339,
    "BF16": 1,
    "Q6K": 2,
    "NVFP4": 378
  },
  "first_tensor": {
    "name": "output_norm.weight",
    "tensor_type": "F32"
  }
}
```

Differences vs our produced row, stated without claims: **378 NVFP4** (this uploader
also NVFP4-quantized the 84 `inp_gate`/`proj` tensors), **2 Q6_K embeddings**
(`token_embd` + `per_layer_token_embd` — explains the smaller 5.19 GB file; our row
keeps them Q8_0), same single BF16 `per_layer_model_proj.weight`, same 339 F32.

Sidecar scan: **zero `.scale`/`.input_scale` tensors** — a `llama-quantize`-convention
file, so it does NOT exercise the D-B2 sidecar refusal (that path remains covered by
the synthetic-descriptor unit tests; no real-world sidecar-bearing fixture exists yet).

### 3b. `camelid runnable-smoke` — refused. **Exit code 1.** Verbatim stderr:

```
$ <cam-basalt>/target/release/camelid.exe runnable-smoke <camelid>/models/wild-FreedomAISVR-gemma-4-E4B-it-NVFP4.gguf
smoke-admission REFUSED/FAILED: unsupported GGUF feature: unsupported quant BF16 in tensor per_layer_model_proj.weight; runnable v1 covers F32, F16, Q8_0, Q4_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS (NVFP4: gemma4 pilot only until Gate G3)
```

Same BF16 refusal class as the produced pilot row (§1b) — deterministic, clean, no
crash, no partial load. Interop verdict for this leg: the wild file **parses and is
classified identically to our own row's shape** at the metadata level; nothing beyond
parse-or-refuse is claimed.
