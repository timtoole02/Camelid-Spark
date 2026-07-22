# BASALT Phase 0 — Baseline NVFP4 Refusal Receipt

## (i) Date, host, Camelid provenance

- **Date:** 2026-07-16
- **Host:** Windows 11 Home 10.0.26220, dev laptop (RTX 3060 Laptop 6 GB, 16 GB RAM; see `hw_probe.json` in this bundle)
- **Camelid git SHA:** `4f9603f07f12ffe10557ecfa66857dca7e7678ab` (branch `main`, "Merge pull request #465 from timtoole02/muster/ma2-phi3-mini", committed 2026-07-16T12:57:09-07:00)
- **Exe provenance:** `<camelid>/target/release/camelid.exe`, 18,310,656 bytes, mtime **2026-07-16 13:42:38** — rebuilt via `cargo build --release` (incremental) in `<camelid>` AFTER the HEAD commit (12:57), so the binary reflects HEAD `4f9603f0`. Not rebuilt for this receipt (mtime story holds).

## (ii) Pin identity

- **llama.cpp pin:** `acd79d603` (build **9632**; quantize log prints `build = 1 (acd79d6)`, MSVC 19.44.35228.0; the interactive banner prints `b9632-acd79d603`)
- **Binary dirs used:**
  - Quantize (and determinism re-run): `<llama.cpp>/build/bin/llama-quantize.exe` (CUDA build of the pin)
  - Sanity generation: `<pin-tools>/llama-completion.exe` (legacy CPU-only build of the same pinned source), sha256 `9547c4559eed03627856587a7e7158628502923d80e5f0d445b62fbccf951ab3`

## (iii) Quantize provenance

**Exact command** (recovered verbatim from the producing agent's transcript; git-bash):

```
<llama.cpp>/build/bin/llama-quantize.exe --allow-requantize --tensor-type '.*=nvfp4' --override-kv general.file_type=int:39 <camelid>/models/Qwen3-0.6B-Q8_0.gguf <camelid>/models/qwen3-0.6b-NVFP4-basalt-refusal.gguf Q8_0
```

| File | SHA256 | Size (bytes) |
|---|---|---|
| Source `models/Qwen3-0.6B-Q8_0.gguf` | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` | 639,446,688 |
| Output `models/qwen3-0.6b-NVFP4-basalt-refusal.gguf` | `7337b616141b2436f839b353fb40dc2f77023989316ea7d83624f4f45e2a9146` | 341,454,496 |

**Determinism re-check (2026-07-16):** the identical command re-run to a scratchpad temp path produced a file of identical size (341,454,496) and **identical SHA256** (`7337b616…9146`) — **byte-identical / deterministic**. Temp file deleted after hashing.

**Per-tensor decisions** (full log preserved in this bundle as `quantize_nvfp4.txt`, 549 lines):

- 310 tensors total: **197 q8_0 → nvfp4** (all ≥2D `*.weight` matmul tensors, incl. token_embd), **113 kept f32** (all `*_norm.weight` and other 1D tensors — excluded by `tensor_allows_quantization`, see §iv).
- `token_embd.weight` — **q8_0 → nvfp4** (manual override), `[1024, 151936]`, 157.65 MiB → 83.46 MiB:
  ```
  llama_tensor_get_type: token_embd.weight                    - applying manual override: q8_0 -> nvfp4
  [   2/ 310] token_embd.weight                    - [  1024, 151936,      1,      1], type =   q8_0, converting to nvfp4 .. size =   157.65 MiB ->    83.46 MiB
  ```
- `output.weight` — **ABSENT** (Qwen3-0.6B has tied embeddings; no separate LM head tensor). `output_norm.weight` `[1024,1,1,1]` kept **f32**.
- **Zero shape fallbacks**: no "not divisible" / "falling back" warnings anywhere in the log (every quantized tensor has ncols ∈ {1024, 2048, 3072}, all divisible by 64).
- Totals: `model size = 604.15 MiB (8.50 BPW)` → `quant size = 319.96 MiB (4.50 BPW)`.
- Log head also shows the load-time KV override taking effect: `validate_override: Using metadata override (  int) 'general.file_type' = 39`, and the header line `llama_quantize: quantizing ... as Q8_0` (the *positional* ftype was Q8_0 — see §iv).

## (iv) llama-quantize discrepancy resolution (source vs. binary)

**Earlier recon claim:** the pin "CANNOT quantize to NVFP4" because `llama_ftype_get_default_type` (`src/llama-quant.cpp:792-833`) has no `LLAMA_FTYPE_MOSTLY_NVFP4` case → `throw "invalid output file type"` (`:866-868`), and `grep tools/ for nvfp4` = zero hits.

**Resolution: the claim's premises are all TRUE, but the conclusion is FALSE.** The empirical quantize never presented NVFP4 as the *file type*; it rode the **per-tensor override path**, which resolves type names against ggml's type-trait table, not against the CLI's `QUANT_OPTIONS` or `llama_ftype_get_default_type`:

1. **CLI parse:** positional ftype arg was `Q8_0` → `QUANT_OPTIONS` (`tools/quantize/quantize.cpp:34`, entry `:68`) → `LLAMA_FTYPE_MOSTLY_Q8_0`. `--tensor-type '.*=nvfp4'` is parsed by `parse_tensor_type` (`tools/quantize/quantize.cpp:313-343`; pattern is lowercased at `:331`) → `parse_ggml_type` (`:301-311`), which loops over **all** `GGML_TYPE_COUNT` ggml types and matches `ggml_type_name(type)` case-insensitively (`striequals`). `GGML_TYPE_NVFP4` (= **40**, `ggml/include/ggml.h:430`) is registered in the trait table with `type_name = "nvfp4"` (`ggml/src/ggml.c:744-751`), so the string resolves — no `QUANT_OPTIONS` / ftype entry needed. (The zero-grep-hits observation stands: `tools/` never names nvfp4; it inherits it from ggml.)
2. **No "invalid output file type" throw:** `llama_model_quantize_impl` (`src/llama-quant.cpp:857`) calls `llama_ftype_get_default_type(Q8_0)` → `GGML_TYPE_Q8_0` (`:798`) at `:866`, so the `:867-868` throw is never reached. The earlier claim is *correct* that passing NVFP4 as the ftype would fail — `:832 default: return GGML_TYPE_COUNT` → throw. That path was simply bypassed.
3. **Per-tensor selection:** `llama_tensor_get_type` (`src/llama-quant.cpp:661-703`): for each tensor passing `tensor_allows_quantization` (`:288-318`: ≥2D, name ends `weight`, not `*_norm.weight`, not `ffn_gate_inp`/altup/laurel/pos-embd/token-types — these keep rules produced the 113 kept-f32 tensors), the user regex patterns are checked **first** (`:678-691`); a match logs `applying manual override: q8_0 -> nvfp4` (`:683-684`) and **skips** the standard mixture logic `llama_tensor_get_type_impl` (`:694-696`) entirely. `llama_ftype_get_default_type` plays no role in per-tensor NVFP4 selection.
4. **K%64 fallback truth:** `QK_NVFP4 = 64` (`ggml/src/ggml-common.h:211`; block = 4 UE4M3 sub-scales + 32 packed E2M1 bytes, `:213-215`). `tensor_type_fallback` (`src/llama-quant.cpp:362-408`) engages only when `ncols % 64 != 0`; its switch (`:373-393`) has **no NVFP4 case**, so such a tensor hits `default:` → `throw "no tensor type fallback is defined for type nvfp4"` (`:390-392`). **The "throws on K%64≠0" part of the earlier claim STANDS** — but it is a *fallback* throw per incompatible tensor, not a blanket inability, and this model triggered it zero times (all ncols divisible by 64).
5. **Output file_type = 39:** the quantizer writes `general.file_type = ftype` (Q8_0 = 7) at `src/llama-quant.cpp:936`, then applies `--override-kv` entries to the output context **after** (`:943-949`), so `general.file_type=int:39` (= `LLAMA_FTYPE_MOSTLY_NVFP4`, `include/llama.h:156`) wins. The loader independently derives NVFP4-ness from tensor types anyway (`src/llama-model-loader.cpp:763` maps `GGML_TYPE_NVFP4 → LLAMA_FTYPE_MOSTLY_NVFP4`; `:46` names it "NVFP4").

**Verdict:** WRONG — "the pin cannot quantize to NVFP4" (it can, via `--tensor-type` regex overrides + a valid dummy positional ftype + `--override-kv general.file_type=int:39`). STANDS — no NVFP4 ftype case in `llama_ftype_get_default_type` (`:792-833`), zero nvfp4 hits in `tools/`, and the K%64≠0 fallback throw.

## (v) Pin sanity generation

**Tool history (honest record):** the first sanity attempt (2026-07-16 ~14:43) used `llama-cli.exe`, which in pin build 9632 is a **conversation-only interactive REPL** — it printed `--no-conversation is not supported by llama-cli / please use llama-completion instead`, answered the prompt in chat mode ("[Start thinking] Okay, the user is asking…"), then spun on EOF emitting empty `> ` prompts; an orphaned instance of it subsequently hard-hung the machine. See `incident-20260716-hard-hang.md` and `pin_sanity_excerpt.txt` in this bundle. **Do not use llama-cli from this pin in scripts.** The receipt's sanity run below uses `llama-completion.exe` (sha256 in §ii), wrapped in `timeout -k 10 300`, with a free-RAM check (5.83 GB free ≥ 0.34 GB model + 3 GB) and a post-run orphan sweep (none found).

**Command** (git-bash):

```
timeout -k 10 300 <pin-tools>/llama-completion.exe -m <camelid>/models/qwen3-0.6b-NVFP4-basalt-refusal.gguf -p "The capital of France is" -n 8 --temp 0 -no-cnv --no-warmup
```

**Exit code 0.** Generated continuation (greedy, deterministic):

```
The capital of France is Paris. The capital of France is also
```

Perf: prompt 10.15 tok/s (5 tokens), eval 7.89 tok/s (7 runs), CPU AVX2/AVX512 path.

**Load-time type report** (default verbosity suppresses the loader dump in this build; captured via an identical re-run with `-v`):

```
llama_model_loader: - kv  27:                          general.file_type u32              = 39
llama_model_loader: - type  f32:  113 tensors
llama_model_loader: - type nvfp4:  197 tensors
print_info: file type   = NVFP4
print_info: arch                  = qwen3
```

The pin loads, reports, and coherently generates from the NVFP4 artifact: **the file is a valid, runnable NVFP4 GGUF by the pin's own standard.**

## (vi) Camelid refusal captures (current main = the "before" photo)

Both commands run against `main` HEAD `4f9603f0` release exe (§i). No env vars set. Refusals occur at GGUF **parse** (metadata/tensor-table read), not at admission policy: `tensor_nbytes` at `src/gguf/reader.rs:530-535` returns `BackendError::UnsupportedGguf` when `GgufTensorType::layout()` has no layout for `Unknown(40)`. `token_embd.weight` is the first NVFP4 tensor encountered, hence the tensor named in the error.

**a. inspect — exit code 1:**

```
$ <camelid>/target/release/camelid.exe inspect <camelid>/models/qwen3-0.6b-NVFP4-basalt-refusal.gguf
Error: unsupported GGUF feature: tensor token_embd.weight has unknown or removed GGML type Unknown(40)
```

**b. runnable-smoke — exit code 1:**

```
$ <camelid>/target/release/camelid.exe runnable-smoke <camelid>/models/qwen3-0.6b-NVFP4-basalt-refusal.gguf
smoke-admission REFUSED/FAILED: unsupported GGUF feature: tensor token_embd.weight has unknown or removed GGML type Unknown(40)
```

## (vii) Conclusion

**Current main fails closed at GGUF parse on NVFP4 (type id 40):** both `inspect` and `runnable-smoke` exit 1 with `UnsupportedGguf "tensor token_embd.weight has unknown or removed GGML type Unknown(40)"` from `src/gguf/reader.rs:530-535`, against a pin-validated, deterministically reproducible NVFP4 artifact that the pinned llama.cpp loads and generates from correctly.

**Artifact retention:** `<camelid>/models/qwen3-0.6b-NVFP4-basalt-refusal.gguf` (sha256 `7337b616…9146`) stays in place for **Phase 1 golden-vector reuse**.
