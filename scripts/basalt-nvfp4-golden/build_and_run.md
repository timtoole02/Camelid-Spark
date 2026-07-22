# BASALT Phase 1 — NVFP4 golden-vector fixtures: build & run log

Date: 2026-07-16. Pin: `<llama.cpp>` @ git `acd79d603cb2e1c84c0886137b80f1ad649b6857` (READ-ONLY, untouched).
Route: **linked-libs** — every fixture value produced by the pin's own compiled C code
(`quantize_row_nvfp4_ref` / `dequantize_row_nvfp4` exported from the pin-built
`ggml-base.dll`, import lib `<llama.cpp>/build/ggml/src/ggml-base.lib`).
The UE4M3 helpers are the pin's static-inline `ggml/src/ggml-impl.h` code compiled into
the harness verbatim; `ue4m3_table.json` values are additionally derived THROUGH the DLL
(element code 1 => output = 1.0f*d) and asserted bit-identical to the inline path.

## Files

- `nvfp4_fixture_gen.c` — the C harness (final, self-contained; header comment has full provenance)
- `build.bat` — MSVC build script
- `extract_real_blocks.mjs` — GGUF block sampler + real_blocks.json assembler (JS only slices bytes/formats JSON; all numeric truth from the pin exe)
- `crosscheck.mjs` — internal-consistency verifier
- `out1/` — the five fixture JSONs
- `fixtures_sha256.txt` — sha256 of each fixture

## Exact commands used

```bat
:: build (from this directory; vcvars64 = VS2022 BuildTools)
cmd /c build.bat
::   effective compile line:
::   cl /nologo /O2 /std:c11 /MD /DGGML_SHARED ^
::      /I <llama.cpp>\ggml\include ^
::      /I <llama.cpp>\ggml\src ^
::      nvfp4_fixture_gen.c ^
::      /link /LIBPATH:<llama.cpp>\build\ggml\src ggml-base.lib
::   compiler: Microsoft C/C++ Version 19.44.35228 for x64 (_MSC_FULL_VER=194435228)
::   runtime dep: ggml-base.dll copied from <llama.cpp>\build\bin (unmodified)
```

```sh
# generate fixtures 1,2,3,5 (twice, for the reproducibility byte-compare)
./nvfp4_fixture_gen.exe gen out1
./nvfp4_fixture_gen.exe gen out2
cmp out1/ue4m3_table.json  out2/ue4m3_table.json   # identical
cmp out1/decode_table.json out2/decode_table.json  # identical
cmp out1/random_blocks.json out2/random_blocks.json # identical
cmp out1/encode_vectors.json out2/encode_vectors.json # identical
# (out2 deleted after the compare passed)

# fixture 4: real blocks from the GGUF (sha256 verified in-script:
#   7337b616141b2436f839b353fb40dc2f77023989316ea7d83624f4f45e2a9146)
node extract_real_blocks.mjs extract          # -> real_blocks.bin (2048*36 B) + real_blocks_meta.json
./nvfp4_fixture_gen.exe dequant real_blocks.bin 2048 real_blocks_expected.txt
./nvfp4_fixture_gen.exe dequant real_blocks.bin 2048 real_blocks_expected2.txt  # repro: byte-identical
node extract_real_blocks.mjs assemble         # -> out1/real_blocks.json

# cross-check (all PASS)
node crosscheck.mjs out1
```

## Cross-check results

- decode_table: 4096/4096 entries bit-equal to `fround(kvalues[code] * ue4m3_table[scale])`
- nibble probes: 4/4 reconstruct under the packing rule (sub-block s owns qs[s*8..s*8+7];
  LOW nibble of qs[s*8+j] = element s*16+j, HIGH nibble = element s*16+8+j)
- random_blocks: 10031/10031 expected outputs reconstruct bit-exact from wire via decode table
- encode_vectors: 27/27; real_blocks: 2048/2048
- rt-* encode vectors: 6/6 exact bit round-trips
- run-twice reproducibility: byte-identical (gen fixtures and dequant mode)

## Golden behaviors recorded (not judged)

- All-NaN input row: amax comparisons are false -> scale byte 0x00, all element codes 0 -> wire all zeros, dequant all +0.0.
- All ±Inf rows: amax=Inf -> scale saturates to 0x7e (d=224), but every |kvalue*d - Inf| = Inf so first-wins picks index 0 -> elements decode to +0.0.
- Nearest-LUT ties (exact midpoints) resolve to the LOWER index (first-wins), i.e. the smaller magnitude, for both signs.
- amax/6 exactly 448 and above both produce scale byte 0x7e (=448 raw, d=224); max representable element value is 12*224 = 2688.
- Scale bytes 0x00 and 0x7f both decode to d=+0.0 (0x7f is the UE4M3 NaN slot, masked to zero by the pin).
- Negative kvalues times d=0.0 produce -0.0 outputs in the decode table (sign preserved through float multiply).

## Committed-copy note (2026-07-16)

The fixtures under `tests/fixtures/dequant/nvfp4_*.json` are the generator's outputs with
host paths in **provenance string fields only** replaced by placeholders (`<llama.cpp>`,
`<camelid>`, `<scratchpad>`, `<home>`) for the public repo. No table/block/expected-value
data was altered. `fixtures_sha256.txt` records the hashes of the committed (sanitized)
copies; the pre-sanitization hashes are in the Phase 1 evidence bundle.
