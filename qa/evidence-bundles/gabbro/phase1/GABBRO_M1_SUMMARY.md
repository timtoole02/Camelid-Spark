# GABBRO M1 — ARM NVFP4 decode receipt (Apple M4)

**This note promotes no model row, makes no throughput claim, and no portability claim beyond the one it proves: the committed NVFP4 decode/format/refusal tests pass bit-exact on Apple Silicon (ARM).** It records a test run, not a support decision. GABBRO is a DRAFT campaign; nothing here is merged.

## Provenance
- Campaign: GABBRO (macOS/Metal expansion of BASALT), Gate **G-M1**
- Host: `Apple M4`, 10-core (4P+6E), 16 GiB unified, macOS 26.5 (25F5058e), Darwin arm64
- Toolchain: AppleClang 17.0.0, Command Line Tools SDK (no Xcode.app)
- Worktree: `<worktree>` @ `510fa51` (origin/main), branch `gabbro/m1-arm-decode`
- Target dir: `<target-dir>` (documented T7 build target)
- UTC: 2026-07-18T18:36Z

## What ran and the result
`cargo test -p camelid --test nvfp4_format --test nvfp4_e4b_spotcheck --test nvfp4_wire_lane_refusals` — **13/13 passed**, exit 0 (compile 45.36s, warm deps).

| Test binary | Tests | Result |
|---|---|---|
| `nvfp4_format.rs` | 9 | ✅ `ue4m3_table_bit_exact_vs_pin`; `decode_table_all_4096_scale_code_pairs_bit_exact`; `decode_table_nibble_probes_lock_packing_order`; `random_blocks_bit_exact_through_both_paths`; `real_gguf_blocks_bit_exact_through_both_paths`; + 4 fail-closed refusals (NaN-sentinel / wrong byte-length / non-64-multiple / scanner) |
| `nvfp4_e4b_spotcheck.rs` | 1 | ✅ `produced_pilot_row_blocks_bit_exact_through_both_paths` — blocks from the produced gemma-4-E4B NVFP4-mm pilot decode bit-exact on ARM |
| `nvfp4_wire_lane_refusals.rs` | 3 | ✅ `fixtures_are_byte_pinned`; `sidecar_fixture_trips_d_b2_end_to_end`; `nan_sentinel_fixture_trips_the_scan_seam` |

Hygiene on the unmodified `origin/main` worktree: `cargo fmt --all --check` PASS (exit 0); `cargo clippy -p camelid --lib --tests` PASS (exit 0, 0 warnings).

## What this establishes (and what it does not)
- The golden vectors are **pin-generated**; bit-exact here means Camelid's Apple-Silicon NVFP4 decode reproduces the pin's decode bit-for-bit — including the UE4M3 scale table, all 4096 (scale × E2M1 code) pairs, real GGUF blocks, and real produced-pilot blocks.
- **O2 (the reported "high error on ARM", llama.cpp #21462) is resolved favorably for our lane.** That defect concerned llama.cpp's own ARM CPU path; Camelid uses its own decode, and it is bit-exact on this M4. (Narrow: this validates Camelid's decode, not llama.cpp's.)
- The D-B2 sidecar refusal and the D17/T5 NaN-sentinel seam fire correctly off-Windows too.
- **Not proven here:** a full end-to-end pilot token run (needs the 6 GB `gemma-4-E4B-it-NVFP4-mm.gguf`, sha256 `eb293344…9863d9`, which is not committed); `cargo-test-all` on ARM (this run was scoped to the decode/format/refusal binaries).

## Artifacts
- `nvfp4-arm-decode-test.txt` — full test output
- `cargo-clippy.txt`, `cargo-clippy.status`, `cargo-fmt.status`, `cargo-test-nvfp4.status`
- `hw_probe_mac.json`
- `SHA256SUMS`
