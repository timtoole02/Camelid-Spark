# GABBRO M3 — Metal NVFP4 GEMV kernel + self-parity gate

**Scope claim:** delivers the NVFP4 Metal decode kernel and proves it correct by
Metal-vs-CPU self-parity (§3 leg 3, correctness before timing). This is the
correctness half of M3; the perf table and full resident-lane integration are the
M3-followup (see below). No support-status change — NVFP4 stays engine-facts, not a
supported row (G3 NO-GO stands), and the Metal *resident* lane still typed-refuses
NVFP4 via `nvfp4_metal_lane_check` (the kernel here is exercised through the
standalone `try_gemma4_nvfp4_matmul_f32y` path, behind the existing opt-in gating).

## Provenance
- Campaign: GABBRO, Gate **G-M3** (STOP for review)
- Host: Apple M4, 16 GiB, macOS 26.5, Darwin arm64; branch `gabbro/m3-metal-nvfp4` (stacked on M2)
- UTC: 2026-07-18

## The kernel
`nvfp4_block_linear_row_ksplit_f32y_wire` (in `LINEAR_ROW_SHADER`) reproduces the CPU
oracle `nvfp4_wire_block_dequant` × f32 activation. Decode primitives are bit-for-bit
twins of `src/tensor/mod.rs`: `NVFP4_KVALUES` (E2M1 codebook) + `nvfp4_ue4m3_to_f32`
(UE4M3 sub-block scale via `ldexp` — exact powers of two, so no rounding vs the Rust
const). **Geometry:** `s = lane%4` owns one 16-value sub-block *and* its matching
UE4M3 scale `d[s]`; `ix = lane/4` selects the 64-value superblock slot in flight — the
4-subscale structure maps exactly onto the existing 4-way lane split, and the
`simd_sum` reduction is identical to the q4_0 kernel. No per-tensor scale (D-B2 refuses
sidecar-scale NVFP4).

## The §5 geometry gotcha, fixed
`blocks_per_row` was hardcoded `dim / 32` across the gemma4 encode paths — wrong for the
64-value NVFP4 superblock. Added `GemmaWireFmt::block_elements()` (32 for Q8_0/Q4_0, 64
for NVFP4) as the single source of truth; the standalone NVFP4 entry sizes activations
by it. NOTE: ~40 `dim/32` sites exist across FFN/attention/resident/test paths, most
Q8/Q4-only or `fmt`-less; auditing which the NVFP4 resident lane traverses is the
M3-followup, not this kernel commit.

## Seam touched (small)
`GemmaWireFmt::Nvfp4` + `wire_bytes()=>36` + `block_elements()`; the
`encode_gemma4_matmul` dispatch arm; a pipeline field + creation;
`encode_gemma4_nvfp4_matmul`; `try_gemma4_nvfp4_matmul_f32y` (standalone, `y` sized
`blocks_per_row * 64`); the MSL kernel + decode helpers; the parity test. Rust forced
only the 2 `GemmaWireFmt` match arms.

## Verify (Apple M4) — all green
- `cargo fmt --all --check`: PASS
- `cargo clippy -p camelid --lib --tests`: clean, no warnings
- `metal_gemma4_nvfp4_matmul_f32y_matches_cpu`: **OK on the M4 GPU** (bpr 3/40/160, tol
  2e-2 vs `nvfp4_wire_block_dequant`) — ran alongside q4_0 + q8 parity (all OK, so the
  MSL library compiled clean). No regression: **lib 862/0**, nvfp4_format 9,
  nvfp4_e4b_spotcheck 1, invariant_matrix_binding 9, nvfp4_wire_lane_refusals 3.
- Logs: `nvfp4-parity.txt`, `m3-verify.txt`, `m3-diff.stat`.

## M3-followup (not in this commit)
1. Perf table: decode tok/s + achieved GB/s vs Q8_0 / Q4_K_M on this box (STAMPEDE
   hygiene, ≥5 warm medians) — needs the kernel exercised at scale.
2. Wire NVFP4 through the resident lane: format-aware `blocks_per_row` audit across the
   FFN/attention/layer paths (using `block_elements()`), the `GgufTensorType→GemmaWireFmt`
   arm, and lifting `nvfp4_metal_lane_check` — behind the existing opt-in flags.
