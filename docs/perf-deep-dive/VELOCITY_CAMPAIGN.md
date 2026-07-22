# Velocity Campaign — receipts

All-Rust (inline-NVRTC CUDA, no C++/CUTLASS). Box: RTX 3060 **Laptop** (GA106, sm_86,
30 SM, **6 GB**, ~273 GB/s STREAM), i7-11800H. Competitor: llama.cpp `acd79d603`.
Scoreboard unit = decode tok/s and **% of roofline** (273 GB/s ÷ weight-bytes/token).
Source commit: `feat(perf): velocity campaign …` (this branch).

## Wins (proven, bit-parity / lossless, receipted)

### 1. 8B on a 6 GB card — Q4_K/Q6_K fused-dequant resident decode (capability headline)
**Qwen3-8B-Q4_K_M runs FULLY VRAM-resident on the 6 GB laptop 3060** — a model that
cannot fit in Q8 (8.4 GB → host-offload). Confirmed first-hand, twice:
```
layers_resident=36  layers_offloaded=0  peak=5.76GB  decode ~11.5–12.8 t/s (~20–22% roofline)
RESIDENCY: PASS — zero host offload ; coherent output
```
- `q4k_gemv` / `q6k_gemv`: warp-per-row, **lane-0 ordered-sum bit-parity anchor**, on-the-fly
  packed-wire unpack (nibbles + kmask scales / 6-bit), Q8_K activation. Bit-identical to the
  CPU oracles `q4_k_wire_row_dot` / `q6_k_wire_row_dot` (`q4k_gemv_matches_oracle`,
  `q6k_gemv_matches_oracle`, 96/96 rows, 0.000 diff).
- Wire-only K-quant loader (keeping f32 would be ~32 GB → OOM); per-tensor `ProjQuant`
  dispatch; `resident_decode_eligible` admits Q8_0 | Q4_K | Q6_K (Q8_0 byte-identical).
- Q4_K_M is a *mixed* quant (217 Q4_K + 37 Q6_K tensors), so Q6_K resident support was
  mandatory — added.
- **+15% from a parity-safe q4k_gemv rewrite** (drop `char[256]` local array → inline expand
  + `uint4` wide loads on the 16-aligned 144 B super-block; SM 42%→69%; 11.1→12.78 t/s,
  matched-clock interleaved A/B). Harness: `qa/speed/residency_check.sh`.

### 2. Lossless GPU tree verification (correctness moat)
Generalized linear speculation into a verified token-**tree** in one weight-read.
- `attention_tree_batched` (dense prefix + ancestor-bitset mask), `kv_scatter_tree_batched`,
  `verify_tree`, **compact-by-rescatter** KV commit; CPU seam `TokenTree` +
  `accept_longest_path` + Suffix/Recycling/Merge drafters.
- **Lossless, proven token-identical to greedy**: `tree_linear_matches_verify_batch`,
  `tree_verify_multiround_lossless` (40 tok/37 rounds), `tree_verify_forced_compaction_lossless`
  (16/16 compacting rounds — the silent-KV-corruption guard).
- Acceptance-gated drafting (`CAMELID_SPEC_TREE`) cut the prose regression (0.76→0.90 S_sync)
  while keeping the repetitive win (1.2×).

## Dead-ends (receipts too)
- **Coalesced split-K attention** (`CAMELID_ATTN_COALESCED`, default-off): parity-clean but
  **SPEED-NULL** — only +10% (~+4% matched-clock); coalescing does not stack on split-K.
  Receipt: `qa/speed/coalesced-attn-spike-result.md`.
- **Tree-verify as a *uniform* speed win**: floor-limited on this 6 GB box — wins repetitive
  (1.27×), but ~0.90–0.94× on code/JSON/prose even gated (per-verify overhead floor).
- **q6k_gemv wide-load rewrite**: 2.3× *slower* — the 210 B Q6_K wire is not 16-aligned;
  reverted.

## Honest assessment
The shock is **capability** — 8B on a 6 GB card, all-Rust, lossless tree-spec — **not a raw
tok/s record.** ncu shows the k-quant GEMVs are occupancy/compute-bound (not DRAM-bound), and
this throttling 6 GB laptop has hard floors (bandwidth, occupancy, per-token + per-verify
overhead). Reaching 25–35 t/s on 8B-Q4_K needs cracking q6k's non-aligned wire + the non-GEMV
per-token overhead — diminishing returns under the bit-parity accumulation-order constraint.
Bigger speed gains likely need better silicon.

## Reproduce (this box)
```
cargo build --release --features cuda --bin camelid
cargo test --release --features cuda --lib -- --include-ignored q4k q6k tree_verify   # parity gates
bash qa/speed/residency_check.sh                                                       # 8B-on-6GB residency
```
Harnesses: `qa/speed/{residency_check,spike_speed_ab}.sh`,
`{parity_check,ab_summary}.mjs`.

`tree_verify_check.sh` was retired (BARCHAN Phase 0). It drove `bench-generate`, which
never reads `CAMELID_SPEC_TREE` — the only read site is `generate_run_speculative`
(`src/main.rs`, reached from `bench-speculative`) — so it silently compared plain decode
against plain decode and could never have exercised the tree verify. Its intended job is
done correctly by the tree lane in `qa/speed/spec-verify-parity.sh`, which drives
`bench-speculative` and gates on `lossless && gpu_verify_rounds > 0`.
