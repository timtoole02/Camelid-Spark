# WIN2METAL Phase 3 — Metal Linear Speculative Verify (`verify_drafts_metal` / `verify_batch`)

> Build spec for Phase 3 (Bucket C.1). Grounded in the recon (`WIN2METAL_RECON.md`),
> the §A2 reachability probe (`WIN2METAL_A2_REACHABILITY.md`), and a deep read of the
> Metal resident forward path. Line numbers are at base `28f224b`+phase0/A2 — verify
> against current code before editing (code may have shifted).

## 0. Goal & contract recap
Swap the macOS `#[cfg(not(feature="cuda"))]` stub of `verify_drafts_gpu`
(src/inference.rs ~2620, currently `Ok(None)`) to a real Metal speculative-verify lane
that is **BIT-IDENTICAL to `k` independent `forward_token` decodes** at positions
`[position, position+k)` (intra-backend parity-class A vs Metal decode). The
`#[cfg(feature="cuda")]` body (inference.rs ~2397) is **never touched**. We mirror the
CUDA host algorithm (embed `[last_token, drafts…]`, per-position RoPE, `verify_batch` →
greedy argmax per position, accept-longest-prefix + bonus,
`set_filled(position+accepted.len())`), `MAX_VERIFY_K=8`.

---

## 1. `verify_batch` signature + step-by-step algorithm

### 1.1 Method (new, on `metal::ResidentDecodeState`, near `prefill_tokens` ~metal.rs:10501)
```rust
pub fn verify_batch(
    &mut self,
    embeddings: &[f32],   // k * hidden  (host embedding_lookup of [last_token, drafts...])
    cos_all: &[f32],      // k * half_rope, position-major (cos for base+0 .. base+k-1)
    sin_all: &[f32],      // k * half_rope
    layers: &[ResidentLayerWeights],
    logits: &LogitsStage, // final_norm + output_weight_blocks + vocab_size
    base_position: usize, // == filled()
    k: usize,             // drafts.len() + 1, <= MAX_VERIFY_K(8)
    scale: f32,
) -> Option<Vec<u32>>     // k greedy argmax token ids (predicted[0..k])
```
Returns `None` (→ caller falls back, lossless) on any unsupported config. The host-side
accept loop lives in `verify_drafts_metal`, NOT here (matches CUDA's split: engine
returns predictions, host accepts).

### 1.2 Eligibility gate (return `None` otherwise)
- `1 <= k <= MAX_VERIFY_K(8)`, `base_position == self.filled()`, `base_position + k <= kv_cap`, `ensure_capacity(base_position + k)`.
- Production GEMV path active: `f32y_gemv_enabled() && wire_weights_enabled() && wire_nsg8_enabled()` — this is the kernel the new batched GEMV mirrors.
- `head_dim % 32 == 0 && head_dim <= 128` (v2/split-K attention precondition), `!self.kv16` (f32 cache only for Phase 3; kv16 is a follow-up).
- Weights resolvable to Q8_0 wire/blocks for every layer + output projection.

### 1.3 Buffers (allocate once; row-major `[token][dim]`, token = verify row)
`emb_buf[k*hidden]`, ping-pong `act_a/act_b[k*hidden]`, `norm_buf[k*hidden]`,
`q_buf[k*q_dim]`, `k_buf[k*kv_dim]`, `v_buf[k*kv_dim]`, `ctx_buf[k*q_dim]`,
`o_buf[k*hidden]`, `ffn_norm_buf[k*hidden]`, `gate_buf[k*ffn]`, `up_buf[k*ffn]`,
`act_ffn[k*ffn]`, `down_buf[k*hidden]`, `fnorm_buf[k*hidden]`, `logits_buf[k*vocab]`,
`pred_buf[k]` (u32). Scalar buffers reuse the existing layouts. KV cache is the resident
`cache_k/cache_v` (f16 mirrors gated out — kv16 follow-up).

### 1.4 Per-layer pipeline (single serial `MTLComputeCommandEncoder`)
For layer `l` (input `in_buf` = `emb_buf` for l=0, else ping-pong):
1. **Input RMSNorm — BATCHED.** `rms_norm_batch_f32` (~2494), grid `(k,)` 256-thread groups → `norm_buf`. Byte-exact (§2).
2. **Q/K/V projections — BATCHED, NEW KERNEL.** `encode_q8_matmul_f32y_batched` (§3.1) over `norm_buf[k][hidden]` → `q_buf`/`k_buf`/`v_buf`. Each weight streamed ONCE across all k rows.
3. **Per-head Q/K-norm (Qwen3) — PER-ROW.** For `i in 0..k`: `encode_rms_norm_per_head` (~1481) at row byte-offset `i*q_dim` (Q) / `i*kv_dim` (K).
4. **RoPE — PER-ROW.** For `i in 0..k`: `encode_rope` (~8506) on `q_buf`/`k_buf` at row `i`, binding `cos_all[i*half_rope..]`/`sin_all[i*half_rope..]` (position `base+i`). Exact single-token formula + `split_half_pairing`.
5. **K/V scatter — PER-ROW (all k before any attention).** For `i in 0..k`: `kv_scatter_f32` (~2425) `write_position = base+i`, source row offset `i*kv_dim`. Slot formula unchanged: `(h*max_positions + base+i)*head_dim + d`.
6. **Attention — PER-ROW (after all scatters).** For `i in 0..k`: `encode_attention` (~8529), query/out at row offset `i*q_dim`, scalar `position_count = base+i+1`. Reproduces forward_token's EXACT routing per row: split-K when `pc>=128 && group∈1..4 && !kv16` (`n_splits=ceil(pc/64).clamp(2,64)`), else v2. Row `i`'s `pc` caps its KV read to `[0, base+i]`, so future rows' slots are invisible despite being scattered.
7. **O-projection — BATCHED, NEW KERNEL.** `encode_q8_matmul_f32y_batched(ctx_buf → o_buf)`.
8. **Attention residual — BATCHED.** `residual_add_f32` (~1465) over `k*hidden`: `out_buf = in_buf + o_buf`.
9. **FFN — BATCHED.** `rms_norm_batch_f32(out → ffn_norm_buf)`; `encode_q8_matmul_f32y_batched` ×2 (gate, up); `silu_mul` over `k*ffn`; `encode_q8_matmul_f32y_batched` (down); `residual_add_f32`. Result → next layer's `in_buf` (ping-pong).

### 1.5 Final stage (after all layers)
10. **Final RMSNorm — BATCHED.** `rms_norm_batch_f32(final → fnorm_buf)` with `logits.final_norm`.
11. **Output GEMV — BATCHED, NEW KERNEL.** `encode_q8_matmul_f32y_batched(fnorm_buf → logits_buf[k][vocab])`.
12. **Per-row argmax — PER-ROW.** For `i in 0..k`: `argmax_f32_greedy_pipeline` (~10452, grid 1×1024) over `logits_buf[i*vocab..]` → `pred_buf[i]`. Same first-max tie-break as forward_token.
13. Commit + wait, read `pred_buf` → `Vec<u32>`.

**KV-append / rejected-slot handling:** all k K/V scattered into `[base..base+k)`. The host accept loop sets `filled = base + accepted.len()`. Rejected slots `[new_position..base+k)` are left unreferenced (`set_filled` makes them logically absent) and overwritten by the next single-token decode at `new_position`. Identical to CUDA — no explicit cleanup.

---

## 2. Byte-exactness strategy & why each batched step is identical
Invariant: each verify row `i` must produce the same logits (hence same argmax) as a standalone `forward_token` at position `base+i`.

- **RMSNorm (batched).** `rms_norm_batch_f32` vs single `rms_norm_f32` (~1436): identical `threadgroup float partial[256]`, identical tree reduction, identical `1/sqrt(partial[0]/width+eps)`, identical `in*inv*weight`. Only delta is `input + row*width`. **Proven identical.**
- **GEMV projections (batched, NEW KERNEL).** Single-token production GEMV `q8_0_block_linear_row_ksplit_f32y_wire_nsg8` (~989): NSG=8, each owns blocks `ib=sg*8+lane/4` step `NSG*8=64`; per block `sumq=Σ_{i<8} wq[i]*y[i]` then `sumf += sumq*w_scale`; finalize `simd_sum→shmem[row*32+sg]→simd_sum`. The new kernel keeps this verbatim + an inner `for t in 0..k` carrying k independent accumulators `sumf[row][t]`. Columns never interact; identical block assignment, `(sumq)*w_scale` order, two-stage reduction → **column t equals single-token bit-for-bit.** Do **NOT** reuse `wire_gemm` (~1057, NSG=4, folds `w_scale` into weight — different rounding) nor `wire_mm` (~1165, tile-MMA, self-declared "not byte-exact").
- **Per-head Q/K-norm, RoPE, scatter (per-row).** Exact single-token kernels with only a buffer base-offset and `write_position`/`position_count = base+i`. Done per-row (not the batched prefill rope/scatter kernels) to eliminate any cos/sin pairing or write-order doubt.
- **Attention (per-row).** `encode_attention` with identical scalar layout and v2/split-K routing as forward_token at `pc=base+i+1`. Identical by reuse.
- **Residual / SiLU-mul (batched).** Pure elementwise; one dispatch over `k*dim` = k single dispatches bit-for-bit.
- **Argmax (per-row).** Same kernel + first-max tie-break, bound at the row's vocab slice.

Net: the only arithmetic that changes shape is the GEMV, covered by the new mirror kernel; everything else is a proven-identical batched kernel, an elementwise batch, or the exact single-token kernel run per-row.

---

## 3. Wiring

### 3.1 metal.rs additions
- **MSL kernel** `q8_0_block_linear_ksplit_f32y_wire_nsg8_verify` next to wire_gemm (~1139). Body = `..._wire_nsg8` (989) with `constexpr uint MAX_T = 8`, `float sumf[NR0][MAX_T]`, inner `for t` over activation columns, output `output[(t0+t)*rows + r0+row]` (matches wire_gemm output layout so the next GEMM reads `[token][dim]`).
- **Pipeline field** `q8_0_block_ksplit_f32y_wire_nsg8_verify_pipeline` in `MetalLinearKernel` + compile in `new()` (next to the existing nsg8 pipeline).
- **Helper** `encode_q8_matmul_f32y_batched(e, k, y, weight, out, scalar, rows, n_rows_in)` — clone of `encode_q8_matmul_f32y` (~7162) dispatch (256 threads/TG, `width=rows.div_ceil(2)`) binding the verify pipeline + writing `n_rows_in` to the scalar.
- **Additive offset params** on `encode_rms_norm_per_head`, `encode_rope`, the kv-scatter encode, `encode_attention`, and the argmax dispatch (`*_byte_off`, default 0 for all existing callers) — mechanical, behavior-preserving.
- **`verify_batch`** method (§1). **Test** `metal_spec_verify_bit_identical` (§4).

### 3.2 src/inference/metal_resident.rs additions
`MAX_VERIFY_K=8`; `verify_drafts_metal(&mut self, last_token, drafts) -> Result<Option<Vec<u32>>>` with `#[cfg(target_os="macos")]` real / `#[cfg(not)]` `Ok(None)`. Real body mirrors CUDA `verify_drafts_gpu` (inference.rs ~2397-2492) over `self.resident_decode`:
1. early-out: `drafts.is_empty()` or resident disabled → `None`.
2. `position = kv_cache.position; k = drafts.len()+1`; guard `k<=MAX_VERIFY_K`, `position+k<=max_seq_len`, `resident_decode_eligible(true)`.
3. `inputs=[last_token, drafts…]`, `embedding_lookup`.
4. `cos_all/sin_all` via `resident_decode_rope_tables(position+i)` for `i in 0..k`.
5. `layer_views: Vec<ResidentLayerWeights>` + `LogitsStage` as in `try_resident_decode_forward_metal`.
6. Readiness: `resident_decode.is_some() && session.filled()==position` else `None`.
7. `predicted = session.verify_batch(...)?`.
8. `acc = accepted_draft_prefix(drafts, &predicted[..drafts.len()])` (speculative.rs:44 — backend-neutral); `emitted=[predicted[0], predicted[1..=acc]…]`; `new_position=position+emitted.len()`; `session.set_filled(new_position); kv_cache.position=new_position; Ok(Some(emitted))`.

### 3.3 src/inference.rs stub swap (CUDA block UNTOUCHED)
Replace ONLY the `#[cfg(not(feature="cuda"))]` body (~2625): `Ok(None)` → `self.verify_drafts_metal(last_token, drafts)`. Drop `#[allow(unused_variables)]`. The `#[cfg(feature="cuda")]` `verify_drafts_gpu` (2397) and `verify_tree_gpu` are untouched.

---

## 4. Gate: `metal_spec_verify_bit_identical`
Model-in-test pattern from `metal_resident_decode_state_matches_full_upload` (~16557): synthetic Q8_0 weights via `mkw()`, `head_dim=32`, `n_heads=4`, `n_kv=2` (group=2), `hidden=128`, `ffn=256`, `layers=2`, `vocab=256`.
- Guard `if !detect_metal_device().available { return; }`; SKIP (eprintln) when `!(f32y && wire && nsg8)` (OnceLock gates — see risks).
- **Straddle the split-K thresholds:** `base=126,k=6` (rows `pc∈[127..132]`, crossing 128 mid-window — some v2, some split-K); `base=510,k=6` (rows `pc∈[511..516]`, all >512, deep split-K merge).
- **Reference:** seed a fresh `ResidentDecodeState` with `base` synthetic KV positions, run `k` independent `forward_token` decodes at `base+i`, capture per-row argmax id AND per-row logits.
- **Candidate:** identically-seeded session, call `verify_batch`.
- **Assert:** `predicted_verify[i]==predicted_forward[i]` (u32) AND per-row pre-argmax logits **bit-cast equal** (`to_bits(a)==to_bits(b)`) — not epsilon; this is intra-backend byte-exact, so exact bits catch offset/reduction bugs argmax would mask.

---

## 5. Harness wiring + parity receipt
Add a Metal lane to `qa/speed/spec-verify-parity.sh`: export `CAMELID_SPEC_GPU=1` + the resident stack (`CAMELID_METAL_F32Y=1 CAMELID_METAL_WIRE=1 CAMELID_METAL_WIRE_NSG8=1`, resident-decode on) and run the speculative serve/bench so `verify_drafts_gpu` is actually invoked while resident decode stays on (per §A2). Losslessness check: speculative stream byte-identical to the non-speculative greedy baseline (a diff is a hard failure). Emit a `camelid.spec-verify/v1` receipt to `qa/speed/receipts/` (model id, per-round `base_position`/`k`/`accepted`, `bit_identical:true`, stream SHA vs baseline).

---

## 6. File-by-file change list + sequenced build order

**Files touched:** `src/metal.rs` (kernel + pipeline + helper + offset params + verify_batch + gate test); `src/inference/metal_resident.rs` (`MAX_VERIFY_K`, `verify_drafts_metal`); `src/inference.rs` (one-line stub swap, CUDA untouched); `qa/speed/spec-verify-parity.sh` (Metal lane + receipt).

**Build order with checkpoints:**
- **C0 — kernel + helper.** New `..._nsg8_verify` MSL, pipeline, `encode_q8_matmul_f32y_batched`. **Checkpoint:** a micro-test dispatches it over k=1..8 random rows and asserts each column **bit-equals** the single-token `encode_q8_matmul_f32y`. *Gates the entire byte-exact claim — do this first.*
- **C1 — offset params.** Thread `*_byte_off` into per-head-norm/rope/scatter/attention/argmax (default 0). **Checkpoint:** existing resident tests still pass unchanged.
- **C2 — `verify_batch`.** Assemble §1.4-1.5. **Checkpoint:** `metal_spec_verify_bit_identical` straddle-128 case green.
- **C3 — deep split-K.** **Checkpoint:** straddle-512 case green.
- **C4 — host seam.** `verify_drafts_metal` + stub swap. **Checkpoint:** `cargo build` (no-cuda macOS) clean; `cargo test --all-targets` green; serve smoke with `CAMELID_SPEC_GPU=1` shows accepted>0 and stream == greedy baseline.
- **C5 — harness + receipt.** **Checkpoint:** `spec-verify-parity.sh` emits `bit_identical:true`; lossless diff clean.

**Open questions (carry into implementation):**
1. Does the production CLI always set `CAMELID_METAL_WIRE_NSG8` (apply_default_fast_stack sets it — confirm)? If nsg8 can be off, `verify_batch` returns `None` (silent no-op) — may warrant a `_wire` (non-nsg8) batched variant.
2. Confirm `argmax_f32_greedy` tie-break (first-max) is bit-stable when bound at a row offset.
3. Phase 3 gates out kv16; confirm the default CLI decode runs f32 KV (so the lane fires).
4. Verify the next single-token `forward_token` after `set_filled(new_position)` writes at `new_position` (not `base+k`), so rejected slots are correctly overwritten.

## 7. Top risks
1. **OnceLock env-gate latching** (`f32y/wire/nsg8_enabled` cache first read process-wide) — test must read gates and SKIP if inactive; harness sets env before process start.
2. **The new batched GEMV must reproduce the nsg8 reduction tree EXACTLY** — the trap is folding `*w_scale` onto the weight like `wire_gemm` (math-equal, byte-different). Caught by the C0 micro-test + straddle gates.
3. **Per-row buffer offset binding** off-by-one (`i*q_dim` vs `i*n_heads*head_dim`, cos/sin stride, vocab stride) → wrong-but-plausible; gate compares per-row logits by bit-cast, not just argmax.
4. **Production GEMV-path assumption** — if the CLI runs non-nsg8/non-wire GEMV, verify_batch returns None (silent lossless no-op, no speedup). Confirm the env stack.
5. **Single serial compute encoder ordering** — k scatters must finish before any attention reads; keep one serial encoder.
6. **Session-state coupling** — run only when `resident_decode.is_some() && filled()==base_position`; `set_filled` must leave state consistent for the next decode.
