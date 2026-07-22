# WIN2METAL Phase 4 — Metal TREE speculative verify (`verify_tree_gpu` → `verify_tree_metal`)

> Build spec for Phase 4 (Bucket C.2). Branches off the proven Phase 3 (`verify_batch`).
> Line numbers are at the Phase 3 tip (`2168a4f`) — verify against current code before editing.

## 0. Frame & invariants
- CUDA blocks (`#[cfg(feature="cuda")]`) are NEVER touched. `cuda_resident.rs::verify_tree`/`compact_tree_kv_path` and the cuda `verify_tree_gpu` (inference.rs ~2503) are the **reference oracle**, not edited.
- Tree orchestration is shared/platform-neutral — reuse `TokenTree`, `node_depth()` (spec_tree.rs:101), `node_kvslot(base)` (:110), `ancestor_bitset()` (:119), `accept_longest_path()` (:152), `path_to()` (:83), `TREE_MAX_NODES=16` (:27). Do NOT reimplement.
- **Primary gate (ANCHOR):** a single-branch (linear) tree through the new tree path must be BYTE-IDENTICAL to Phase-3 `verify_batch` (already proven bit-identical to k single-token decodes). Achieved by *construction* (§1), not parallel reimplementation.
- **Parity-class A:** along each accepted path the tree verify == the backend's own greedy decode (every emitted token is the target's own argmax via `accept_longest_path`, exactly as the CUDA arm).

## 1. Central decision — how a node attends its ANCESTOR SET
Each node `i` attends `prefix [0, base)` ∪ `{ancestor draft slots}`, excluding siblings. The existing Metal attention kernels iterate contiguous `for p in 0..position_count` and split-K chunks over `[0,position_count)` by index. A non-contiguous ancestor set breaks the loop bound AND the split-K chunk boundaries → changes float association → not byte-identical.

**Decision: NEW `*_tree` attention kernels = line-for-line clones of the kernels the linear verify path uses, with ONE change — a transparent slot-indirection on the draft tail only.** Mirrors CUDA's `attention_tree_batched` but adapted to Metal's reduction structure so linear stays byte-exact.

Mechanism (per node, per head):
- Add two params: `constant uint& base` and `device const uint* tail_slots [[buffer(13)]]` (per-node ancestor draft slots, length `tail_count`, increasing order).
- `position_count = count = base + tail_count`.
- Replace every cache position deref `p` with `uint slot = (a < base) ? a : tail_slots[a - base];` then index `kv_base + slot*position_stride + d`.
- Keep lane striding (`a += 32` / split-K chunking over `[0,count)`), `simd_max`/`simd_sum`, Phase-3 ordering, and split boundaries (`n_splits = count.div_ceil(64).clamp(2,64)`) **identical** to the cloned kernel.
- **Prefix stays direct (not gathered):** the huge `[0,base)` history is always all-ancestor and contiguous, so `slot==a` there with zero indirection — avoids the Apple-GPU ~32KB threadgroup-memory blow-up a full CUDA-style shared slot list would hit. Only the ≤16 draft tail needs the indirection buffer.

**Byte-exact argument for linear:** on `TokenTree::linear`, node `i` has ancestor bits `{0..i}` → `tail_slots=[base..base+i]`, `tail_count=i+1`, `count=base+i+1`. Then `slot=(a<base)?a:tail_slots[a-base]==a` for every `a` — same instruction stream, floats, order, split boundaries (`count==base+i+1` = the `position_count` linear `verify_batch` passes). ⇒ bit-identical.

Rejected alternative — an ancestor mask/flag *inside* the existing hot kernels: perturbs the mainline non-spec decode codegen and risks regressing the Phase-3-proven linear path; cloning isolates risk to the new tree variants.

## 2. New Metal engine method — `verify_batch_tree`
In `src/metal.rs`, sibling to `verify_batch`; refactor `verify_batch_inner` to be tree-aware (extract a `TreeAttn` descriptor; `None` ⇒ today's contiguous `encode_attention`, byte-unchanged):
```rust
pub fn verify_batch_tree(
    &mut self,
    embeddings: &[f32],     // BFS node order, n*hidden
    cos_all: &[f32], sin_all: &[f32], // per-NODE, position base+depth[i]
    layers: &[ResidentLayerWeights],
    logits: &LogitsStage,
    node_kvslot: &[i32],    // = base+i (BFS) — scatter target, reused unchanged
    ancestor_bits: &[u32], words: usize,
    base_position: usize, n: usize, scale: f32,
) -> Option<Vec<u32>>
```
Steps (mirror `verify_batch_inner`; only attention + the descriptor differ):
1. **Eligibility** (None ⇒ lossless fallback): `(1..=TREE_MAX_NODES).contains(&n)`, `!self.kv16`, `f32y && wire && nsg8`, `head_dim%32==0 && ≤128`, `base_position==self.filled()`, `base_position+n<=cap`, `layers.len()==n_layers`, `embeddings.len()==n*hidden`, RoPE-length checks.
2. **Per-node RoPE / embeddings / norm / GEMV — reused unchanged** over `k:=n` rows (`rms_norm_batch`, `encode_rope` per-row offsets, `encode_q8_matmul_f32y_batched`, residual/FFN/argmax).
3. **K/V scatter — reused unchanged.** Node `i` → slot `node_kvslot[i]==base+i`; "ALL n before any attention" ordering preserved. No new scatter kernel.
4. **Host-side per-node `tail_slots` build** (before the layer loop): scan `ancestor_bits[i*words + j/32]` bit `j%32` for `j in 0..n`; collect `base+j` for set bits in increasing `j` → `tail_slots[i]`, `tail_count[i]`. Flatten into one buffer `tail_slots_all` + write `count=base+tail_count[i]` into each node's `attn_scalar` (offset 8). Linear ⇒ `tail_slots[i]=[base..base+i]`, `count=base+i+1`.
5. **Attention — call new `encode_attention_tree`** per node, routing (v2 / splitk_kv16 / splitk_kv16_direct / f32) **exactly like `encode_attention`** but dispatching `*_tree` pipelines + binding `base` + `tail_slots_all` at the node's offset. Reduction/struct identical to the linear kernel at that `count`.
6. **Logits + argmax** reused → return `predicted: Vec<u32>` (n entries).

Add `#[cfg(test)] verify_batch_tree_logits(...) -> (Vec<u32>, Vec<f32>)` (n*vocab) mirroring `verify_batch_logits` for the gate.

## 3. KV compaction — `compact_tree_kv_path` (Metal)
Mirror CUDA `compact_tree_kv_path`, but exploit unified memory — direct `contents()` memcpy, no dtoh/htod:
```rust
pub fn compact_tree_kv_path(&mut self, path: &[usize], base: usize) -> Result<(), String>
```
For each rank `r` with `path[r]!=r` (CUDA proves `path[r]>=r` ⇒ in-place gather is order-independent), copy `head_dim` rows from slot `base+path[r]` → `base+r`, for **every layer, every kv_head, in BOTH the f32 cache AND the f16 mirrors** (both are read by subsequent decode/attention). Linear path `=[0..L-1]` ⇒ all `r==node` ⇒ no-op (the byte-identity anchor: linear tree leaves the cache exactly as a linear decode).

## 4. Seam wiring
**`src/inference/metal_resident.rs`** — `verify_tree_metal` (tree twin of `verify_drafts_metal`) + non-macOS `Ok(None)` stub:
```rust
#[cfg(target_os = "macos")]
pub(super) fn verify_tree_metal(&mut self, tree: &spec_tree::TokenTree) -> Result<Option<Vec<u32>>>
```
Body mirrors the cuda `verify_tree_gpu` host but on `self.resident_decode`:
- gates: `resident_decode_metal_enabled()`, `!resident_paths_disabled`, `n=tree.nodes()` in `1..=TREE_MAX_NODES`, `position+n<=max_sequence_length`, `resident_decode_eligible(true)`, `weights.layer_range.is_none()`, `filled()==position`.
- `embeddings=lookup(&tree.tokens)`; per-node `cos_all/sin_all` at `position+node_depth[i]` (reuse the node_depth loop); `node_kvslot=tree.node_kvslot(position)`; `(ancestor_bits, words)=tree.ancestor_bitset()`; `layer_views`+`logits_stage` (reuse `verify_drafts_metal`'s builders).
- `predicted = session.verify_batch_tree(...)?` → `None` ⇒ `Ok(None)`.
- `let (emitted, leaf) = tree.accept_longest_path(&predicted); let path = tree.path_to(leaf); session.compact_tree_kv_path(&path, position)?;`
- `session.set_filled(position+emitted.len()); self.kv_cache.position += emitted.len(); Ok(Some(emitted))`.

**`src/inference.rs`** — swap ONLY the `#[cfg(not(feature="cuda"))] verify_tree_gpu` stub body `Ok(None)` → `self.verify_tree_metal(tree)`. The cuda arm is untouched. `main.rs` call site unchanged.

## 5. New MSL kernels + pipelines
Clone + register: `attention_decode_v2_tree` (the pc<128 path the base=126 gate hits), `attention_decode_splitk_kv16_tree`, `attention_decode_splitk_kv16_direct_tree` (head_dim==128), `attention_decode_f32_tree` (universal fallback for any gate config). Each = verbatim clone of its base kernel + the `base`/`tail_slots` params + the `slot=(a<base)?a:tail_slots[a-base]` deref. No new scatter/rope/norm/gemv/argmax kernels.

## 6. Gate
1. **Unit test `metal_tree_verify_bit_identical`** (clone of `metal_spec_verify_bit_identical`): same synthetic 2-layer/4-head/group-2/head_dim-32 setup, seeded `base` history. For each base, build `TokenTree::linear(anchor, drafts)` and assert `verify_batch_tree_logits` == `verify_batch_logits` by exact `to_bits()` on every `n*vocab` logit AND argmax id. **Straddle split-K: run(126,…) and run(510,…)** (mandatory — deep split-K). Add a **branching** tree case asserting (a) compaction makes the accepted-path cache == an independent linear decode of that path, and (b) Parity-class A: each accepted token == reference `forward_token` argmax along the path. Latch F32Y/WIRE/NSG8/ATTN2 with the skip-guard.
2. **`spec_tree_lossless.rs`**: host test asserting `accept_longest_path` on a linear tree reproduces `accepted_draft_prefix` (GPU-free).
3. **`spec-verify-parity.sh` tree lane → LOSSLESS gate.** Upgrade the existing "trivially lossless / no rounds" tree lane: drive `serve` with a tree-spec config so `verify_tree_gpu` actually fires (`gpu_verify_rounds>0`), assert the spec token+text stream is SHA-256 byte-identical to the plain-greedy baseline, require the tree verify trace fired (a silent CPU-fallback pass is a FAILURE). Emit a `camelid.spec-verify/v1` receipt with a `tree` lane block.

## 7. Sequenced build order (checkpoints)
1. **MSL kernels** — add the 3–4 `*_tree` kernels + pipelines (checkpoint: pipelines build, no dispatch yet).
2. **`encode_attention_tree`** router + `compact_tree_kv_path` host (checkpoint: `cargo build` macOS clean).
3. **`verify_batch_tree`** (refactor `verify_batch_inner` tree-aware; None-tree path keeps `verify_batch` byte-unchanged) + `verify_batch_tree_logits` (checkpoint: **`metal_spec_verify_bit_identical` still passes** → linear path unregressed).
4. **Unit test `metal_tree_verify_bit_identical`** (checkpoint: linear-tree==verify_batch bit-exact at base 126 & 510; branching compaction + class-A). **The gating moment.**
5. **Seam**: `verify_tree_metal` + the inference.rs stub swap (checkpoint: `cargo build` macOS + `cargo build --features cuda` both clean; cuda arm diff-free).
6. **Harness**: tree lane LOSSLESS + receipt (checkpoint: e2e serve tree lane LOSSLESS + gpu_verify_rounds>0).
7. **Full gate**: targeted tree+linear tests; `cargo test --all-targets`; parity script; cuda untouched verified.

## 8. File-by-file
- `src/metal.rs`: +`verify_batch_tree`, +`verify_batch_tree_logits`(test), refactor `verify_batch_inner` tree-aware, +`encode_attention_tree`, +`compact_tree_kv_path`, +3–4 MSL kernels & pipeline fields, +`metal_tree_verify_bit_identical` test.
- `src/inference/metal_resident.rs`: +`verify_tree_metal` (macos) + non-macos `Ok(None)` stub.
- `src/inference.rs`: swap non-cuda `verify_tree_gpu` body (cuda arm untouched).
- `src/inference/spec_tree*.rs`: reuse; +1 host equivalence test (no API change).
- `qa/speed/spec-verify-parity.sh`: tree lane → LOSSLESS gate + receipt block.
- (No changes to `cuda_resident.rs`, `main.rs`.)

## 9. Top risks
1. **Split-K float-association**: byte-identity hinges on the tree kernel computing `n_splits`/chunk boundaries over `count=base+tail_count` exactly as the linear kernel over `position_count=base+i+1`. Any off-by-one in `count` silently breaks bit-identity only on deep (>512) contexts — the base=510 straddle test is mandatory.
2. **Refactoring `verify_batch_inner`** could perturb the proven linear path — keep the None path calling existing `encode_attention` unchanged and re-run `metal_spec_verify_bit_identical` as a regression checkpoint before touching the tree path.
3. **Forgetting to compact the f16 mirrors** (cache_k16/v16) as well as f32 — split-K decode reads the mirrors; mirror-stale compaction passes linear (no-op) but fails the branching/class-A test.
4. **Kernel-variant coverage gap** — clone every variant `encode_attention` can route to under the verify gate (v2, splitk_kv16, splitk_kv16_direct, f32), else a silent non-fire (gate FAILURE).
5. **`tail_slots` indexing/stride bugs** corrupt only branching trees while linear passes — needs an explicit multi-branch fan-out test, not just the linear anchor.
6. **Readiness mismatch** (filled()==position, layer_range.is_none, n in 1..=TREE_MAX_NODES) — keep the checks identical to the cuda host and `verify_drafts_metal`.
