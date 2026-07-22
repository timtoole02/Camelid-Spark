# WIN2METAL — Phase 0 Recon

Spec id: `WIN2METAL` · Base: `28f224b` (Merge PR #341, fix/spec-verify-splitk-parity)
Host: Apple **M4**, 16 GB, 10-core (4P + 6E) · Quant: Q8_0 · Decoding: greedy
llama.cpp oracle (macOS-built): `acd79d603` (`/Volumes/Untitled/llama.cpp-metal-parity`)
Worktree: `/Volumes/Untitled/Camelid-win2metal` · branch `win2metal/conductor`
Base build: clean on macOS/Metal, `camelid v0.1.7-92-g28f224b`, release 1m35s.

This document records the §2.1 facts the later phases assume **and** the empirical
findings from running the speculative path on this host. The empirical findings
**revise three load-bearing assumptions in the conductor** — read §A first.

---

## §A — Headline findings that revise the conductor (read first)

### A1. The CPU chunk verify is NOT lossless on Mac at base `28f224b`
The conductor states repeatedly that the non-CUDA CPU chunk verify is "lossless
either way." **On this M4 it is not.** Running the authored harness
(`qa/speed/spec-verify-parity.sh`, Llama-3.2-1B-Instruct-Q8_0, greedy):

```
Linear lane:  DIVERGE on ALL 7 columns   (div_idx 2–110, accept 0–71%, cpu_rounds 1–25)
Tree lane:    LOSSLESS on all 7 — trivially (no spec rounds ever fire; cpu_rounds=0)
FAIL: 7/14 (column x lane) pairs diverged from plain greedy.   exit 1
```

Root cause (discriminated, not assumed):
- With **no verify round** (`rounds:0`), spec == plain greedy → `lossless:true`
  (e.g. a short creative prompt).
- The moment a **verify round fires**, the stream diverges. `creative_writing`
  is the cleanest proof: **1 verify round, 0% acceptance, DIVERGE at token 110** —
  a single CPU verify forward emitted a correction token ≠ plain greedy's argmax.
- `generate_run` (plain) and `generate_run_speculative` (spec) are independent
  fresh-session runs (`src/main.rs:3521` / `:3475`), so the only differing
  component is the verify path. The emitted token on a 0-acceptance round is
  `predictions[0]` from `forward_greedy_verify_chunk(&batch)` (`src/main.rs:3272`,
  a **batched multi-token CPU forward**). Therefore **`forward_greedy_verify_chunk`
  is not byte-exact with single-token greedy decode** on Mac at this base.

This is the campaign's exact byte-exactness problem — but on the **CPU** path the
conductor assumed was safe. It means Phase 0's literal exit ("harness runs green
in degenerate/fallback form") **cannot be met as written**: the harness is correct
and is catching a real pre-existing defect. The gate goes green only once the
verify forward is made byte-exact (CPU) and/or the Metal GPU verify (byte-exact by
construction) becomes the active path.

### A2. The Metal resident engine is not reached by `bench-speculative` on Mac
The conductor's Phase 3/4 e2e gate is `spec-verify-parity.sh`, which (mirroring the
`.ps1`) drives `camelid bench-speculative`. On Mac:
- `bench-speculative` runs on the **CPU forward path**. `CAMELID_METAL_RESIDENT_DECODE=1`
  (and the `+WIRE +F32Y` serve bundle) produced **byte-identical output and timing**
  in both `bench-generate` and `bench-speculative` — i.e. **no effect**; the resident
  Metal path is not engaged by these CLI harnesses.
- The auto-enable of the Metal resident stack lives only in the `serve` arm
  (`apply_serve_nocopy_default`, `src/main.rs:3798`), and the comment there notes
  "speculative decoding **disables resident decode** (its CPU repack plan needs the
  materialized blocks)" (`src/main.rs:3793`, `:3815`). `generate_run_speculative`
  calls `set_resident_paths_disabled(false)` (`src/main.rs:2932`) "so verify_drafts_gpu
  engages", but that is predicated on a CUDA device being present.

**Consequence:** even after Phase 3 lands a byte-exact Metal `verify_batch`, this
harness will **not** exercise it until `bench-speculative` (or the harness) is wired
to engage the Metal resident engine on Mac under speculation. This reachability gap
is a prerequisite for an honest Phase 3/4 e2e gate and is **not in the conductor**.

### A3. The Metal split-K threshold is **128**, not 512
The conductor repeatedly calls the Metal split-K regime "the analog of CUDA
`SPLITK_THRESHOLD=512`". The actual Metal numbers (`src/metal.rs`):
- Switch into split-K decode when **`position_count >= 128`** (`:8558`).
- **`n_splits = position_count.div_ceil(64).clamp(2, 64)`** (`:8560`).
- Gated additionally on `v2 && !kv16_enabled() && splitk_attention_enabled() &&
  group ∈ 1..=4` (`:8554`).

So the split-K reduction is hit by **nearly every realistic prompt**, not just
`longctx_splitk`. Phase 3's verify must reproduce split-K decode's chunked reduction
above 128 positions (the merge is deterministic, `src/metal.rs:1943`), exactly as
CUDA emulates split-K above 512. The flat reduction in `attention_prefill_v2_f32`
matches non-split decode only (`pc < 128`).

### A4. `attention_prefill_v2_f32` is position-0-only
The byte-exact multi-token kernel hard-codes a position-0 base:
`position_count = t + 1u` (`src/metal.rs:3655`), `q_base = (t*n_heads+head)*head_dim`
(`:3657`), KV loop `for p in 0..position_count` with `keys + kv_base + p*position_stride`
(`:3671`–`:3681`), and `kv_base_offset` carries head/layer strides only, **not** a
position offset (`:1618`). The per-(token,head) reduction order is position-agnostic
(NSG=4 strided + threadgroup merge), but it reduces over `[0, t]`, not `[0, base+t]`.
**Phase 3 needs a base-position parameter on the kernel** so query row `i` attends
to the pre-seeded KV `[0, base)` + in-batch K/V `[base, base+i]` — a kernel change,
not a dispatch-time offset.

### A5. Minor: prompt-pack column names; banner is Metal-blind; host RAM is None
- `qa/speed/prompts.json` columns are: `code_completion`, `structured_json`,
  `repetitive_extraction`, `normal_chat`, `creative_writing`, `adversarial_lowaccept`,
  `longctx_splitk`. There is **no standalone `longctx`** column (the conductor lists
  one); `structured_json` exists and the conductor omits it. The mandatory split-K
  canary `longctx_splitk` is present.
- `[hw] GPU: none detected` is **CUDA-centric / Metal-blind** (`src/capability.rs:104`,
  `:114`) — it is not evidence of the active path on Mac.
- `[hw] … RAM 0.0 GiB free / 0.0 GiB total` confirms `gait::host_ram_status()` returns
  `None` on macOS live (Phase 1's target).

---

## §B — §2.1 fact confirmations (file:line)

### B1. Metal resident engine surface — `metal::ResidentDecodeState` (`src/metal.rs`)
`verify_batch` will sit beside these (all confirmed):
| method | line | signature |
|---|---|---|
| `new` | 9842 | `new(n_layers, n_heads, n_kv_heads, head_dim, hidden, ffn_dim, max_positions, cap, eps, split_half_pairing) -> Option<Self>` |
| `prefill_tokens` | 10508 | `prefill_tokens(&mut self, embeddings, n_tokens, layers, cos_all, sin_all, scale) -> Option<()>` |
| `seed_layer` | 11528 | `seed_layer(&mut self, layer, keys, values, seed_positions) -> bool` |
| `set_filled` | 10035 | `set_filled(&mut self, n)` → `self.filled = n` |
| `filled` | 10030 | `filled(&self) -> usize` |
| `forward_token` | 10046 | `forward_token(&mut self, embedding, layers, cos_t, sin_t, position, scale, logits_stage, sample_stage, input_token_id, next_rope) -> Option<ResidentTokenOut>` |
| `forward_token_hidden` | 11963 | stub returning `None` (Gemma4 path) |

KV cache: four `Vec<Buffer>` — `cache_k`, `cache_v` (f32), `cache_k16`, `cache_v16`
(f16) (`:9775`). `filled` (`:9788`) = positions materialized (seeded + appended),
updated **only** via `set_filled`. The model-in-test pattern for the new gate is
`metal_resident_decode_state_matches_full_upload` (`:16557`): `ResidentDecodeState::new(...)`
then `forward_token(... None, None ...)` returning `ResidentTokenOut::Data(Vec<f32>)`.

### B2. Metal split-K decode kernels (`src/metal.rs`)
`attention_decode_splitk_f32` (`:1846`), `attention_decode_splitk_merge_f32` (`:1933`),
`attention_decode_splitk_kv16` (`:1969`), `attention_decode_splitk_kv16_direct`
(head_dim==128, `:2180`). Dispatch: split groups `(n_kv_heads, n_splits, 1)` ×128
threads (`:8601`); merge `(n_heads,1,1)` ×128 (`:8618`). chunk = `(pc + n_splits-1)/n_splits`
(`:1870`). Threshold/formula/determinism per §A3. **Open:** whether the GPU-**resident**
decode (`forward_token`) dispatches this same split-K path or a different variant
(memory notes a `CAMELID_METAL_ATTN_QUAD` position-quad split-K kernel, default-on) —
**Phase 3 must confirm which attention `forward_token` actually runs at each pc**, and
make verify match *that*. (Not resolved in Phase 0.)

### B3. Existing Metal-vs-CPU parity tests (`src/metal.rs`)
`metal_q8_0_ksplit_gemv_matches_cpu_reference` (`:12559`),
`metal_attention_decode_v2_matches_cpu_reference` (`:12627`),
`metal_attention_decode_splitk_kv16_matches_cpu_reference` (`:12686`).
Shared pattern: `#[cfg(target_os="macos")] #[test]`; early-return if
`!detect_metal_device().available` (`:12400`); inline pure-Rust CPU reference;
buffers via `metal_linear_kernel()?` + `StorageModeShared` + `write_buffer_*`
(`:8253`). **They assert float tolerance (`abs < 1e-4`), NOT u32 bit-identity** —
because they compare GPU vs a CPU reference. Phase 3's `metal_spec_verify_bit_identical`
compares **GPU verify vs GPU decode** (same precision) → **u32 bit-cast identity is
the correct, achievable bar**, swept across a pc set straddling 128 (and 512+).

### B4. CUDA verify orchestration — the spec to mirror (do NOT edit)
`src/inference.rs`: `#[cfg(feature="cuda")] verify_drafts_gpu` (`:2397`),
`verify_tree_gpu` (`:2504`). Linear: embeds `[last_token, drafts...]` via
`embedding_lookup` (`:2417`), builds per-position RoPE `cos_all/sin_all` over k
positions via `rope::resident_decode_rope_tables(position+i, …)` (`:2425`), calls
`engine.verify_batch(&embeddings, &cos_all, &sin_all, position, k, scale)` (`:2470`).
**Acceptance loop (verbatim, `:2479`):**
```rust
let mut accepted = vec![predicted[0]];
let mut j = 0usize;
while j < drafts.len() && drafts[j] == predicted[j] {
    accepted.push(predicted[j + 1]);
    j += 1;
}
let new_position = position + accepted.len();
slot.engine.set_filled(new_position);
```
CUDA `verify_batch` (`src/cuda_resident.rs:4758`):
`verify_batch(&mut self, embeddings, cos_all, sin_all, base_position, k, scale)
-> Result<Vec<u32>, String>` (greedy argmax per position). `MAX_VERIFY_K=8` (`:3315`),
CUDA `SPLITK_THRESHOLD=512` (`:2783`).
- The **linear** acceptance loop is **inline** in the cuda block (not shared). A
  reusable `accepted_draft_prefix(drafts, predictions) -> usize` already exists at
  `src/inference/speculative.rs:44` (used by the CPU chunk verify, `src/main.rs:3273`)
  but the cuda linear path does not call it. **Phase 3 should lift the accept-prefix
  + bonus into a backend-neutral helper** and have Metal reuse it.
- The **tree** path already uses backend-neutral `tree.accept_longest_path(&predicted)`
  (`src/inference.rs:2591`) over `spec_tree.rs` / `spec_tree_lossless.rs` — Phase 4
  only adds the Metal engine method + stub swap.

### B5. The §0.2 trap — confirmed
`#[cfg(not(feature="cuda"))] verify_drafts_gpu` (`src/inference.rs:2620`) and
`verify_tree_gpu` (`:2608`) both return `Ok(None)` — these stub bodies are what
Phase 3/4 swap to `self.verify_drafts_metal(...)` / `verify_tree_metal(...)`. The
`#[cfg(feature="cuda")]` bodies stay byte-for-byte untouched.

### B6. Harness contract (`qa/speed/spec-verify-parity.ps1`)
Drives `camelid bench-speculative <model> --drafter ngram --workload <id>
--prompt-file <f> --max-tokens <n> --warmup`; lane toggle `CAMELID_SPEC_TREE` (unset =
linear, `=1` = tree). Reads one stdout JSON line; verdict from `lossless` +
`first_divergent_generated_token_index` (`< 0` ⇒ lossless). The Mac twin
(`qa/speed/spec-verify-parity.sh`, authored this phase) mirrors this exactly and
additionally derives the GPU-verify lane status from `gpu_verify_rounds` /
`cpu_verify_rounds` (record struct `src/main.rs:3337`–`3352`, all fields confirmed).

---

## §C — Deliverables & state

- **`qa/speed/spec-verify-parity.sh`** — authored, executable, verified to run
  end-to-end against the base binary. It is a **correct gate**: it currently reports
  `FAIL 7/14` because of finding §A1 (not a harness bug). Its degenerate/fallback
  "green" state is **blocked on §A1** (CPU verify byte-exactness) and its ability to
  exercise the GPU lane is **blocked on §A2** (Metal-resident reachability).
- **This doc** — `docs/perf-deep-dive/WIN2METAL_RECON.md`.
- Phase 0 `git diff` touches only `qa/` and `docs/` (per the contract).

## §D — Impact on Phases 1–4
- **Phase 1** (`host_ram_status`): unaffected, confirmed needed (§A5). Good warmup.
- **Phase 2** (capability validation): unaffected; runs on the shared API/CPU path.
- **Phase 3** (Metal linear verify): two **new** prerequisites the conductor omitted —
  (i) §A2 reachability (engage the Metal resident engine under speculation on Mac so
  the verify is exercised + e2e-testable), and (ii) the byte-exact reference must be
  split-K-aware at **pc≥128** (§A3) with a base-position kernel param (§A4). The
  §A1 CPU-verify divergence is the same byte-exactness class and is worth fixing so
  there is a trustworthy lossless baseline to compare against.
- **Phase 4** (tree verify): orchestration already shared; same §A2/§A3 prerequisites.
