# BARCHAN — Phase 0 pin: re-pin and 3B tree-lane reachability

**Verdict: GATE 0 = GO.** The Metal tree verify fires at 3B on this host, losslessly, with
zero CPU-verify rounds on every probed column.

This document is the campaign's environment pin and the record of what Phase 0 established.
Everything below was measured on the host of record on 2026-07-20; nothing here is inherited
from the RTX 3060 / Windows measurements baked into `generate_run_speculative`.

---

## 1. Environment fingerprint

| Item | Value |
|---|---|
| Host | `tims-mac-mini` — Apple M4 (10-core CPU 4P/6E, 10-core GPU), 16 GiB unified |
| OS | macOS 26.5, `Darwin 25.5.0`, `arm64` |
| Toolchain | `rustc 1.95.0 (59807616e 2026-04-14) (Homebrew)` |
| Repo | clean `main` @ `a8e4dd5c7a94` (`docs(readme): rewrite root README around a new-user journey (#483)`) |
| Worktree | **clean** at build time (`git status --porcelain` empty) |
| Binary version | `camelid v0.3.1-153-ga8e4dd5` — **no `-dirty` suffix** |
| Binary SHA-256 | `ca7f5aa99f8495738822d03888d1cad508dcfd8ca674eb6031873976577fed96` |
| Build | `CARGO_TARGET_DIR=<external-target-dir> cargo build --release --bin camelid` |

The `-dirty` suffix on the prior receipt's binary (`v0.1.7-101-g26c68d9-dirty`) is exactly what
forced this re-pin. This binary is clean.

### Models (moved to internal storage per conductor §2)

| Model | Path | Bytes | SHA-256 |
|---|---|---|---|
| Llama-3.2-3B-Instruct-Q8_0 (target of record) | `~/models/Llama-3.2-3B-Instruct-Q8_0.gguf` | 3421899296 | `b5607b5090a8280063fff2d706bb3408ca6542341b06aab39c3eca0a28575921` |
| Llama-3.2-1B-Instruct-Q8_0 (reachability control) | `~/models/Llama-3.2-1B-Instruct-Q8_0.gguf` | 1321083008 | `432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3` |

Internal disk had 32 GiB free before the copy, 27 GiB after. **Caveat:** the 1B control run in §2
used `spec-verify-parity.sh`'s default external-volume model path (byte-identical file); the 3B
runs in §3 used internal storage. Both SHA-256s above are of the internal copies.

---

## 2. 1B control — the existing gate is still green

`bash qa/speed/spec-verify-parity.sh` run **as-is** (all defaults, informational lane included).

**Result: PASS (exit 0).** Receipt: `qa/speed/receipts/spec-verify-20260720T200134Z-ga8e4dd5.json`.

Linear lane via `serve` — 4 columns, 42 verify rounds total (12 / 8 / 6 / 16), all LOSSLESS,
baseline server fired 0 verify traces as expected. `max_k=6` on every column.

Tree lane via `bench-speculative`:

| column | verdict | gpu_verify_rounds | cpu_verify_rounds | max_fanout |
|---|---|---|---|---|
| repetitive_extraction | LOSSLESS | 8 | 0 | 2 |
| code_completion | LOSSLESS | 22 | 0 | 2 |
| structured_json | LOSSLESS | 18 | 0 | 2 |

**This reproduces the 2026-06-28 receipt exactly** — same 8/22/18 GPU rounds, same 0/0/0 CPU
rounds, same fan-out 2, same 42 linear rounds — on a clean v0.3.1 binary rather than the dirty
v0.1.7 one. The conductor's §0.1 characterisation of that receipt is confirmed in full.

All 14 runs of the informational lane (7 columns × linear/tree) were LOSSLESS via `metal-verify`
with `cpu_verify_rounds = 0`. No CPU ratchet anywhere at 1B.

---

## 3. 3B reachability — GATE 0

Driven directly against `bench-speculative` (the only path that reads `CAMELID_SPEC_TREE`) with the
conductor §2.1 env block verbatim, including `CAMELID_SPEC_CPU_VERIFY=0`. Runner:
`target/barchan-phase0-20260720T195445Z-head-a8e4dd5c7a94/run-3b-reach.sh`.

| column | `lossless` | `first_divergent…` | `gpu_verify_rounds` | `cpu_verify_rounds` | `max_fanout` |
|---|---|---|---|---|---|
| repetitive_extraction | **true** | −1 | **26** | **0** | 2 |
| code_completion | **true** | −1 | **21** | **0** | 2 |
| structured_json | **true** | −1 | **23** | **0** | 2 |

**Gate 0 GO condition — all three columns `lossless == true`, `gpu_verify_rounds > 0`,
`cpu_verify_rounds == 0` — is met.**

Sanity anchor (conductor §5): `plain_tokens_per_second` = 26.93 / 26.17 / 26.55, inside the
26–29 t/s band from `METAL_ROOFLINE.md`. The denominator is sound.

Peak RSS 2.49 GB on a 16 GiB box — ample headroom at 3B.

### 3.1 The 2026-07-07 panic does not reproduce

A prior `bench-speculative` attempt on this host died with:

```
thread 'main' panicked at src/inference/metal_resident.rs:213:69:
range end index 128 out of range for slice of length 0
```

**Not reproduced** — all three 3B columns ran past that point. The mechanism has since been
identified, and the bug is **real and still present on main**; Phase 0 simply cannot reach it.

`rollback_resident_to_position` (`src/inference.rs:2084-2096`) resets the resident engine's
`filled` under `#[cfg(feature = "cuda")]` **only** — there is no Metal branch. On macOS it lowers
`kv_cache.position` while leaving `resident_decode.filled()` stale, which trips
`rebuild = s.filled() != position` (`metal_resident.rs:191-193`), enters the seeding loop at `:211`
(`if position > 0`), and indexes `self.kv_cache.keys[src..src + head_dim]` at `:221` while `keys`
is still zero-length (`KvCache::keys` starts `Vec::new()`, grown only by
`ensure_position_capacity`). The `128` in the message is `head_dim` — it identifies the model
(3B and Qwen3 = 128, 1B = 64), **not** the mechanism.

Reachability: the sole caller is `ModelDrafter::draft` (`src/inference/speculative.rs:199`),
constructed only for `--drafter draft` **without** `--cpu-draft`. The `--drafter ngram` path cannot
reach it — its CPU fallback runs `forward_greedy_verify_chunk` → `ensure_position_capacity`, which
allocates `keys`. **The defect is config-dependent, not model-size-dependent**, and the 1B-vs-3B
framing of the original report was a red herring.

It sits in the `--drafter draft` lane, which conductor §1 puts out of scope, so it is **not fixed
here** (§10.6 — do not move the denominator mid-campaign). Tracked separately.

---

## 4. Conductor amendments

These correct the conductor's situation map. Numbering is referenced from the campaign's
Amendment log.

**A1 — `--workload` is a label only.** Per `--help`: "Workload label recorded in the JSON". It does
**not** select a prompt. The 7 columns live inside `qa/speed/prompts.json` (`camelid.speed-prompts/v1`,
a `columns[]` array of `{id, class, n_gen, spec_friendly, prompt}`), and a caller must materialize
each `prompt` to a file for `--prompt-file`. Column ids match the conductor's list exactly.
`longctx_splitk` carries `n_gen = 96`, not 128 — using `--max-tokens 128` overrides the pack.

**A2 — the tree has never had a 16-node budget.** `--draft-tokens` defaults to
`DEFAULT_NGRAM_DRAFT_TOKENS = 5` (`src/inference/speculative.rs:35`), and the per-round budget is
`full_tree = ((budget + 1).min(TREE_MAX_NODES), budget)` (`src/main.rs:4302`). At the default that
is `max_nodes = 6, max_depth = 5`, confirmed live in the trace: `budget=5 -> Some((6, 5))`. So the
conductor's §0.2 point 4 understates the case — the lane has not merely run "a 2-wide bush against a
16-node cap", it has run against a **6-node budget**. Reaching `TREE_MAX_NODES = 16` requires
`--draft-tokens 15`.

**A3 — the k sweep is viable on the tree path, but only with the T1 kill-switch.** Two different
caps exist: the tree path widens to `TREE_MAX_NODES = 16` (`src/metal.rs:12615`), while the linear
path keeps `MAX_VERIFY_K = 8` (`src/cuda_resident.rs:5524`). `--help` documents only the linear cap
("Capped at MAX_VERIFY_K - 1 = 7"). Above k=7 every *linear fall-through* round returns `Ok(None)`
(`src/inference/metal_resident.rs:343`) — which is precisely the T1 ratchet trigger. Phase 1's
`k ∈ {11, 15}` is therefore reachable **only** because `CAMELID_SPEC_CPU_VERIFY=0` converts those
misses into plain steps. Without it the k>7 cells would silently pin to CPU and yield a
plausible, wrong cost curve.

**A4 — `verify_ms` in the record is NOT verify-only. Phase 1 §4.3 as written is invalid.**
`run.verify_us` is accumulated at **seven** sites in `src/main.rs` (4350, 4373, 4458, 4483, 4509,
4547, 4586), and three of them (4350, 4509, 4547) are *normal-step* sites paired directly with
`run.normal_steps += 1`. The field therefore conflates batched-verify time with plain decode time;
empirically it came to **100.0%** of `spec_decode_ms` on repetitive_extraction, leaving 0.1 ms for
each of 33 normal steps. The conductor's residual formula
(`residual = verify_us − gpu_busy_us`) would attribute normal-step time to verify overhead.
**Phase 1 must either split the accumulator (a `normal_step_us` counter, additive) or subtract
`normal_steps × measured_plain_step_ms` explicitly.**

**A5 — trace tags.** The tree trace is `[metal-tree-verify] base=… n=… emitted_len=… max_fanout=…`
(`src/inference/metal_resident.rs:601`). `[metal-spec-verify]` is the **linear** trace. The
conductor's §2.1 and Phase 1 refer to the linear tag for tree work. There is also a
`[spec-tree] round_seen=… budget=… -> …` latch trace under `CAMELID_SPEC_TREE_TRACE`.

**A6 — `max_tree_fanout` is not in the JSON record.** It exists only in the stderr trace, so any
receipt must parse `max_fanout=` from stderr (as `spec-verify-parity.sh` already does). Conductor §8
lists it as a record field.

**A7 — the `[hw]` probe lies on macOS.** Every run prints
`[hw] GPU: none detected — CPU backend is the inference path` and `RAM 0.0 GiB free / 0.0 GiB total`
**while the Metal tree verify is demonstrably firing in the same run**. It is cosmetic and
CUDA/x86-shaped. This retires the most alarming line in the 2026-07-07 failure log — it was never
evidence that Metal was unavailable.

**A8 — `qa/speed/spec-verify-parity.sh` already implements the 3B probe.** It is macOS-native
(unlike its retired sibling), defaults `BIN`/`MODEL` to this host's paths, and already drives the
tree lane through `bench-speculative` with the conductor's env block. Phase 0 step 3 is a
`MODEL=` override of it; the conductor treats it as 1B-only.

**A9 — T4 resolved by retirement.** See §5.

**A10 — `--model` does not exist on `bench-speculative`.** The GGUF is a bare **positional**
(`src/main.rs`, `Command::BenchSpeculative { model: PathBuf, .. }`). The conductor's §2.1 command
block is correct on this point; noted because sibling harnesses in the repo use `--model` for
`serve` and the two are easy to conflate.

**A11 — set `CAMELID_COMMIT`.** The record's `commit` field is
`std::env::var("CAMELID_COMMIT").unwrap_or("unknown")`. Phase 0's records all read `"unknown"`.
Every measured run from Phase 1 on exports `CAMELID_COMMIT=$(git rev-parse --short HEAD)` so the
record is self-identifying independent of the receipt wrapper.

**A12 — the failed-verify-attempt charge is intentional.** With `CAMELID_SPEC_CPU_VERIFY=0`, a
round whose tree verify declines charges `verify_us` at the fall-through site for the wasted
verify attempt, then charges `normal_step_us` for the plain step that replaces it. That is correct
accounting — the wasted attempt *is* speculation overhead — but it means `verify_ms` on a
high-miss run is not purely successful-verify time. Read it alongside `rounds` and `normal_steps`.

---

## 5. Trap status

| Trap | Status | Notes |
|---|---|---|
| **T1** one-way CPU ratchet | **CONFIRMED verbatim** | `cpu_verify_pinned = true; session.set_resident_paths_disabled(true); tree_drafter.branch = 1` at `src/main.rs:4405-4413`; never reset. Kill-switch `CAMELID_SPEC_CPU_VERIFY` is real (`src/main.rs:4280`, `v != "0"`). Both mitigations applied; all 3B columns returned `cpu_verify_rounds = 0`. |
| **T2** `apply_spec_decode_env` | Not exercised | `bench-speculative` never calls it. Still live for Phase 4. |
| **T3** no serve nocopy default | **CONFIRMED** | `CAMELID_METAL_NOCOPY=1` set explicitly; run log confirms `loading Q8_0 weights as page-aligned wire pages`. |
| **T4** inert harness | **CONFIRMED, resolved** | Retired — see below. |
| **T5** `--drafter` near-inert under the tree | **CONFIRMED** | The tree round hardcodes `SuffixDecodingDrafter` (`src/main.rs:4225`). `--drafter ngram` kept so **no draft model is loaded**. |

### T4 resolution: `qa/speed/tree_verify_check.sh` retired

It drove `bench-generate`, which never reads `CAMELID_SPEC_TREE` — the sole read site is
`src/main.rs:4222`, inside `generate_run_speculative`, reached only from `bench-speculative`. It
therefore compared plain decode against plain decode and reported the ratio as "the speculative
speedup". It also invoked `./target/release/camelid.exe`. It was referenced by no CI job and by one
prose line in `docs/perf-deep-dive/VELOCITY_CAMPAIGN.md`, now corrected. A "fix" would have
duplicated `spec-verify-parity.sh`'s tree lane, which already does the job correctly and gates on
`lossless && gpu_verify_rounds > 0`.

`CAMELID_SPEC_TREE` semantics, for the record: `v != "0" && !v.is_empty()` — any non-empty,
non-`"0"` value is truthy, so both `=1` and the retired script's `=suffix` would have enabled it.

---

## 6. Incidental observation — NOT a result

The Gate-0 records carry economics fields for free. **This is n = 1 per column, single-shot,
cross-invocation, and violates the conductor's own §2 harness discipline. It is not a Phase 2
result and must not be quoted as one.** It is recorded only because it sharpens Phase 1.

| column | `s_sync` | acc/round | net verify ms/round |
|---|---|---|---|
| repetitive_extraction | 0.764 | 3.62 | ~191 |
| code_completion | 0.665 | 2.19 | ~200 |
| structured_json | 0.593 | 1.91 | ~214 |

"Net verify ms/round" nets out normal steps at the measured plain rate per A4; it is a subtraction,
not an instrumented measurement, and Phase 1 exists to replace it.

Against a ~37 ms plain decode step, a **6-row** tree verify costs ~5.2–5.8×. If that survives
instrumentation, the conductor's §0.3 thesis — that verifying up to ~15 rows costs about what
verifying 1 costs — is refuted on this host, and refuted despite acceptance here (3.62 drafts/round
on repetitive) being *better* than the 3060's measured 2.6. That is the conductor's own standing
falsifier firing, and it is the reason Phase 1 is sequenced before Phase 2.

Phase 1 must attribute this before any tuning. The prime suspects, in order:
1. **Resident-engine teardown/rebuild per round.** If `rollback_to_position` drops the resident
   engine, the KV re-seed loop at `metal_resident.rs:212-228` re-copies the whole cache every
   round — a fixed per-round cost matching this shape.
2. `compact_tree_kv_path` (`src/inference.rs:2861`), the conductor's original suspect.
3. The batched verify GEMM itself — the only suspect the §0.3 arithmetic actually predicts, and
   the least likely given the numbers.

---

## 7. Phase 1 entry point

- Artifacts: `target/barchan-phase0-20260720T195445Z-head-a8e4dd5c7a94/`
  (`records/`, `logs/`, `prompts/`, `run-3b-reach.sh`, `prior-spec-verify-latest.json`).
- Verified command line and env block: as in `run-3b-reach.sh`, which supersedes conductor §2.1
  (adds `CAMELID_NO_OPEN=1`, corrects the trace tag per A5).
- **Do first:** fix the `verify_us` conflation (A4). Every Phase 1 number depends on it.
- Then add `gpu_busy_us` / `kernel_window_us` to `verify_batch_tree` behind the existing
  `CAMELID_SPEC_VERIFY_TRACE`, additive and trace-gated only.
- Sweep `--draft-tokens ∈ {1,3,5,7,11,15}` → tree budgets `max_nodes ∈ {2,4,6,8,12,16}`, keeping
  `CAMELID_SPEC_CPU_VERIFY=0` (A3), interleaved, N ≥ 5, `--warmup`.
