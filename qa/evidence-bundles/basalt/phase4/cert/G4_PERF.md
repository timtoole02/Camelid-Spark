# BASALT Gate G4 — Phase 4 CERT + perf table (measured, this box)

Status: **CERT PASS. Perf: a surprise — NVFP4-mm CUDA is CORRECT but currently SLOWER than
the Q8_0 lane it was meant to beat.**

- Engine: branch `basalt/phase4-cuda-decode`, HEAD `892672ca`, `cargo build --release --features cuda`.
- Hardware (every figure carries this row): **RTX 3060 Laptop GPU, sm_86, driver 576.83,
  CUDA 12.9, Windows 11.** Theoretical DRAM bandwidth used: **336.0 GB/s** (6 GB variant:
  192-bit GDDR6 @ 14 Gbps = 14 × 192 / 8; NVIDIA published spec).
- Pilot model: `gemma-4-E4B-it-NVFP4-mm.gguf` sha256 `eb293344…9863d9` (matmuls NVFP4,
  head/PLE kept Q8_0/f32); baseline `gemma-4-E4B-it-Q8_0.gguf` sha256 `a2232a64…`.
- Bundle id: `qa/evidence-bundles/basalt/phase4/cert`.

All numbers measured on this box, never general.

---

## 1. CERT — end-to-end self-parity (correctness gate, banked before perf)

NVFP4-mm greedy, **Camelid CPU wire lane vs Camelid CUDA-resident lane** (`gemma4-generate`
vs `gemma4-cuda-generate`), gemma4 lane-native pack (`basic_v1` 5 + `deep_v1` 4 = 9 prompts,
320 continuation positions). Target: token-identical (the CUDA lane's greedy-token contract).

| result | value |
|---|---|
| Fully token-identical prompts | **6 / 9** |
| Prompts with a divergence | 3 / 9 |
| All divergences near-tie + attributed | **YES** |

The three divergences are textbook **near-tie argmax flips**, every one attributable to the
known, accepted cross-lane difference — **CUDA f16 KV cache vs CPU f32 KV cache** (recon E7),
the same contract the shipping Q8_0 row runs under:

| prompt | step | CPU tok | CUDA tok | CPU top-2 gap (raw logits) | CUDA tok's rank in CPU dist |
|---|---:|---:|---:|---:|---|
| primary-colors | 12 | 659 | 7913 | 0.111 | **#2** |
| village-story | 44 | 5597 | 186743 | 0.070 | **#2** |
| pancake-recipe | 15 | 236770 | 236778 | 0.047 | **#2** |

In **every** case the CUDA token is exactly the CPU's **#2** candidate, at a raw-logit gap
of 0.047–0.111 — the same band as the Phase-3 cross-engine flip (0.084, village-story). No
non-near-tie divergence exists (no large-gap flip, no out-of-top-32 token), so there is **no
wiring or kernel bug**. CERT passes under the greedy-token contract. Kernel-level bit-parity
was already sealed at implementation time (`nvfp4_gemv_matches_oracle: 46/46 rows
bit-identical, worst rel diff 0.000e0`).

Receipt: `parity_cert.json` (per-prompt cpu_ids, cuda_ids, identical flag, and the probed
divergence records with CPU top-2 logit gaps).

---

## 2. G4 perf table

Method: median of **5 warm runs** (run 1 discarded = kernel compile + mmap fault-in), fixed
prompt `"The water cycle works as follows:"`, **128 greedy tokens**, decode tok/s. Achieved
GB/s = (per-token GPU weight read bytes, from **actual GGUF tensor sizes**) × decode tok/s.
Single engine process on the GPU at all times; VRAM verified freed to 0 after every load.

| # | row | lane | decode tok/s (median) | peak VRAM | per-token read | achieved GB/s | % of 336 GB/s |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | **Q8_0 CUDA** (baseline) | Camelid resident | **25.80** | 5559 MiB | 5.182 GB | 133.7 | **39.8 %** |
| 2 | **NVFP4-mm CUDA** (campaign) | Camelid resident (new `nvfp4_gemv`) | **14.64** | 3479 MiB | 3.048 GB | 44.6 | **13.3 %** |
| 3 | **NVFP4-mm CPU** (reference) | Camelid CPU wire | **1.57** | — | 3.048 GB | — (host DDR) | — |
| 4 | pin GPU (NVFP4-mm) | llama-completion `-ngl 99` | **skipped** | — | — | — | — |

Run-to-run stability was tight: NVFP4 CUDA 14.6–14.8 tok/s across all 5; Q8_0 CUDA
25.8 tok/s across all 5; CPU 1.56–1.58 tok/s. Decode-only steady (2nd-half) figures:
Q8_0 25.44, NVFP4 14.57.

**Row 4 (pin GPU) skipped — documented, memory-safety.** A full `-ngl 99` offload of the
6.06 GB NVFP4-mm model on this 6144 MiB card is not comfortable headroom: the pin does not
use Camelid's file-backed-embedding trick, so it would attempt ~6 GB of resident weights
(matmuls + `token_embd` + `per_layer_token_embd`) plus KV/context → OOM / WDDM-hang risk.
The brief authorizes this optional 4th GPU path only "if headroom is comfortable; else
skip-with-note." The Camelid NVFP4-vs-Q8_0 GPU comparison stands on its own.

---

## 3. The headline finding (honest framing — no threshold-shopping)

**The byte reduction is real; the speed win is not — yet.**

- **Per-token weight read shrinks 1.70×** (Q8_0 5.182 GB → NVFP4 3.048 GB), measured from
  actual GGUF tensor sizes. The **matmul-only** shrink is exactly **1.889×** (the 294
  NVFP4 matmuls: 4.192 GB → 2.219 GB). The pre-registered expectation (recon §3.5) was
  ~1.6× because the Q8_0 head and PLE stay resident and read every token; the
  format-isolated figure (matmuls-only differ) is **1.647×**, dead-on. The measured 1.70×
  is a touch higher only because the two GGUFs also differ incidentally in `inp_gate`/`proj`
  precision (F32 in the Q8_0 file vs Q8_0 in the -mm file — a quantize-tool artifact, not
  part of the NVFP4 story).

- **But NVFP4-mm CUDA decodes at 0.57× the speed of Q8_0** (14.64 vs 25.80 tok/s) despite
  reading fewer bytes. The tell is the roofline column: Q8_0 achieves **39.8 %** of the
  336 GB/s DRAM peak (a healthy memory-bound GEMV), while NVFP4 achieves only **13.3 %**.
  NVFP4 is nowhere near the memory roofline → the **v1 scalar-LUT dequant kernel is
  COMPUTE-bound**, not bandwidth-bound. The nibble-unpack + per-element codebook lookup +
  per-sub-block UE4M3 float decode costs more than the bandwidth it saves.

- This is exactly the receipt recon **P4_KERNEL_RECON §5 Q1** pre-registered: the
  `__byte_perm` + `__dp4a` inner-loop upgrade (the pin's `get_int_from_table_16` trick) was
  deferred to a "measured, parity-neutral follow-up **only if** the first perf receipt shows
  the kernel clearly below the memory roofline." **It does (13.3 % vs 39.8 %). The dp4a
  upgrade is now warranted** — and it is parity-neutral (identical i32 result), so it cannot
  disturb the CERT that just passed. The gpu_head lever (§5 Q4) is the other outstanding
  per-token-read reducer (~0.71 GB Q8_0 head ≈ 23 % of the NVFP4 read set).

**Bottom line for G4:** Phase 4 lands the NVFP4 CUDA-resident lane **correct** (CERT pass,
kernel bit-exact, VRAM-comfortable at 3.48 GB peak vs Q8_0's 5.56 GB) but **not yet fast** —
the v1 scalar kernel leaves its 1.70× bandwidth advantage on the table. NVFP4's win on this
box today is **VRAM headroom** (2.08 GB more free), not decode speed.

---

## 4. Provenance

- Commands + resource log: `commands_log.txt`, `resource_log.csv` (host paths sanitized).
- Byte accounting: `byte_accounting.json` (per-tensor read-set breakdown, both formats).
- Raw per-run timing: `perf_table.json` (all 5 kept runs per row, load + decode + steady).
- CERT: `parity_cert.json`.
- Determinism: warm-run spread ≤ 1.4 % on every row; peak VRAM identical across all 5 runs
  of each GPU row (NVFP4 3479 MiB, Q8_0 5559 MiB).
