# GABBRO — NVFP4 gemma4-E4B promoted to a Supported row on macOS

**Date:** 2026-07-18 · **Host:** Apple M4 (16 GB) · **Engine:** current `main` (release)
**Artifact:** `gemma-4-E4B-it-NVFP4-mm.gguf`, sha256 `eb293344972e2b292a043b8e7649b9788dca915b034e5c2721cfc531cf9863d9`, 6,058,607,776 B (byte-exact to the BASALT-sanctioned artifact; reproduced locally from the Q8_0 parent via the pinned `llama-quantize` acd79d603 — NVFP4's RTN quantizer is ISA-deterministic).

This bundle records the evidence promoting `gemma4_e4b_it_nvfp4` from *engine-facts / planned* to
**`supported_exact_row_smoke`**, support scope **`exact_row_gpu_resident_raw_decode_parity_smoke_only`**.

## What changed since the frozen G3 NO-GO

The frozen Gate G3 (BASALT, engine `8038abba`, x86) recorded **NO-GO**: teacher-forced top-1
agreement vs the Q8_0 parent was NVFP4-mm **88.5%** (262/296) vs Q4K-mm **92.6%** (274/296),
gap 4.1 pp > the pre-registered 2.0 pp tolerance. **That result stands as history and is not
altered here.** This promotion rests on a NEW, disclosed measurement on the CURRENT engine.

## Primary metric — teacher-forced top-1 agreement (basalt_eval_protocol.md §5.1), current engine

Harness: `camelid gemma4-eval-pack` (load-once form of the §5.1 forced-decode harness; CPU path,
no engine-math change). Baseline = Q8_0 greedy over the committed `basic_v1` + `deep_v1` packs
(296 continuation positions). Validation: Q8_0 self-agreement **296/296 = 100.0%**; the Q8_0
baseline is token-identical to BASALT's committed `baseline_continuations.json` for **8/9 prompts**
(only `village-story` diverges, and only in its tail at position 38 — a known f32 near-tie).

| Row | matches / 296 | agreement | note |
|---|---:|---:|---|
| Q8_0 (baseline) | 296/296 | 100.0% | reference |
| **NVFP4-mm** | **268/296** | **90.5%** | +6 vs frozen 88.5% (same byte-exact file → current-engine decode-fidelity gain) |
| Q4K-mm (comparator) | 272/296 | 91.9% | ARM-quantized (sha `23aad5d0…`); Q4_K's float-search quantizer is ISA-sensitive |
| hybrid ffn_down→Q8_0 | 272/296 | 91.9% | robustly ties Q4_K; 6.16 GiB |

**GO rule `agreement(NVFP4) ≥ agreement(Q4K) − 2.0`:** gap **1.4 pp ≤ 2.0 → GO.**

### Honest characterization (binding, per DECISIONS D17 / Option-B claim-lint)

This is a **near-tie**, not a quality win. The pure-NVFP4 GO is **comparator-sensitive**: measured
against a Q4_K comparator at the frozen 92.6% level the gap is 2.1 pp (marginal NO-GO). No surface
may state NVFP4 is *better than*, or unqualifiedly *quality-competitive with*, Q4_K. The honest
claim is: **in the current engine NVFP4 is within the pre-registered tolerance of the format-isolated
Q4_K comparator produced the same way** — it is no longer *clearly behind*. It remains a **space/speed**
quant; the best-quality practical row is Q4_K_M-im (95.3%, report-only), not NVFP4.

### Why the gap closed (attributed, not hand-waved)

- NVFP4 +6 matches (88.5→90.5): the current engine decodes the **byte-identical** NVFP4 file more
  faithfully than the frozen engine (baseline near-identical, so this is engine, not baseline drift).
- Q4K −2 matches (92.6→91.9): my Q4K comparator is ARM-quantized (`23aad5d0…` ≠ BASALT x86 `d306fa77…`).
  NVFP4 matched byte-exact across ISA; Q4_K did not — expected, its quantizer does a float search.
- **imatrix is not a lever for NVFP4** (proven: `ggml-quants.c:2237 GGML_UNUSED(quant_weights)`).

## Decode-parity (oracle) — the support anchor

- **BASALT Leg B:** cross-engine token parity Camelid vs pinned `llama-completion`, same byte-exact
  NVFP4 file, 8/9 exact + 1 attributed 0.084-logit near-tie.
- **G-M1:** NVFP4 CPU wire-lane decode bit-exact on Apple Silicon/ARM (13/13 `nvfp4_*`).
- **Metal self-parity:** `metal_gemma4_resident_nvfp4_forward_matches_cpu` (GPU forward == CPU oracle).
- **Fresh Mac spot-check (this bundle):** llama.cpp `acd79d603` and Camelid both decode
  "The capital of France is" → "Paris" on the NVFP4-mm file.

## Performance & end-to-end (macOS Metal resident lane)

First end-to-end real-artifact run: `camelid gemma4-generate-gpu` loads NVFP4-mm resident and
returns coherent output ("Paris."). Isolated 128-token greedy decode, Apple M4:

| Model | decode tok/s | file |
|---|---:|---:|
| **NVFP4-mm** | **12.12** | 5.64 GiB |
| Q8_0 parent | 8.34 | 7.63 GiB |
| Q4_0 | 4.14 | 4.80 GiB |

NVFP4 is the fastest of the three and **1.45× faster than its Q8_0 parent** at 26% smaller.
(Load time is T7-read-bound at ~39 MB/s — a disk artifact, excluded from the decode figures.)

## Scope of the support row

`supported_exact_row_smoke`, `exact_row_gpu_resident_raw_decode_parity_smoke_only` — the exact
`gemma-4-E4B-it-NVFP4-mm.gguf` row only. NOT implied: quality-competitiveness beyond the tolerance
above, other architectures/quants, bounded-context packs (not run for NVFP4), production throughput,
arbitrary templates, or full support. gemma4-E4B pilot only.
