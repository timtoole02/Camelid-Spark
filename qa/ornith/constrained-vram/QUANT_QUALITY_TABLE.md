# Ornith-1.0-9B quant quality × residency table (Item 4)

Conductor: ORNITH_9B_CONSTRAINED_VRAM_CONDUCTOR.md Item 4. All new quants produced at
REF_QWEN35 (`acd79d6`) from the hash-verified bf16 with
`imatrix_ornith_agentic.gguf` (imatrix over the frozen 20-trace agentic-coding
corpus TRACES_agentic_20.txt, 5 chunks @ c=2048; computed on the Q8_0 host —
bf16 17.9GB exceeds the 15.7GB host RAM and would page-thrash; documented
deviation, Q8_0 is the parity-certified quant).

PPL = llama-perplexity on `heldout_coding.txt` (165 KB repo Rust + llama.cpp C++,
disjoint from calibration), c=2048. Residency = llama-server `-c 16384 -ngl 99`,
peak nvidia-smi memory.used across load + one 64-token greedy completion
(full decode state; llama.cpp's hybrid-arch KV allocates only the 8
full-attention layers, same shape as Camelid's sparse KV). Card total 6144 MiB.

## Quality × size × 16K residency

| quant | bytes | PPL (held-out coding) | vs Q6_K | peak VRAM @16K | headroom | ≥512 MiB bar | provenance |
|---|---|---|---|---|---|---|---|
| Q6_K | 7,359,259,072 | 2.3636 ±0.0303 | — | n/a (7.4GB > card) | — | — | HF pristine (Item 6 verifier) |
| IQ4_XS | 5,196,440,096 | 2.3900 ±0.0309 | +1.12% | 5,381 MiB | 763 MiB | PASS (thin) | bf16 + imatrix |
| Q4_K_M (home requant) | 5,629,108,416 | 2.4112 ±0.0311 | +2.01% | not run @16K (8K proven: 5,259 MiB peak, ~700 MiB margin) | — | expected FAIL @16K (weights alone 5,368 MiB) | Q8_0→Q4 requant, NO imatrix |
| **Q3_K_M** | 4,623,524,384 | 2.4693 ±0.0328 | +4.47% | **4,935 MiB** | **1,209 MiB** | **PASS** | bf16 + imatrix |
| IQ3_XXS | 3,938,165,280 | 2.5323 ±0.0332 | +7.14% | 4,281 MiB | 1,863 MiB | PASS | bf16 + imatrix |

Kill criterion status: **NOT fired** — multiple ≤Q3-class (and one 4-bit-class)
quants achieve full residency at 16K with ≥512 MiB headroom.

## Decision: STOCK Q3_K_M is the Item 4/5/6 lane quant

(A custom "CAM" remix was attempted and abandoned: llama.cpp's Q3_K_M recipe
allocates its q5_K extra-precision budget by INSTANCE COUNT, so overriding the
four q5_K tensors just shifts q5_K onto the next four — override whack-a-mole.
Instead Camelid handles q5_K at LOAD: `q5k_to_q8_0_blocks` re-encodes the four
q5_K tensors to 36-byte Q8_0 blocks — a strict precision upcast, ~+40 MiB over
the measured file — and runs them on the existing Q8_0 GEMV lane. The measured
PPL/residency of stock Q3_K_M therefore applies to the exact shipped file.)

1. **Zero new CUDA kernels.** q3_K/q4_K/q6_K GEMV already shipped (merged
   K-quant lanes); q5_K rides the Q8_0 lane via the load-time upcast —
   enablement is a builder mapping + runnable dequant/admission (commit
   9736d794 + follow-up), not a kernel campaign. IQ4_XS/IQ3_XXS would each
   require a new IQ codebook kernel family AND GGUF-reader support (Camelid
   cannot currently parse IQ files at all).
2. **Headroom hosts the Item 6 pipeline.** ~1,170 MiB (measured 1,209 minus the
   ~40 MiB upcast) fits the draft-side scratch, the k-deep DeltaNet state ring
   (~6 MiB/checkpoint), and growth margin; IQ4_XS's 763 MiB is workable for
   plain serving but thin for the speculative lane that is this conductor's
   point.
3. **Draft quality only matters via acceptance rate** (Item 5 measures it
   directly against the Q6_K/Q8_0 verifier); +4.5% held-out PPL is acceptable
   for a draft, and IQ3_XXS's +7.1% needlessly risks the ≥80% acceptance bar.

**Recorded follow-up:** IQ4_XS is the quality-per-byte champion (beats the home
Q4_K_M by 0.9% PPL while 412 MiB smaller). If an IQ kernel family ever lands in
Camelid, IQ4_XS becomes the preferred plain-serving quant at 16K on 6GB.
Q4_K_M (home) remains the proven 8K full-residency row (Supported, merged).
