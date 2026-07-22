# W3 parity matrix — CAMELID_CUDA_HOSTREG on the RTX 3060 Laptop 6 GB (2026-07-22, branch flint-w2-hostreg @ c50aa93)

**VERDICT: token-identical across the entire matrix. The hostreg lane is parity-clean on the discrete
reference card; throughput is PCIe-bound as designed (the slowdown is itself the engagement proof).**

| Leg | Model | Mode | HOSTREG 0 vs 1 | tok/s 0 → 1 | zero-copy/uploaded |
|---|---|---|---|---|---|
| A | TinyLlama-1.1B Q8_0 | plain greedy, 64 tok | IDENTICAL | 130.6 → 5.3 | 155 / 0 |
| B | Qwen3-0.6B Q8_0 | plain greedy, 64 tok | IDENTICAL | 155.5 → 10.9 | 197 / 0 |
| C | Qwen3-1.7B Q8_0 | plain greedy, 64 tok | IDENTICAL | 84.6 → 3.1 | 197 / 0 |
| D | TinyLlama Q8_0 | prefill-batched 0/1 × hostreg 0/1, 1082-tok prompt + 32 tok | 4-way IDENTICAL | — | 155 / 0 |
| E | TinyLlama Q8_0 | CAMELID_CUDA_GRAPHS=1 × hostreg=1, 64 tok | IDENTICAL to non-graphs legs | — | 155 / 0 |
| F | synthetic engines | 42 ignored GPU unit tests (gemv bit-exact, verify_batch, prefill+decode, tree-verify lossless) under HOSTREG=1 | 42/42 ok (4.27 s) | — | — |
| G1 | Llama-3.2-1B Q8_0 | spec_draft_rollback under HOSTREG=1 | ok, resident_steps=5 cpu_steps=0 (10.3 s vs 3.6 s baseline) | — | 113 / 0 |
| G2 | Llama-3.2-1B Q8_0 | resident_kv_cpu_fallback under HOSTREG=1, --test-threads=1 | 2/2 ok (31.6 s), incl. the forced CPU verify-chunk over file-backed weights | — | 113 / 0 |

Counts sanity: TinyLlama 22×7+1=155, qwen3 rows 28×7+1=197, Llama-1B 16×7+1=113 — every Q8_0 weight
tensor took the registered path in every leg; zero per-tensor fallbacks observed on this driver
(576.83, attrs: host_register_supported=1 can_use_host_pointer=0 unified_addressing=1 dptr_eq_host=false).

Open-question outcomes:
- Q1 registration under WDDM: works (all legs). Q5 graphs over mapped pointers: capture+replay
  token-identical (leg E). Q9 batched GEMM over registered SoA: 4-way identical (leg D).
- Q10 CPU-fallback cost with blocks absent: the forced CPU verify-chunk leg (G2) works and stays
  parity-clean; the weight pass streams via page cache (bounded — 31.6 s serial suite vs 9.2 s
  baseline). No silent hang, no cliff beyond the expected streaming cost.

Anomalies, documented and exonerated:
1. G2 run with default test parallelism fails 1/2 under hostreg ("resident lane declined at
   generated token 0"). PRE-EXISTING test-harness artifact, not a hostreg regression: the two tests
   build resident engines for the SAME model concurrently and contend for the process-global
   resident slot — the flag-off BASELINE run shows the same collision as a vacuous
   "SKIP … resident lane unavailable" pass in the sibling test (legG2-kv-fallback-baseline.txt).
   Hostreg's slower engine build only shifts which test loses the race and when. Serial run: green.
   (Candidate follow-up outside this PR: serialize these two tests or give the loser a clean skip
   mid-run instead of a panic.)
2. One unreproduced >10-min hang of spec_draft_rollback under hostreg immediately after the 42-test
   leg F churn (rapid register/unregister cycles); killed at the 10-min timeout, process tree swept
   clean, never reproduced (10.3 s on rerun, and again green in later runs). Recorded here for
   honesty; watch for it on the Spark.
