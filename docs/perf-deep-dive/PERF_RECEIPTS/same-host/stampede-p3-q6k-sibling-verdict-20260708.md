# STAMPEDE P3 Lane B — Q6_K owner sibling: verdict (REVISED after the reachability fix)

## What was built
`q6_k_owner_prefill_tiled` (same flag `CAMELID_X86_KQUANT_MATMUL_OWNER`, same rows≥4 dispatch,
same 2D blocking as the Q4_K owner): hoists the per-cell 256-value 6-bit weight rebuild per
weight row (the default per-cell path re-runs the full scalar rebuild for EVERY cell) and uses
the in-tree bit-identity-proven AVX2 `aux32` lane kernel; the bits-sensitive per-cell f32 shape
(8 lane accumulators, `sums[l] += d·aux32[l]` per superblock, final left-fold) is kept verbatim
per `q6_k_wire_row_dot` — the shape KQUANT_RECON proved must not be restructured. Bitwise twin
tests (owner-vs-block-dot with engaged-check; scalar-vs-AVX2 aux32; full-byte-coverage wire with
a forced −128 scale per superblock).

## The reachability bug the adversarial review caught (MAJOR)
As first committed, the owner dispatch lived only in `q6_k_block_dot_core` — but production
Q6_K traffic calls `matmul_rhs_transposed_q6_k_block_dot`, which carried its own inline
duplicate of the core body and never reached the core. **The owner was production-unreachable
for Q6_K**, which is why the first end-to-end receipts
(`stampede-p3-kquant-owner-v3q6k-on-*.json`) were flat (0.99×/0.95× vs the Q4_K-only owner) —
they measured the unreachable version, NOT a small Q6_K share (this file's earlier "~4% of the
stream" explanation was wrong and is retracted). Fix: the wrapper now delegates to the core
(the Q4_K wrapper's pattern), de-duplicating the body and routing every Q6_K block-dot consumer
— the main linear intercept, the tied lm-head path, and the borrowed dispatch — through the
owner dispatch.

## Measured after the fix (single-engine probe, bench-memory-safety rules)
`scripts/` single-engine protocol: ONE camelid serve (no concurrent llama-server), CUDA hidden,
3B Q4_K_M, 4 cold prefills per leg (nonce cache-defeat), off → on → off drift check, engaged
counter from response telemetry:

| leg | cold prefill tok/s (4 probes) | median | kquant_owner_prefill_taken |
|---|---|---|---|
| off | 13.62, 13.68, 13.81, 13.82 | 13.74 | 0 |
| on (Q4_K+Q6_K owner) | 23.76, 23.77, 23.81, 23.89 | **23.79** | **392** |
| off (drift check) | 13.97, 14.04, 14.22, 14.39 | 14.13 | 0 |

**Speedup 1.73× (drift 1.028×)** — vs the Q4_K-only owner's ~1.50× on the dual-engine harness.
Byte-identity: bitwise twin tests over both quants; greedy text identity was verified e2e on the
earlier legs and the kernels are unchanged since (only routing moved).

Note: single-engine absolute numbers differ from the dual-engine medN receipts (no concurrent
llama-server contention); ratios are the honest comparison. Dual-engine measurement is now
restricted per the bench-memory-safety rules (this box crashed on memory pressure 2026-07-08).

## Standing discovery (unchanged)
A locally-requantized pure-Q6_K GGUF routes to the EXPERIMENTAL lane via serve — native K-quant
kernels never run for it (owner counter 0, lane:"experimental"). Local requants cannot validate
native-lane kernels; use supported rows.
(`stampede-p3-q6k-owner-{off,on}-tinyllama-q6k-20260708.json` kept as the honest negative.)

## Verdict
GO as opt-in: combined K-quant owner now delivers **1.73× on 3B Q4_K_M prefill** (0.15× → ~0.26×
of llama.cpp b9918), byte-identical. Remaining escalation to the campaign's 0.6× target:
8-row interleaved repack, AVX-512 VNNI main side.
