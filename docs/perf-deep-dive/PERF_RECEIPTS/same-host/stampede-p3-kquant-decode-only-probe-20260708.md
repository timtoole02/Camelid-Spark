# STAMPEDE P3 Lane B — decode-only probe: the medN decode dip is NOT a decode-path effect

Context: the medN A/B receipts for the Q4_K prefill owner
(`stampede-p3-kquant-owner-v2-{off,on}-llama3b-q4km-20260708.json`) show camelid decode
lower in the flag-ON leg (14.76 → 11.47). The owner dispatches only at rows ≥ 4, so a
decode (rows==1) effect would falsify the design. This probe isolates decode.

## Method
`scripts/kquant-decode-only-probe.mjs`: single camelid serve (no concurrent llama-server),
CUDA hidden, load Llama-3.2-3B Q4_K_M, warm the decode prompt's prefix cache, then measure
4 × 64-token greedy decode windows — **no cold-prefill probes between decode measurements**
(the medN harness interleaves a 516-token cold prefill before every decode window).
Sequence run: flag=0 → flag=1 → flag=0 (the repeat-0 leg detects monotonic drift).

## Result (camelid 582781da + Phase-3 working tree, 2026-07-08)

| leg | decode tok/s (4 windows) |
|---|---|
| flag=0 (first) | 9.35, 9.44, 9.30, 9.28 |
| flag=1 | 9.12, 9.07, 9.17, 9.12 |
| flag=0 (again) | 9.05, 9.06, 9.02, 8.99 |

Monotonic decline INDEPENDENT of the flag (the second flag=0 leg is the slowest) —
thermal drift, no flag-correlated decode effect. The medN dip is prefill-coupled
measurement state (the owner's faster/denser prefill immediately precedes each decode
window in that harness), not a decode-path change. Greedy text in the medN receipts is
byte-identical OFF↔ON, consistent with rows==1 never entering the owner.

Caveat kept honest: prefill→decode thermal coupling is a real serving phenomenon on this
thermally-limited laptop, but it is a property of running prefill faster, not of the
kernel's decode behavior; net wall time for prefill+64-token decode improves (~8s saved
prefill vs ~0.9s slower decode on the 3B Q4_K_M row).
