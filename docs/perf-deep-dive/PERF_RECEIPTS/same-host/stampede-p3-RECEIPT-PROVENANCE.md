# STAMPEDE Phase 3 receipt provenance (2026-07-08)

The `stampede-p3-*.json` medN receipts in this directory were minted BEFORE
`cpu-baseline-medN.mjs` learned to record `flags_env` (added in the same Phase-3 commit),
so their flag provenance is documented here instead of inside the JSON:

| receipt | camelid build | flag state |
|---|---|---|
| `stampede-p3-kquant-owner-{off,on}-llama3b-q4km-…` | 582781da + Lane B v1 (working tree) | off: `CAMELID_X86_KQUANT_MATMUL_OWNER` unset; on: `=1` |
| `stampede-p3-kquant-owner-v2-{off,on}-llama3b-q4km-…` | 582781da + Lane B v2 (working tree = commit af9427ab's kernel) | same |
| `stampede-p3-kquant-owner-v2-on-qwen3-4b-q4km-…` | same v2 build | `CAMELID_X86_KQUANT_MATMUL_OWNER=1` |
| `stampede-p3-q8-owner-default-on-llama3b-q8-…` | same v2 build (D15 flip compiled in) | Q8 owner: default (= All on win-x86_64); kquant unset |

All legs: `CUDA_VISIBLE_DEVICES=-1`, llama.cpp pin b9918 (0512ef1e5), REPEATS=5,
`camelid_head` field records the dirty-tree tag used at mint time. Engagement evidence for
the kquant owner lives in the flag-ON legs' prefill deltas (+50%/+66%) plus the unit-level
engaged-check added to the bitwise twin test; the Q8 owner sweep receipts
(`q8-prefill-owner-b9918-revalidation-20260708/`) carry per-record `owner_prefill_taken`
counts (off=0, owner=280). The decode-dip artifact in kquant-ON legs is analyzed in
`stampede-p3-kquant-decode-only-probe-20260708.md`.

Known limitation (review finding, accepted): the sweep's engaged-check is variant-blind —
it proves the OWNER fired, not which microkernel variant; this host has AVX-512 VNNI so
the vnni/4x8 configs are genuine here, but a non-VNNI host would silently measure the AVX2
kernel in those configs. The sweep also hard-aborts (fail-loud, intended) on models with
no Q8_0 linears or when a CUDA-resident plan disables the repack.
