# cron-95495a91 AVX2 register-sum same-host check

## Scope

Current-`main` Ubuntu/Linux x86_64 Llama 3.2 3B Instruct Q8_0 recheck for the default-off AVX2 baseline lane:

- `CAMELID_X86_Q8_REPACK=on`
- `CAMELID_X86_Q8_KERNEL=avx2`

This compares current `main` commit `e27a4e1` against pre-change baseline `214f733` on the same host, same model row, same harness family, and the same default-off AVX2 lane.

No support, default-on, portability, or broad throughput promotion is claimed.

## Validation

The run stayed on Linux x86_64 and used the shared lane target with disk-guard hygiene intact.

One-token exact-row parity check for `hello` passed on both commits:

- prompt tokens matched the llama.cpp reference
- generated token ids matched the llama.cpp reference
- generated text matched the llama.cpp reference

The first repeated 1-token timing attempt was discarded as a harness-shape artifact because later measured requests returned empty Camelid output. The retained timing comparison below uses unique prompts plus marker enforcement so every measured run produces bounded non-empty output.

## Measured result

Retained same-host timing slice:

- request shape: unique-prompt marker run, `max_tokens=8`, `warmup=0`, `repeats=2`, `threads=16`
- Camelid guardrail: passed
- llama.cpp guardrail: passed

Measured means:

| Commit | Camelid TTFT ms | Camelid total ms | llama.cpp TTFT ms | llama.cpp total ms |
| --- | ---: | ---: | ---: | ---: |
| `e27a4e1` current main | 8837.36 | 8837.65 | 309.33 | 498.45 |
| `214f733` baseline | 8823.09 | 8823.37 | 310.98 | 500.23 |

Delta versus the retained baseline:

- Camelid TTFT: `+14.27 ms` slower on current main
- Camelid total elapsed: `+14.28 ms` slower on current main
- llama.cpp reference stayed effectively flat across the paired run

## Retain/reject decision

Reject any performance retention or promotion for the AVX2 register-sum follow-on from this slice.

The change preserved exact-row one-token parity, but the cleaned same-host marker run did not beat the `214f733` baseline on Camelid TTFT or total elapsed. Keep the implementation merged as a narrow low-level cleanup only; do not treat it as a measured Ubuntu x86_64 speed win.
