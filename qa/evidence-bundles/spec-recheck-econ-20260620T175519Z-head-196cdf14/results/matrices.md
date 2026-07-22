
## ngram — 24 cells | lossless: ALL ✓

### S_sync (spec t/s ÷ plain t/s)

| workload | γ=2 | γ=4 | γ=6 | γ=7 |
|---|---|---|---|---|
| code | 1.25× | 1.22× | 1.17× | 1.12× |
| json | 1.24× | 1.26× | 1.19× | 1.10× |
| extraction | 1.00× ✗ | 1.02× | 0.93× ✗ | 0.90× ✗ |
| chat | 0.92× ✗ | 1.00× ✗ | 0.97× ✗ | 0.98× ✗ |
| creative | 0.99× ✗ | 1.00× | 0.92× ✗ | 0.98× ✗ |
| adversarial | 1.03× | 1.01× | 0.96× ✗ | 0.93× ✗ |

### accept rate (accepted drafts ÷ drafted)

| workload | γ=2 | γ=4 | γ=6 | γ=7 |
|---|---|---|---|---|
| code | 84.2% | 66.1% | 54.2% | 46.4% |
| json | 68.8% | 65.5% | 53.7% | 46.0% |
| extraction | 46.2% | 31.8% | 25.0% | 21.4% |
| chat | 40.0% | 50.0% | 33.3% | 28.6% |
| creative | 0.0% | 0.0% | 0.0% | 0.0% |
| adversarial | 53.8% | 42.5% | 33.3% | 28.6% |

### detail at best γ per workload (by S_sync)

| workload | best γ | accept% | tok/round | f_draft | plain t/s | spec t/s | S_sync | gpu/cpu verify | lossless |
|---|---|---|---|---|---|---|---|---|---|
| code | 2 | 84.2% | 2.68 | 0.0001 | 40.29 | 50.19 | 1.25× | 19/0 | ✓ |
| json | 4 | 65.5% | 3.62 | 0.0000 | 37.68 | 47.53 | 1.26× | 21/0 | ✓ |
| extraction | 4 | 31.8% | 2.27 | 0.0001 | 31.22 | 31.90 | 1.02× | 11/0 | ✓ |
| chat | 4 | 50.0% | 3.00 | 0.0000 | 32.14 | 32.07 | 1.00× | 3/0 | ✓ |
| creative | 4 | 0.0% | 1.00 | 0.0000 | 31.50 | 31.56 | 1.00× | 1/0 | ✓ |
| adversarial | 2 | 53.8% | 2.08 | 0.0000 | 31.67 | 32.59 | 1.03× | 13/0 | ✓ |

## draft-gpu — 12 cells | lossless: ALL ✓

### S_sync (spec t/s ÷ plain t/s)

| workload | γ=2 | γ=4 | γ=6 | γ=7 |
|---|---|---|---|---|
| code | 0.11× ✗ | 0.11× ✗ | — | — |
| json | 0.10× ✗ | 0.10× ✗ | — | — |
| extraction | 0.08× ✗ | 0.08× ✗ | — | — |
| chat | 0.09× ✗ | 0.07× ✗ | — | — |
| creative | 0.10× ✗ | 0.07× ✗ | — | — |
| adversarial | 0.10× ✗ | 0.08× ✗ | — | — |

### accept rate (accepted drafts ÷ drafted)

| workload | γ=2 | γ=4 | γ=6 | γ=7 |
|---|---|---|---|---|
| code | 92.1% | 82.9% | — | — |
| json | 89.0% | 84.5% | — | — |
| extraction | 75.2% | 72.3% | — | — |
| chat | 40.3% | 25.8% | — | — |
| creative | 43.0% | 28.0% | — | — |
| adversarial | 55.5% | 40.3% | — | — |

### detail at best γ per workload (by S_sync)

| workload | best γ | accept% | tok/round | f_draft | plain t/s | spec t/s | S_sync | gpu/cpu verify | lossless |
|---|---|---|---|---|---|---|---|---|---|
| code | 2 | 92.1% | 2.82 | 0.8582 | 41.16 | 4.68 | 0.11× | 45/0 | ✓ |
| json | 4 | 84.5% | 4.38 | 0.8994 | 40.26 | 4.22 | 0.10× | 29/0 | ✓ |
| extraction | 4 | 72.3% | 3.85 | 0.9051 | 39.82 | 3.12 | 0.08× | 33/0 | ✓ |
| chat | 2 | 40.3% | 1.80 | 0.8299 | 41.58 | 3.93 | 0.09× | 70/0 | ✓ |
| creative | 2 | 43.0% | 1.85 | 0.8329 | 41.62 | 4.10 | 0.10× | 68/0 | ✓ |
| adversarial | 2 | 55.5% | 2.10 | 0.8411 | 41.65 | 4.23 | 0.10× | 60/0 | ✓ |
