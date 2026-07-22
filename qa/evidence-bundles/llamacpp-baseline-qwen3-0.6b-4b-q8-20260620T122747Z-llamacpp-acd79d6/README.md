# SPEED_CAMPAIGN.md Phase 1 - llama.cpp honest baseline

Engine: **llama.cpp** @ commit `acd79d603cb2e1c84c0886137b80f1ad649b6857`  
Model pair: qwen3-0.6b-q8 (draft) -> qwen3-4b-q8 (target)  
Generated (UTC): 20260620T122747Z  
Cells: 10

## Machine

| field | value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| Driver | 576.83 |
| VRAM | 6144 MiB |
| SM clock (max / idle snapshot) | 2100 / 210 MHz |
| SM clock policy | free (laptop boost; per-cell pre/post clock bracket recorded) |
| CPU | 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz (16 logical) |
| Host RAM | 15.7 GB |
| CUDA | Cuda compilation tools, release 12.9, V12.9.86 |
| OS | windows 10.0.26220.0 |
| llama.cpp build | -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release (Ninja, VS2022 BuildTools 14.44, CUDA 12.9) |

## Raw decode lane (llama-bench, -fa 1)

| model | prefill pp (t/s) | decode tg (t/s) |
|---|---|---|
| qwen3-4b | 2620.785 +/- 94.261  (n_prompt=512) | 54.506 +/- 0.343  (n_gen=128) |
| qwen3-0.6b | 11890.398 +/- 768.008  (n_prompt=512) | 243.830 +/- 2.325  (n_gen=128) |

## Speculative decode lane (llama-speculative, target+draft, greedy/lossless)

| workload | decode t/s (median +/- sd) | accept % | n_drafted | n_accept |
|---|---|---|---|---|
| code_completion | 74.79 +/- 0.489 | 49.038 | 208 | 102 |
| structured_json | 77.175 +/- 0.555 | 51.5 | 200 | 103 |
| repetitive_extraction | 54.607 +/- 0.195 | 32.292 | 288 | 93 |
| normal_chat | 50.171 +/- 0.295 | 28.438 | 320 | 91 |
| creative_writing | 71.662 +/- 0.32 | 46.429 | 224 | 104 |
| adversarial_lowaccept | 71.346 +/- 0.343 | 45.982 | 224 | 103 |

## llama.cpp spec vs llama.cpp raw-target decode (intra-engine speedup)

Target raw decode baseline: **54.506 t/s**. This is llama.cpp-vs-llama.cpp only;
the cross-engine comparison against Camelid is filled in once Phase 2 lands.

| workload | spec t/s | raw-target t/s | spec/raw |
|---|---|---|---|
| code_completion | 74.79 | 54.506 | 1.37x |
| structured_json | 77.175 | 54.506 | 1.42x |
| repetitive_extraction | 54.607 | 54.506 | 1.00x |
| normal_chat | 50.171 | 54.506 | 0.92x |
| creative_writing | 71.662 | 54.506 | 1.31x |
| adversarial_lowaccept | 71.346 | 54.506 | 1.31x |

## Reproduce

```
git -C <home>\llama.cpp checkout acd79d603cb2e1c84c0886137b80f1ad649b6857
# rebuild (Ninja + CUDA arch 86), then:
pwsh qa/speed/llamacpp-baseline.ps1 -Reps 5
```

**Caveats.** SM clock policy: `free (laptop boost; per-cell pre/post clock bracket recorded)`. On this laptop, boost/thermal drift
is real, so the spec lane runs its columns **interleaved** (round-robin across reps), letting each
column sample the whole thermal timeline instead of penalizing whichever runs last; each cell also
records a pre/post `gpu_clock_bracket_mhz`. Warmup discarded; median +/- stddev over
5 reps. Per D1/D2 (SPEED_CAMPAIGN.md): both lanes are lossless greedy
(each matches its OWN engine's non-spec greedy); this is a SPEED comparison at matched settings,
not a claim that llama.cpp and Camelid emit identical token streams.

