# BASALT Phase 4b — NVFP4-mm CUDA `__dp4a` decode re-measure (measured, this box)

Parity-neutral kernel upgrade of the v1 `nvfp4_gemv` inner loop: the scalar nibble-unpack +
KV-LUT × q8 multiply is replaced by the pin's `get_int_from_table_16` `__byte_perm` codebook
expansion + `__dp4a` 4-way int8 dot (ggml-cuda/vecdotq.cuh:34-80, 331-359). The i32 `sumi` is
identical, so it cannot move parity — confirmed by the unchanged bit-parity gate below.

- Hardware: RTX 3060 Laptop GPU, sm_86, 6144 MiB, driver 576.83, CUDA 12.9, Windows 11.
- Theoretical DRAM bandwidth: 336.0 GB/s (192-bit GDDR6 @ 14 Gbps).
- Engine: `basalt/phase4-cuda-decode`, Phase 4b dp4a kernel, `cargo build --release --features cuda`.
- Model: `<camelid>/models/gemma-4-E4B-it-NVFP4-mm.gguf` sha256 `eb293344…9863d9` (matmuls NVFP4).
- Method: **median of 5 warm runs** (a separate warm-up run discarded first), fixed prompt
  `"The water cycle works as follows:"`, **128 greedy tokens**, decode-only "all" tok/s.
  Command: `<cam-basalt>/target/release/camelid.exe gemma4-cuda-generate <model> --prompt
  "The water cycle works as follows:" --max-tokens 128` (same invocation + prompt as the G4 run).
  Achieved GB/s = per-token GPU weight read bytes (`byte_accounting.json` = 3.048 GB, unchanged
  by dp4a) × decode tok/s. Peak VRAM sampled by polling `nvidia-smi memory.used` while the single
  foreground engine process ran; VRAM verified freed to 0 after every load.

## Bit-parity gate (the whole safety of the change) — HELD

`cargo test --features cuda --lib nvfp4_gemv -- --ignored`:

```
nvfp4_gemv_matches_oracle: 46/46 rows bit-identical, worst rel diff 0.000e0
test cuda_resident::tests::nvfp4_gemv_requires_even_q8_blocks ... ok
test cuda_resident::tests::nvfp4_gemv_decodes_ue4m3_sentinels_pin_cpu_bitwise ... ok
test cuda_resident::tests::nvfp4_gemv_fuses_residual_add ... ok
test cuda_resident::tests::nvfp4_gemv_matches_oracle ... ok
test result: ok. 4 passed; 0 failed; 0 ignored
```

Also green: `cargo fmt --check`; `cargo clippy --all-targets --all-features -- -D warnings`;
plain `cargo test` (no cuda), matrix meta-test included.

## Perf runs (discarded warm-up decode_all = 26.63)

| run | decode_all (tok/s) | decode_steady (tok/s) | overall | load_s | peak VRAM (MiB) |
|---:|---:|---:|---:|---:|---:|
| 1 | 26.86 | 26.50 | 24.94 | 6.7 | 3479 |
| 2 | 26.74 | 26.35 | 24.97 | 6.6 | 3479 |
| 3 | 26.51 | 26.12 | 24.76 | 6.5 | 3479 |
| 4 | 26.49 | 26.13 | 24.74 | 6.7 | 3479 |
| 5 | 26.41 | 26.01 | 24.58 | 6.7 | 3479 |
| **median** | **26.51** | **26.13** | — | — | **3479** |

Derived: achieved BW = 3.048 GB × 26.51 = **80.8 GB/s** = **24.0 %** of the 336 GB/s roofline.

## Result (honest)

| | v1 (scalar LUT) | Phase 4b (`__dp4a`) | Q8_0 baseline |
|---|---:|---:|---:|
| decode tok/s (median) | 14.64 | **26.51** | 25.80 |
| % of 336 GB/s roofline | 13.3 % | **24.0 %** | 39.8 % |
| peak VRAM (MiB) | 3479 | 3479 | 5559 |

- **v1 → dp4a: 14.64 → 26.51 tok/s = +81.1 % (1.81×).**
- **dp4a vs Q8_0: 26.51 vs 25.80 tok/s → NVFP4-mm CUDA is now FASTER than Q8_0 (1.03×)**, while
  reading 1.70× fewer bytes/token and using 2.08 GB less VRAM.
- The kernel did **not** reach the memory roofline (24.0 % vs Q8_0's 39.8 %) — it is not yet
  fully memory-bound — but the ~1.8× throughput lift overtakes Q8_0 because NVFP4 moves fewer
  bytes. Verdict: **Option-B win — NVFP4-mm is now both faster than Q8_0 and lighter in VRAM on
  this box** (narrow, measured, decode-only, this card).

## Safety

Single engine process on the GPU throughout; free-RAM/VRAM checked before every load (≥4 GB free
of 5996), one process at a time, killed by PID, VRAM verified freed to 0 MiB after every run
(and at exit). Peak 3479 MiB, identical across all 5 runs. Zero incidents.
