# Camelid Benchmarks

Last updated: 2026-06-04

This file is Camelid's public performance snapshot.

It is intentionally narrower than a marketing benchmark page:

- it reports only sanitized, reproducible numbers already backed by committed evidence bundles
- it separates **runtime measurements** from **support claims**
- it does **not** treat one good number as broad model-family proof

If a row or host is not listed here, Camelid is not claiming a benchmark result for it yet.

For cross-surface wording discipline, see [`docs/WAR_ROOM_EVIDENCE_INDEX.md`](../WAR_ROOM_EVIDENCE_INDEX.md). That policy keeps benchmark copy tied to scrubbed evidence and prevents timing probes from becoming support, API, or WebUI readiness claims.

## Reading rules

- **Exact row only.** Numbers apply only to the exact GGUF row named in the table.
- **Bounded workload only.** These are short, bounded validation or microbenchmark lanes, not production multi-user load tests.
- **Same-host comparison only.** A fair Camelid-vs-llama.cpp throughput claim requires the same prompt shape, same model file, same host, same thread settings, and the same token budget.
- **Parity first.** Camelid treats 1:1 token parity with llama.cpp as the prerequisite for performance claims, not a substitute for them.

## Benchmark snapshot: current committed numbers

### Apple Silicon same-host throughput: Camelid vs llama.cpp vs MLX-LM

These results come from `qa/evidence-bundles/apple-silicon-m4-3b-q8-throughput-camelid-llamacpp-mlx-20260604T214257Z-head-0c6ec54/manifest.json`.

Method summary:

- host: one Apple M4 (10-core GPU, 16GB unified memory), warm
- exact row: Llama 3.2 3B Instruct Q8_0 (GGUF for Camelid and llama.cpp; `mlx-community` 8-bit weights for MLX-LM)
- workload: 601-token prompt (prefill/TTFT) and a short prompt with 128 greedy tokens (decode), bounded iterations
- comparators: llama.cpp `llama-bench` (brew, Metal; `-p 601 -n 128`), MLX-LM (versions recorded in the bundle)
- method: three same-session rounds with alternating runtime order; the headline number is the per-runtime median across rounds (Camelid per-round value is the median of 5 measured iterations)

| Lane | Camelid | llama.cpp | MLX-LM (8-bit) |
| --- | ---: | ---: | ---: |
| Prefill, 601-token prompt (tok/s, median of rounds) | 587.3 | 543.7 | 577.9 |
| Decode, short context (tok/s, median of rounds) | 29.74 (128 tokens) | 29.14 (tg128) | 29.13 (two-point, 64) |
| Time to first token, 601-token prompt (ms, median) | 1068 | - | - |

Reading boundary:

- Both lanes on this row and host read above both comparators in every round of this session (prefill: Camelid 587.3 / 587.3 / 587.4 vs MLX-LM 579.3 / 577.9 / 577.9 and llama.cpp 543.7 / 543.7 / 543.8; decode: Camelid 29.74 / 29.85 / 29.74 vs MLX-LM 29.54 / 29.13 / 29.01 and llama.cpp 29.08 / 29.14 / 29.22). Session-median margins are +1.6% (prefill) and +2.1% (decode) over MLX-LM; every Camelid round exceeds every comparator round, but the margins are narrow — a same-session win on this exact row, not a durable or general claim.
- All three runtimes read faster on this host in this session than in the prior snapshot (llama.cpp tg128 28.7 -> 29.1, MLX-LM decode 28.7 -> 29.1), so cross-session comparisons are invalid; the claim rests on the same-session rounds only.
- Camelid's greedy decode rides the resident GPU-sampling fast lane (the default at temperature 0); its 128-token continuation matches the CPU sampling path token-for-token in every iteration of every round.
- This is one exact row on one host. Nothing here transfers to other models, quantizations, context shapes, or hosts.


### Apple Silicon 1B / 8B rows (same host, three runtimes)

Same method as the 3B table (three same-session alternating rounds; medians), from
`qa/evidence-bundles/apple-silicon-m4-1b-8b-q8-throughput-camelid-llamacpp-mlx-20260605T043953Z-head-d7c2940/`:

| Row / lane | Camelid | llama.cpp | MLX-LM (8-bit) |
| --- | ---: | ---: | ---: |
| Llama 3.2 1B Q8_0 prefill (tok/s) | 1664.3 | 1472.8 | 1670.0 |
| Llama 3.2 1B Q8_0 decode (tok/s) | **74.8** | 67.2 | 69.7 |
| Llama 3 8B Q8_0 prefill (tok/s) | **234.2** | 220.4 | 229.2 |
| Llama 3 8B Q8_0 decode (tok/s) | 12.1 | 12.1 | 12.0 |

Reading boundary:

- 1B decode reads above both comparators in every round; 1B prefill reads above
  llama.cpp and parity-level with MLX-LM (median difference below inter-round
  spread — no win claimed on that lane).
- 8B prefill reads narrowly above both; 8B decode is a three-way parity band — the
  lane is weight-bandwidth-bound and no runtime separates from the others.
- Greedy continuations were identical across every iteration of every round on
  both rows. Same-session snapshots on exact rows; nothing transfers.

### Apple Silicon context-depth boundary (same host, Camelid vs llama.cpp)

A 2026-06-04 context sweep — the first measurement past the published 601-token row —
bounded how far the row above extends. It does not: throughput falls steeply with
prompt depth, and above roughly 1.7k tokens (the attention-as-matmul cap on this
head count) llama.cpp leads decisively.

| Prompt depth | Camelid prefill (tok/s) | llama.cpp prefill (tok/s) |
| --- | ---: | ---: |
| 601 tokens | 587.3 (table above) | 543.7 |
| ~2k tokens | ~558 | 521.7 (pp2048) |
| ~4k tokens | ~516 | - (not probed) |
| ~8k tokens | ~445 | 380.8 (pp8192) |
| ~16k tokens | ~285 | - (not probed) |

Reading boundary:

- The attention-as-matmul prefill tiles its S/P panels by query blocks, so its
  memory is linear in prompt depth and the path now reaches every depth the
  resident prefill admits (16k checked; scratch budget an eighth of physical RAM
  capped at 2 GiB, CAMELID_METAL_ATTN_MM_CAP_MB overrides; prompts that fit one
  block reproduce the untiled math bit-for-bit). Camelid 2k/4k/8k probes read
  above llama.cpp's pp2048/pp8192 in single warm runs — recorded here as probes,
  NOT a protocol-grade claim. Anchored-recall continuations stay intact through
  16k.
- Decode-at-depth is a measured comparator lane. The latest session (bundle
  `...-decode-at-depth-...-head-ec73ff2/`, after the split-K kv16 mirrors)
  reads Camelid 25.22 / 18.72 tok/s at ~1.5k / ~8k vs llama.cpp 25.69 / 18.92
  and MLX-LM 26.51 / 21.84: parity-class with llama.cpp at 8k (overlapping
  round bands), slightly below it at 1.5k, and behind MLX-LM at both depths —
  no Camelid win is claimed here. For scale: before the split-K kernel these
  probes read ~16 and ~6, and before the kv16 mirrors ~25 and ~17 (prior
  bundle `...-20260605T022916Z-head-d7c2940/`).
- The sweep also caught a correctness bug on the >cap path (out-of-bounds GPU writes
  producing non-finite logits on every prompt above ~1.7k tokens) and a precision
  cliff in the flash prefill kernel, both fixed/bounded at head 358db2a; the depth
  numbers here are from the fixed paths, which match the CPU reference on anchored
  recall probes through 8k tokens.
- Camelid figures are single warm runs (bounded probes, not a three-round protocol);
  llama.cpp figures are llama-bench means of 2-3 runs in the same session.

### Ubuntu bounded unique-chat envelope

These results come from `qa/evidence-bundles/llama32-1b-3b-unique-chat-perf-rss-20260505T061644Z-head-e9f28572e090/manifest.json`.

Method summary:

- endpoint: `/v1/chat/completions`
- warmup: 2
- measured repeats: 4
- max tokens: 5
- prompt style: unique prompts, average prompt token count `22.5`
- weight cache: hot on measured runs
- prompt cache: no hits on measured runs

| Exact row | Avg wall ms | Avg generate ms | Approx ms / output token | Max backend RSS MiB | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Llama 3.2 1B Instruct Q8_0 | 7379.73 | 7065.25 | 1413.05 | 274.31 | Exact-row bounded unique-chat envelope only |
| Llama 3.2 3B Instruct Q8_0 | 19762.21 | 19449.25 | 3889.85 | 287.21 | Exact-row bounded unique-chat envelope only |

### Apple Silicon memory-first profile vs MLX-LM

These results come from `qa/evidence-bundles/apple-silicon-camelid-vs-mlx-memory-20260514T001835Z-head-775db673af32/summary.json`.

Method summary:

- scope: same-host Apple Silicon resident-memory profile
- Camelid mode: memory-first lazy GGUF Q8_0 (`CAMELID_LAZY_Q8_0_LINEAR=1`, retained Q8 blocks disabled)
- MLX mode: public `mlx-community` 4-bit MLX-LM weights, cache already warm for committed timing rows
- metric: observed process RSS sampled with `ps`; Camelid rows also include structured forward RSS and Q8 file-read counters
- boundary: this is a **memory** comparison, not a strict quant-equivalent speed claim; MLX is much faster in this short probe

| Model family | Camelid row/profile | Camelid observed RSS MiB | MLX-LM row/profile | MLX-LM observed RSS MiB | MLX / Camelid RSS |
| --- | --- | ---: | --- | ---: | ---: |
| Llama 3.2 1B Instruct | GGUF Q8_0, memory-first lazy | 257.72 | 4-bit MLX-LM | 1062.06 | 4.12x |
| Llama 3.2 3B Instruct | GGUF Q8_0, memory-first lazy | 328.92 | 4-bit MLX-LM | 2139.7 | 6.51x |

This is a useful Camelid result, but the claim must stay precise: Camelid's Rust GGUF path can run the exact Q8_0 rows with substantially lower resident memory in the memory-first profile. It does **not** claim Camelid is faster than MLX-LM on this workload.

### Apple Silicon retained-Q8 scheduler gate

These results come from `qa/evidence-bundles/apple-silicon-retained-q8-hybrid-default-off-20260514T020011Z-head-fbd37d1/summary.json`.

Method summary:

- scope: same-host Apple Silicon retained-Q8 short decode gate
- exact row: Llama 3.2 3B Instruct Q8_0
- endpoint: `/v1/chat/completions`
- max tokens: `8`
- weight cache: hot on measured rows
- prompt cache: unique prompts, no measured prompt-cache hits
- boundary: this is a narrow scheduler tuning gate, not a broad Metal benchmark

| Mode | Repeats | Avg generate ms | Avg layers ms | Avg FFN total ms | Observed peak backend RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous implicit hybrid default, 10% GPU suffix | 2 | 4867 | 4749.75 | 883.52 | 4458.73 |
| Explicit CPU-only sweep | 2 | 4808 | 4689.74 | 858.8 | 4255.06 |
| New default, hybrid off | 3 | 4819.33 | 4701.71 | 846.32 | 3817.66 |

Decision: retained-Q8 CPU+Metal hybrid remains available as an explicit experiment, but normal retained-Q8 serving defaults to the CPU path because the measured hybrid suffix scheduler did not win this gate.

### Ubuntu first-token direction probe

This result comes from `qa/evidence-bundles/llama32-3b-parallel-q8-first-token-20260505T140400Z-head-ffc22b85214f/manifest.json`.

Method summary:

- endpoint: `/v1/chat/completions`
- warmup: 1
- measured repeats: 1
- max tokens: 1
- comparison: default serial file-backed Q8 path vs opt-in parallel output-row partitioning

| Exact row | Mode | Avg generate ms | Avg generate ms / prompt token | Max backend RSS MiB | Delta |
| --- | --- | ---: | ---: | ---: | --- |
| Llama 3.2 3B Instruct Q8_0 | serial Q8 path | 13960 | 775.56 | 283.57 | baseline |
| Llama 3.2 3B Instruct Q8_0 | opt-in parallel Q8 path | 12200 | 677.78 | 282.97 | `-12.61%` generate time |

This closes only the exact 3B **first-token direction** sub-box. It is not a broad production-throughput claim.

### Ubuntu compact smoke timing snapshot

These results come from `qa/evidence-bundles/four-row-perf-portability-public-20260503T025639Z/compact-perf-portability-envelope.json`.

Method summary:

- compact smoke / API + WebUI oriented validation lane
- `hello` message, `max_tokens=1`
- same sanitized Ubuntu validation host

| Exact row | Model load ms | Chat completion ms | Backend generate ms | Max RSS KiB |
| --- | ---: | ---: | ---: | ---: |
| TinyLlama 1.1B Chat Q8_0 | 134 | 40558 | 40487 | 136588 |
| Llama 3.2 1B Instruct Q8_0 | 559 | 20973 | 20674 | 347316 |
| Llama 3.2 3B Instruct Q8_0 | 566 | 45138 | 44830 | 559572 |
| Llama 3 8B Instruct Q8_0 | 566 | 81086 | 80794 | 566980 |

These are useful for release-audit snapshots, not for broad “fastest local runtime” marketing.

## Llama.cpp comparison policy

Camelid already publishes bounded **parity** against llama.cpp for the exact supported rows, but the repo does **not yet** publish a fully normalized same-host throughput table against llama.cpp for those rows.

That means Camelid should say this plainly today:

- **Yes:** Camelid has exact-row 1:1 bounded parity with llama.cpp where cited.
- **Not yet:** Camelid has not published a repo-safe apples-to-apples throughput table versus llama.cpp on the same host for every headline row.

That missing table is an execution gap, not a copywriting gap.

The tracked harness for closing the first same-host 3B slice is:

```sh
CAMELID_BIN=target/release/camelid \
LLAMA3_LLAMA_SERVER=target/reference/llama.cpp/build/bin/llama-server \
node scripts/bench-llama3-same-host.mjs \
  --model /path/to/Llama-3.2-3B-Instruct-Q8_0.gguf \
  --model-id llama32-3b-q8-throughput \
  --row-id llama32_3b_instruct_q8_0 \
  --max-tokens 16 --warmup 1 --repeats 3 --threads 8 \
  --out target/bench-llama32-3b-same-host.json
```

Use `--print-plan` with the same arguments to audit exact spawned commands, stdout keys, JSON schema, and metric bounds before starting servers. The harness reports bounded TTFT, elapsed-time, and streamed-chunk-derived decode estimates only; it does not promote production throughput, 1B, Mixtral, neighboring-row, portability, or broader-family support without separate row-specific evidence.

Latest readiness note: `qa/validation-notes/2026-05-14-throughput-host-readiness-recheck.md` records improved approved Ubuntu lane readiness after the full-root-storage condition, but no same-host benchmark has been run or published from that probe. Treat the apples-to-apples table as still missing until a scrubbed row-specific evidence bundle is committed.

## What should be added next

The next benchmark slice worth publishing is:

1. **Same-host Camelid vs llama.cpp on the exact 3B row**
   - same model SHA
   - same prompt pack
   - same token budget
   - same thread settings
   - report TTFT, decode tok/s, and ms/token
2. **Mac benchmark snapshot**
   - especially the exact Llama 3.2 3B Instruct Q8_0 row that is actively exercised in the UI
3. **8B bounded same-host comparison**
   - only after the same-host run is reproducible and repo-safe

## Claim boundary

This file is a benchmark snapshot, not a support matrix.

It does not widen any support claim beyond `COMPATIBILITY.md`, `STATUS.md`, and the cited evidence bundles.
