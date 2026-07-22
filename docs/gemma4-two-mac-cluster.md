# Gemma 4 two-Mac clustered inference (distributed layer sharding)

Status: proven on real hardware. Two M4 Mac minis (16 GB each) ran
gemma-4-12b-it-Q8_0 (12.67 GB — memory-infeasible single-node on the primary
host) split master 0..24 / worker 24..48: **all five basic_v1 prompts produced
distributed greedy output token-identical to single-node Camelid**, decode
6.17–6.75 tok/s across the pair, activation payload 15,384 B/step, wire
79.8–119.7 ms/step, TTFT 157–161 s (one-time cold mmap faults on the USB-SSD
master shard), master RSS 6.4–6.8 GB / worker RSS 6.7–8.0 GB. Raw logs:
`qa/evidence-bundles/gemma4-12b-it-q8-0-two-mac-20260610T103711Z-head-96a75007b156`.
Same-host TCP parity for E2B/E4B is locked by `tests/gemma4_distributed_parity.rs`.

Operational notes from the real run: the link-local (Thunderbolt-class) route
measured 0.78 ms vs 17 ms LAN but flapped under sustained load — two prompts
were re-run over LAN with connect retries; link-local self-assigned addresses
can also renegotiate after a link drop (resolve the peer via
`dns-sd -G v4 <peer-hostname>.local` instead of hardcoding). The honest verdict
on the second Mac: it did not make 12B faster than a hypothetical fitting host —
it made 12B POSSIBLE, with both nodes inside their memory budgets.

## What this is — and is not

- **Is**: distributed **layer sharding**. The master Mac runs decoder layers
  `[0, split)`, the worker Mac runs `[split, block_count)` plus the output
  head. The hidden state for one token crosses the wire once per step
  (`hidden * 4` bytes + 24 bytes of framing; ~15 KB for E4B, ~15.4 KB for 12B).
- **Is**: memory-headroom expansion through model partitioning. A 12.67 GB
  Q8_0 row does not fit one 16 GB Mac's working budget; ~6.3 GB of weights per
  node does.
- **Is not**: shared memory. No memory is shared across machines; do not call
  it that.
- **Is not**: a throughput win. Pipeline sharding of a single sequence is
  latency-additive: per-token time ≈ master layers + wire round-trip + worker
  layers. The honest benefit is *fitting the model at all* (or freeing memory
  per node), not speed.

## Determinism and fail-closed guards

- Both nodes run the same `Gemma4Runtime::step_range` math as single-node
  Camelid; activations cross as raw little-endian f32. Distributed greedy
  token ids must match single-node Camelid and the llama.cpp oracle —
  `tests/gemma4_distributed_parity.rs` enforces this.
- The session opens with a handshake (wire version, block count, hidden width,
  layer split, model file length). Any mismatch → typed rejection naming both
  sides' values.
- Every activation and logits payload carries an FNV-1a checksum; corruption
  or framing drift fails the step instead of silently diverging.
- A split through the cross-layer-KV block is rejected at load: the trailing
  shared-KV layers read caches owned by the last sliding/full layers before
  `first_kv_shared`, so the whole block (sources included) must sit on one
  node. E4B (42 layers, 18 shared): split ≤ 22. E2B (35 layers, 20 shared):
  split ≤ 13. 12B has no shared KV: any split works.

## Setup

Both Macs need the same GGUF file locally (same bytes — verify SHA256) and a
camelid binary. Each node memory-maps the file and loads only its own layer
range; embedding tables stay file-backed mmap on both.

### Thunderbolt bridge (preferred)

1. Connect the Macs with a Thunderbolt cable.
2. System Settings → Network → Thunderbolt Bridge → enable on both. With
   link-local addressing the interfaces self-assign 169.254.x.x addresses.
3. `ping` the peer's bridge address to confirm.

Regular LAN/Ethernet works identically (higher latency per round trip).

### Worker (Mac 2 — owns the tail + output head)

```
cargo run --release -- gemma4-worker /path/gemma-4-E4B-it-Q8_0.gguf \
  --addr 0.0.0.0:5005 --first-layer 21
```

### Master (Mac 1 — owns layers 0..split, tokenizer, greedy loop)

```
cargo run --release -- gemma4-master /path/gemma-4-E4B-it-Q8_0.gguf \
  --worker-addr <MAC2_IP>:5005 --split 21 \
  --prompt "The capital of France is" --max-tokens 24
```

`--split` must equal the worker's `--first-layer`. The master prints generated
token ids plus a stats JSON: TTFT, decode tok/s, activation payload bytes,
wire round trips, and per-step local/wire millisecond averages.

## Row guidance (Q8_0, 16 GB Macs)

| Row | Weights | One Mac | Two Macs | Split example |
|---|---|---|---|---|
| E2B-it | 5.05 GB | yes | unnecessary | — |
| E4B-it | 8.19 GB | yes (tight) | yes — frees ~4 GB/node | 21 |
| 12B-it | 12.67 GB | **no** (thrashes) | target lane | 24 |
| 26B A4B-it | 26.86 GB | no | **no** — ~13.4 GB/node + KV + OS exceeds 16 GB; also MoE runtime is not implemented (fail-closed) | — |
| 31B-it | 32.64 GB | no | **no** — ~16.3 GB/node exceeds the machine | — |

## Measuring honestly

Record per run: exact filename + SHA256, both machines' hardware, network
type, layer split, command lines, token-parity result vs single-node and vs
the llama.cpp oracle, TTFT, tok/s, RSS on both nodes, and wire stats. The
bundle must state whether the second Mac improved memory headroom, speed, or
only enabled a row that otherwise fails.
