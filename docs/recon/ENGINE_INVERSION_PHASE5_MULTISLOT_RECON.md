# Phase 5 recon — multi-slot / continuous batching feasibility (memo only, no code)

Mission: API Engine Inversion (docs/recon/ENGINE_INVERSION_CONDUCTOR.md).
Question: should Camelid adopt llama.cpp-style multi-slot continuous batching
(`server-context.cpp update_slots`, slot scheduler) now that the engine worker
owns all decode compute?

## Verdict: KILL (defer indefinitely, revisit on hardware/workload change)

## Reasoning

1. **The bandwidth story does not want batch > 1 on target hardware.** Camelid's
   primary deployment is a single-user local box (dev host: RTX 3060 Laptop
   6GB / 16GB RAM). CPU decode is memory-bandwidth-bound (STAMPEDE receipts:
   decode sits at the DRAM roofline; the wins came from cutting bytes moved,
   not adding parallel work). Batching k sequences multiplies KV traffic and
   working set by k while the weight-read cost — the part batching amortizes —
   is already the roofline term at batch 1 for the model sizes this host
   serves. Expected effect on this class of hardware: per-request latency up,
   aggregate tok/s roughly flat until the DRAM knee, VRAM pressure immediately
   worse.

2. **VRAM cannot fund it.** The resident CUDA lane's KV cap is already the
   binding constraint at batch 1 (measured: 2090 positions for Qwen3-4B-Q8_0
   on the 6GB card; the 4B/8192-context promotions had to fall back to
   GPU-resident-prefix + CPU tail). k slots divide the resident KV cap by ~k,
   pushing supported context buckets off the GPU — a direct regression of
   shipped, receipt-backed context promotions.

3. **The workload is single-user.** The desktop app and local API serve one
   human. Concurrency arrives as bursts (an agent loop, a retry storm), not as
   sustained parallel sessions; the bounded queue + 12ms TTFT head already
   gives bursts a responsive experience. There is no throughput SLA that
   batch > 1 would satisfy.

4. **What llama.cpp's slot scheduler buys at Camelid's scale:** slot reuse
   (prompt-prefix retention per slot) and fair interleaving for multi-tenant
   serving. Camelid gets the first cheaply from its existing prompt-prefix
   cache (CPU lane) and deliberately bypasses it on the resident CUDA lane for
   parity reasons; the second has no tenant to serve.

5. **Parity cost is real and permanent.** Batched attention changes reduction
   shapes/order per sequence position. Camelid's product identity is
   receipt-backed bit/token parity with the llama.cpp oracle per exact row;
   every batching kernel would need its own parity story (llama.cpp itself
   accepts cross-slot nondeterminism under batching). This is the same class
   of tax that made strict-parity GPU decode sit at 65% of roofline
   (velocity-campaign finding) — but paid across every row, for a workload
   that does not exist locally.

## What WOULD reopen this

- A genuinely multi-user deployment target (LAN server appliance) with
  ≥16GB-VRAM-class hardware, OR
- Draft-batch verification for speculative decode (SPEC_RECHECK lane) needing
  a second sequence — note this is batch-within-one-request (already done for
  GPU spec verify, PR #290-era) and does NOT require the slot scheduler, OR
- An upstream-parity requirement to certify llama.cpp's batched outputs.

## Cheap alternatives already in place after this mission

- Bounded queue + typed 503 (burst handling, observable depth).
- Prep outside serialization (D3): request setup cost overlaps decode.
- SSE head served immediately (role chunk at ~12ms under load).

If interactive fairness under bursts ever matters more, the next cheap step is
priority ordering in the queue (short prompts first) or a cancel-to-front rule
mirroring llama.cpp's CANCEL-to-front — both queue-policy changes inside
`api/engine.rs`, no batching required.
