# Gate 4 receipt — re-certification of the merge candidate (engine inversion)

Mission: API Engine Inversion (docs/recon/ENGINE_INVERSION_CONDUCTOR.md).
Merge candidate: `d50e0ab4` (+ docs/receipts commits) on `feat/engine-inversion`,
base pin `ffada00f`.

## Re-certification results

1. **Supported-row parity, byte-identical baseline-vs-candidate** — this bundle
   adds Llama-3.2-3B-Q8_0 and Qwen3-1.7B-Q8_0 (`parity/`, full request matrix:
   greedy + seeded sampling, stream + non-stream, cache repeat) to the rows
   already receipted at Gates 1–2 (TinyLlama-1.1B-Q8_0 incl. deterministic-CPU
   lane, Llama-3.2-1B-Q8_0). Four rows total, all IDENTICAL.
2. **Live llama.cpp-oracle parity through the inverted engine** — the standing
   `scripts/chat-parity-tinyllama.mjs` harness against a live pinned
   llama-server (`b9632-acd79d603`, the standing comparator on this host):
   `generated_text_match=true`, 25/25 greedy tokens identical, usage integers
   equal (`receipt/verify-receipt-transcript.txt`).
3. **Receipt replay clean on the NEW replay path** — a receipt captured from
   the candidate server and sealed with `camelid seal-receipt` fully verifies
   with `camelid verify-receipt` (self-digest, lane identity, in-process
   Camelid replay — now an engine job, closing the old unlocked-replay hole —
   and the llama.cpp reference re-run): **RECEIPT VERIFIED**
   (`receipt/`).
4. **Perf vs the Phase-0 baseline** — steady-state wall times and stream TTFB
   within noise on every parity leg (perf arrays in the gate 1/2/4 bundle
   JSONs); TTFT under queued load improved ~233x (gate 2 bundle).
5. **llama-server compat endpoints** — `/props`, `/slots`, `/v1/health`,
   `/models` shape tests (tests/api_vertical_slice.rs) green in the final
   suite; `/v1/slots` additionally carries the new `engine_queue_depth` gauge
   in its `camelid` extension block (additive only).
6. **Final unit suite at the merge candidate** — 677 passed / 0 failed
   (`final-suite-PASS.log`); `cargo fmt --check` and
   `clippy --all-targets --all-features -D warnings` clean.
7. **Docs** — DECISIONS.md D16 records the ownership invariant ("only the
   engine thread touches session/KV/resident GPU state; all mutations are
   engine tasks"); STATUS.md carries the serving-engine note (no support claim
   moves — correctness/ownership change only).

## Host-limited disclosures (carried forward, not silently dropped)

- **Mistral-7B-Instruct-v0.3-Q8_0**: no GGUF on this host and no download
  lane; replay re-cert must run on a host that has the row before any
  Mistral-specific claim about the new engine is made. The change is
  model-agnostic (API layer only), and four other supported rows are
  byte-identical, but the row-specific receipt is OWED.
- **Qwen3-4B/8B Q8_0 and Llama-3-8B**: skipped under the standing free-RAM
  rule (model + 3 GB headroom; 6.9 GB free at run time).

## SEV closure

The Gate 0 failing bundle (orphan demonstrated on all three triggers at
`ffada00f`) and the Gate 2 passing bundle (structural fix, soak, 233x TTFT)
stand side by side in qa/evidence-bundles/engine-inversion-gate{0,2}-*.

GATE 4: **PASS for merge** with the host-limited items above carried as
explicit follow-ups.
