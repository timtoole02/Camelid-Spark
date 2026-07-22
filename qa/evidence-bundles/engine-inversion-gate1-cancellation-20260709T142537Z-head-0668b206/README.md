# Gate 1 receipt — cancellation plumbing: hazard closed, parity + perf preserved

Mission: API Engine Inversion (docs/recon/ENGINE_INVERSION_CONDUCTOR.md).
Candidate head: `0668b206` (Phase 1 on `feat/engine-inversion`, base pin `ffada00f`).

## Gate 1 criteria and results

1. **P0-T1/T2/T3 flipped failing → passing with assertions unchanged** —
   `logs/orphan-tests-PASS.log` (3 passed). The tests' handler mimicry was
   updated to the new handler shape (guard rides the decode, CancelOnDrop in
   frame), mirroring the real handlers line-for-line; the invariant assertions
   (`orphans_at_next_acquire == 0`, `max_concurrent <= 1`) are byte-identical
   to the Gate 0 failing versions.
2. **Full existing unit suite green** — `logs/full-suite-PASS.log`
   (675 passed / 0 failed / 55 ignored, the usual gated set).
3. **fmt + clippy** — `cargo fmt --check` clean;
   `cargo clippy --all-targets --all-features -- -D warnings` clean.
4. **Supported-row parity, byte-identical pre/post** — `parity/`:
   release binaries built at the pin (`baseline`) and at the candidate
   (`phase1`) from the same toolchain/profile; each ran the same fixed greedy
   matrix (3 prompts × {chat 48 tok, completion 32 tok, SSE stream 24 tok})
   against **TinyLlama 1.1B Chat Q8_0** and **Llama 3.2 1B Instruct Q8_0** on
   the default (GPU-resident) lane. Canonicalized outputs (uuid ids and
   timing fields stripped; nothing else) are **byte-identical** per model:
   chat JSON, completion JSON, and full SSE transcripts. Capture harness
   included (`parity/engine-inversion-parity.mjs`); one server at a time,
   PID-verified shutdown between legs.
   *Disclosure:* the brief named TinyLlama + Mistral; no Mistral GGUF exists
   on this host and the host has no model-download lane, so Llama 3.2 1B
   Q8_0 (also a supported exact row) substitutes. Mistral replay is carried
   as a Phase 4 re-cert item for a host that has the row.
5. **Perf within noise** — steady-state 48-token chat wall times (same leg
   data, `perf` arrays in the parity JSONs):
   TinyLlama 435.5/434.4 ms (baseline) vs 432.6/441.4 ms (phase1);
   Llama-1B 459.6/414.3 ms vs 444.7/409.5 ms. Stream TTFB 10–13 ms vs
   10–13 ms (TinyLlama), 35–44 ms vs 40–46 ms (Llama-1B). The per-token
   cancellation check (one atomic load + clock read) does not register.
   First-request wall times differ by warmup variance only.

## Verdict

GATE 1: **PASS**. Guard lifetime now equals compute lifetime on every
generation path; the orphan-decode hazard class is closed at the plumbing
level. Phase 2 (engine inversion proper) may proceed.
