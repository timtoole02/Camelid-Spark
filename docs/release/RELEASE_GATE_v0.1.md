# Camelid v0.1 Release Gate

Date: 2026-06-05

Release checkout: `main` @ `af35d0436c29845bb15d74844fef68c7a00bc29a` (clean, == origin/main; CI green on this head).

Tag status: cutting `v0.1.0` from this gate run.

## Gate Summary

Current status: ALL gates green on the tag head. Real same-host comparator evidence now exists for llama.cpp and MLX-LM across three exact rows plus a decode-at-depth lane; the Ollama baseline is explicitly deferred (rationale below).

The runtime/API/frontend contract now treats Mistral as evidence-only and fail-closed for v0.1. Lightweight code gates pass locally on this branch. This file records the commands that ran, their results, and the remaining release blockers.

## Required Lightweight Gates

Run these from the release-candidate checkout:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
node scripts/check-public-evidence-claims.mjs
bash scripts/check-public-scrub.sh
cd frontend && npm ci && npm run build && npm run smoke:model-state
```

## Gate Results

| Gate | Command | Status | Notes |
| --- | --- | --- | --- |
| Branch/SHA | `git status --short --branch && git rev-parse HEAD` | PASS | Confirmed a clean candidate checkout for that gate run; record the exact branch and SHA alongside the gate output that is used to cut rc1. |
| Rust format | `cargo fmt --all -- --check` | PASS | Source tree is formatted after applying the Mistral contract and clippy fixes. |
| Rust clippy | `CARGO_TERM_COLOR=never cargo clippy --all-targets --all-features -- -D warnings` | PASS | Clippy passed. Cargo emitted build-script hardlink warnings from the external target cache only. |
| Rust check | `CARGO_TERM_COLOR=never cargo check --all-targets --all-features` | PASS | Cargo check passed with external target-cache hardlink warnings only. |
| Rust tests | `CARGO_TERM_COLOR=never cargo test --all-targets --all-features --no-fail-fast` | PASS | Full suite passed: lib tests 310 passed / 1 ignored, main tests 12 passed, integration/example tests passed. Metal unit tests passed after test-only command-buffer reuse was disabled. |
| Release build | `CARGO_TERM_COLOR=never cargo build --release --bin camelid` | PASS | Release binary built successfully. |
| Public evidence claims | `node scripts/check-public-evidence-claims.mjs --root qa/evidence-bundles` | PASS | Checked 96 manifest files and 49 summary files. |
| Public scrub | `bash scripts/check-public-scrub.sh` | PASS | No public scrub violations reported. |
| Frontend build/model-state smoke | `cd frontend && npm run build && npm run smoke:model-state` | PASS | Vite build passed and model-state smoke passed after removing Mistral from tracked full-support rows. |
| Benchmark harness self-test | `node tools/bench/test-v0.1-benchmark-harness.mjs` | PASS | Synthetic self-test passed; this is harness validation only, not real comparator evidence. |

## Comparator and Evidence Gates

| Gate | Status | Required before tag |
| --- | --- | --- |
| v0.1 evidence bundle | PASS | Real three-runtime, three-round alternating-order bundles are committed: 3B prefill+decode `qa/evidence-bundles/apple-silicon-m4-3b-q8-throughput-camelid-llamacpp-mlx-20260604T214257Z-head-0c6ec54/`, 1B/8B rows `...-1b-8b-q8-throughput-...-20260605T043953Z-head-d7c2940/`, decode-at-depth (a recorded known-behind lane) `...-3b-q8-decode-at-depth-...-20260605T022916Z-head-d7c2940/`. |
| llama.cpp baseline | PASS | Same-host pinned llama-bench runs inside each bundle above (raw md logs + version file committed). |
| MLX-LM baseline | PASS | Same-host mlx-lm runs inside each bundle above (raw logs + version file committed). |
| Ollama baseline | DEFERRED | Release-captain deferral, approved in the tag sign-off: no public Camelid claim references Ollama, and Ollama wraps the already-baselined llama.cpp engine on this host. An Ollama lane can be added post-v0.1 without changing any v0.1 claim. |
| Support matrix | Out of scope for this lane | Owned by another lane; do not edit here. |
| Correctness matrix | Out of scope for this lane | Owned by another lane; do not edit here. |

## Current Blocking Failures

None. Gate run of 2026-06-05 on `af35d04`: fmt PASS, clippy --all-targets --all-features -D warnings PASS, cargo test --all-targets --all-features 533 passed / 0 failed, release build PASS, public evidence claims PASS (101 manifests / 49 summaries), public scrub PASS, benchmark harness self-test PASS, frontend npm ci + build + model-state smoke PASS.

## Tag Rule

Do not create `v0.1.0-rc1` or `v0.1.0` from this lane until:

- lightweight gates pass or have documented non-blocking failures
- comparator baseline status is resolved
- a fresh v0.1 evidence bundle exists or is explicitly deferred by the release captain
- release docs and README contain no unsupported performance, model-family, UI, or distributed claims
- release captain signs off
- Tim approves any final `v0.1.0` tag

## v0.1.0 Sign-off

- Gate run: 2026-06-05, `main` @ `af35d04`, all gates green (table above).
- Comparator baselines: llama.cpp and MLX-LM satisfied by committed same-host bundles; Ollama explicitly deferred (rationale above).
- Release docs and README state wins, ties, and losses with per-lane boundaries; no unsupported claims found by the evidence-claims or scrub gates.
- Tim approved cutting `v0.1.0` (2026-06-05 session).
