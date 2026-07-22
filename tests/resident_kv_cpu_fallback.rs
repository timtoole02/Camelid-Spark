//! Regression: a CPU fallback mid-sequence must not destroy a GPU-resident KV history.
//!
//! The GPU-resident lanes advance `kv_cache.position` while writing K/V only into the GPU
//! cache — `try_metal_resident_prefill` sets `position = n` outright, and each resident decode
//! step appends on the GPU. The CPU buffers stay length 0. That is fine while the sequence
//! stays resident, but nothing kept it there:
//!
//!   1. One resident step declines mid-sequence (`forward_token` returns None, a shard/logits
//!      eligibility flip, `resident_paths_disabled` toggled) and the caller falls through to
//!      the CPU layer loop.
//!   2. The CPU layer loop attends over `kv_cache` positions [0, position] — all zeros for
//!      every position the GPU produced. Its `write_kv_cache` then calls
//!      `ensure_position_capacity`, which grows the buffers and writes real K/V for the
//!      CURRENT position only, leaving [0, position) zero-filled.
//!   3. The next resident step sees `filled != position`, decides to reseed, and the old
//!      bounds-only guard (`f32_history_addressable`) now PASSES — the bytes are addressable,
//!      they are simply zero. The seed copies that zeroed prompt onto the GPU.
//!   4. Every later token attends over a zeroed prompt. Wrong output, no error.
//!
//! Step 2 is the part worth stressing: the fallback step is already wrong on its own, before
//! any reseed. So gating the reseed alone does not fix this — declining the resident step just
//! routes to a CPU step that is equally blind. The fix mirrors the resident engine's KV back
//! into the CPU cache before any CPU forward reads it (`ensure_cpu_kv_materialized`), which is
//! what CUDA has always done after its PREFILL (`copy_resident_cuda_kv_to_host`) — but never
//! did per decoded token.
//!
//! BOTH GPU LANES. This test is backend-agnostic on purpose: it drives
//! `generate_next_token_greedy_resident`, which dispatches to whichever resident lane the host
//! has (CUDA when a device is present, else Metal), so the same assertions cover both. The two
//! recoveries differ in how they establish that the engine holds THIS sequence — Metal's engine
//! is a session field, CUDA's is a process-global keyed by model identity, so the CUDA side
//! additionally requires a key match, a matching shard layer count, and `filled == position`
//! exactly. Run it on a CUDA box to cover the CUDA half; a macOS run covers only Metal.
//!
//! THE BINDING ASSERTION is resident-with-fallback == resident-without-fallback. The forced
//! fallback is the only variable between those two runs, so the comparison is confound-free.
//! A pure-CPU run is also compared, but only when it agrees with the clean resident run: the
//! CPU cache rounds K/V through f16 while the resident cache keeps f32, so those two are
//! entitled to diverge at depth for reasons that have nothing to do with this bug.
//!
//! Skips cleanly when the env var is unset (CI carries no model files); run locally:
//!   CAMELID_RESIDENT_KV_FALLBACK_GGUF=/path/Llama-3.2-1B-Instruct-Q8_0.gguf \
//!     cargo test --release --test resident_kv_cpu_fallback -- --nocapture
//! On a CUDA host add `--features cuda` (Linux; Windows builds it by default).

use camelid::gguf::read_metadata;
use camelid::inference::{LlamaInferenceSession, LlamaLoadedWeights, LlamaSampler};
use camelid::model::{LlamaModelConfig, LlamaTensorBinding};
use camelid::tensor::TensorStore;
use camelid::tokenizer::Tokenizer;
use std::path::PathBuf;
use std::sync::Arc;

/// Tokens generated after the prompt. Deliberately short: the point is to catch a history
/// that went to zeros (which derails immediately and unmistakably), not to probe how far two
/// numerically different lanes stay token-identical.
const GENERATE: usize = 8;

/// The step index (into the generated tail) at which the resident lane is forced onto the CPU
/// for exactly one token. Late enough that a real GPU history exists to lose.
const FALLBACK_AT: usize = 2;

struct Model {
    config: LlamaModelConfig,
    weights: Arc<LlamaLoadedWeights>,
    prompt: Vec<u32>,
}

fn load() -> Option<Model> {
    let model = std::env::var_os("CAMELID_RESIDENT_KV_FALLBACK_GGUF").map(PathBuf::from)?;
    let gguf = read_metadata(&model).expect("read gguf metadata");
    let config = LlamaModelConfig::from_gguf(&gguf).expect("model config");
    let binding = LlamaTensorBinding::bind(&gguf, &config).expect("tensor binding");
    let store = TensorStore::open(&model, &gguf);
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let weights = Arc::new(LlamaLoadedWeights::load(&store, &binding, None).expect("load weights"));
    let prompt = tokenizer
        .encode(
            "The capital of France is Paris, and the capital of Italy is",
            true,
            false,
        )
        .expect("encode prompt");
    assert!(prompt.len() > 4, "prompt must give the model real history");
    Some(Model {
        config,
        weights,
        prompt,
    })
}

fn session(model: &Model) -> LlamaInferenceSession {
    LlamaInferenceSession::new(model.config.clone(), Arc::clone(&model.weights)).expect("session")
}

/// Greedy generation entirely on the CPU layer loop — the reference history.
fn cpu_run(model: &Model) -> Vec<u32> {
    let mut s = session(model);
    s.set_resident_paths_disabled(true);
    let mut out = Vec::with_capacity(GENERATE);
    let mut history = model.prompt.clone();
    let mut next = step_cpu(&mut s, &model.prompt, &history);
    for _ in 0..GENERATE {
        out.push(next);
        history.push(next);
        next = step_cpu(&mut s, &[next], &history);
    }
    out
}

fn step_cpu(s: &mut LlamaInferenceSession, feed: &[u32], history: &[u32]) -> u32 {
    s.generate_next_token_with_history_diagnostics(feed, LlamaSampler::Greedy, history, false, None)
        .expect("cpu generation step")
        .next_token_id
}

/// Greedy generation on the GPU-resident lane, optionally forcing exactly ONE token through
/// the CPU layer loop. Returns `None` if the resident lane was never actually taken (no Metal,
/// ineligible model, resident decode off) — the caller then skips rather than passing vacuously.
fn resident_run(model: &Model, force_fallback: bool) -> Option<Vec<u32>> {
    let mut s = session(model);

    // Feed the prompt token-by-token on the resident lane so the whole history is produced by
    // the GPU and the CPU KV buffers stay empty — the state the bug needs.
    let (&first, rest) = model.prompt.split_first().expect("non-empty prompt");
    let mut next = s.generate_next_token_greedy_resident(first).ok()??.0;
    for &tok in rest {
        next = s.generate_next_token_greedy_resident(tok).ok()??.0;
    }

    // LOAD-BEARING PRECONDITION. The bug requires a history the CPU cache does not hold. If
    // the prompt somehow materialized the CPU buffers, this run cannot exercise the defect and
    // would pass on a build with the fix reverted.
    assert!(
        !s.cpu_kv_authoritative(),
        "the resident prompt left the CPU KV cache authoritative, so this run cannot prove \
         anything about a hollow GPU-only history"
    );

    let mut out = Vec::with_capacity(GENERATE);
    let mut history = model.prompt.clone();
    for i in 0..GENERATE {
        out.push(next);
        history.push(next);
        let fed = next;
        next = if force_fallback && i == FALLBACK_AT {
            // Exactly one token off the resident lane: the trigger the chain needs. Any real
            // decline (GPU error, eligibility flip) lands in the same place.
            s.set_resident_paths_disabled(true);
            let id = step_cpu(&mut s, &[fed], &history);
            s.set_resident_paths_disabled(false);
            id
        } else {
            match s.generate_next_token_greedy_resident(fed) {
                Ok(Some((id, _))) => id,
                // An unforced decline would silently turn the control run into another
                // fallback run; fail loudly instead of comparing two contaminated runs.
                Ok(None) => panic!("resident lane declined unexpectedly at generated token {i}"),
                Err(e) => panic!("resident generation step {i} failed: {e}"),
            }
        };
    }
    Some(out)
}

#[test]
fn cpu_fallback_mid_sequence_preserves_the_resident_kv_history() {
    let Some(model) = load() else {
        eprintln!("SKIP resident KV fallback regression: set CAMELID_RESIDENT_KV_FALLBACK_GGUF");
        return;
    };
    // The defect lives on the GPU-resident lane; the CLI turns this on via the execution
    // planner, so a bare test process must ask for it explicitly. Metal needs the opt-in;
    // CUDA is default-on wherever a device is present (and the dispatcher prefers it), so
    // setting this is a no-op on a CUDA host rather than a lane override.
    std::env::set_var("CAMELID_METAL_RESIDENT_DECODE", "1");

    let cpu = cpu_run(&model);
    eprintln!("pure-CPU              : {cpu:?}");

    let Some(clean) = resident_run(&model, false) else {
        eprintln!("SKIP resident KV fallback regression: resident lane unavailable on this host");
        return;
    };
    eprintln!("resident, no fallback : {clean:?}");

    let with_fallback = resident_run(&model, true)
        .expect("the resident lane was available for the control run, so it must be here too");
    eprintln!("resident, one fallback: {with_fallback:?}");

    // THE assertion. One CPU step in the middle of a resident sequence must be transparent.
    // Pre-fix this fails hard: the fallback step attends over a zeroed prompt, and the next
    // resident step reseeds the GPU cache from those same zeros, so the tail is unrelated text.
    assert_eq!(
        with_fallback, clean,
        "one forced CPU fallback changed the output — the GPU-resident KV history did not \
         survive it (the CPU cache holds no history for GPU-produced positions, so the \
         fallback attends over a zeroed prompt and the next resident reseed makes it stick)"
    );

    // Cross-check against pure CPU, but only when it agrees with the clean resident run: the
    // CPU cache rounds K/V through f16 and the resident cache keeps f32, so these two lanes
    // may legitimately part company at depth for reasons unrelated to this bug.
    if cpu == clean {
        assert_eq!(
            with_fallback, cpu,
            "the resident-with-fallback run diverged from the pure-CPU reference"
        );
    } else {
        eprintln!(
            "NOTE: the pure-CPU and clean-resident runs already differ within {GENERATE} tokens \
             (f16 CPU KV vs f32 resident KV); resident-vs-resident stays the binding assertion"
        );
    }
}

/// Greedy generation on the GPU-resident lane, but at `FALLBACK_AT` the next token is produced by
/// the SPECULATIVE CPU verify path (`forward_greedy_verify_chunk`, a one-token batch) instead of a
/// resident decode step. Returns `None` if the resident lane was never actually taken.
///
/// That path is the THIRD CPU KV-history reader (alongside `forward_layer_range_from_hidden` and
/// `forward_single_token_timed_internal`). Before its guard it attended over `[0, position)`
/// without mirroring the resident KV back, so on a hollow CPU cache it read a zero-filled prefix —
/// and its batch write then advanced the materialized-through watermark ACROSS the unwritten gap,
/// so the NEXT resident step's reseed (which trusts the watermark) copied those zeros onto the GPU
/// and made the corruption permanent. In production this path is reached under `CAMELID_SPEC_GPU=1`
/// when `verify_drafts_gpu` declines (an offloaded engine, or a multi-session same-model collision);
/// the test drives it directly, which covers the recovery the same way regardless of how it is
/// reached.
fn resident_run_verify_chunk(model: &Model, force_verify_chunk: bool) -> Option<Vec<u32>> {
    let mut s = session(model);

    // Produce the whole history on the resident lane so the CPU KV buffers stay empty — the state
    // the bug needs.
    let (&first, rest) = model.prompt.split_first().expect("non-empty prompt");
    let mut next = s.generate_next_token_greedy_resident(first).ok()??.0;
    for &tok in rest {
        next = s.generate_next_token_greedy_resident(tok).ok()??.0;
    }

    // Same load-bearing precondition as the single-token regression: a hollow CPU history is what
    // makes the verify chunk's read observable. If the prompt materialized the CPU buffers this
    // run would pass even with the guard reverted.
    assert!(
        !s.cpu_kv_authoritative(),
        "the resident prompt left the CPU KV cache authoritative, so this run cannot prove \
         anything about a hollow GPU-only history"
    );

    let mut out = Vec::with_capacity(GENERATE);
    for i in 0..GENERATE {
        out.push(next);
        let fed = next;
        next = if force_verify_chunk && i == FALLBACK_AT {
            // One token off the resident lane, through the speculative CPU verify chunk. A batch of
            // exactly the fed token appends one position and predicts the token after it — the same
            // net effect as a decode step, but computed on the CPU path that reads `[0, position)`.
            let (predictions, _timings) = s
                .forward_greedy_verify_chunk(&[fed])
                .expect("verify chunk forward");
            assert_eq!(
                predictions.len(),
                1,
                "a one-token chunk yields one prediction"
            );
            predictions[0]
        } else {
            match s.generate_next_token_greedy_resident(fed) {
                Ok(Some((id, _))) => id,
                Ok(None) => panic!("resident lane declined unexpectedly at generated token {i}"),
                Err(e) => panic!("resident generation step {i} failed: {e}"),
            }
        };
    }
    Some(out)
}

#[test]
fn speculative_verify_chunk_preserves_the_resident_kv_history() {
    let Some(model) = load() else {
        eprintln!(
            "SKIP verify-chunk resident KV regression: set CAMELID_RESIDENT_KV_FALLBACK_GGUF"
        );
        return;
    };
    // Same lane opt-in as the sibling test: inert on a CUDA host (default-on), required on Metal.
    std::env::set_var("CAMELID_METAL_RESIDENT_DECODE", "1");

    let Some(clean) = resident_run_verify_chunk(&model, false) else {
        eprintln!(
            "SKIP verify-chunk resident KV regression: resident lane unavailable on this host"
        );
        return;
    };
    eprintln!("resident, no verify-chunk : {clean:?}");

    let with_chunk = resident_run_verify_chunk(&model, true)
        .expect("the resident lane was available for the control run, so it must be here too");
    eprintln!("resident, one verify-chunk: {with_chunk:?}");

    // THE assertion. One speculative CPU verify chunk in the middle of a resident sequence must be
    // transparent. Pre-guard this fails hard: the chunk attends over a zeroed prefix, then the next
    // resident reseed copies those zeros onto the GPU, so the tail is unrelated text.
    assert_eq!(
        with_chunk, clean,
        "one speculative CPU verify chunk changed the output — forward_greedy_verify_chunk read a \
         hollow GPU-only KV history (the CPU cache holds no history for GPU-produced positions), so \
         it attended over a zeroed prefix and the next resident reseed made it stick"
    );
}
