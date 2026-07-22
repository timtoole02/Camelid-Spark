//! Pillar Two regression — the deterministic forward pass produces a stable execution-trace
//! rollup digest for the supported TinyLlama 1.1B Chat Q8_0 lane.
//!
//! The rollup (`camelid.execution-trace/v1`, `sha256-rollup-v1`) folds every transformer
//! layer's output hidden state and the final logits, across every generated token, into one
//! streaming SHA-256 (see `ExecutionTraceHasher`). It is only meaningful on the order-stable
//! CPU lane (deterministic mode); this test asserts, with tolerance zero:
//!   1. The digest is identical across two runs and across rayon thread counts (1 vs default).
//!   2. A different prompt yields a different digest.
//!   3. The digest equals the committed M4 / i8mm reference (regression pin, i8mm-guarded).
//!   4. Arming the trace OUTSIDE deterministic mode fails closed (no digest).
//!
//! Env-gated on the real GGUF (self-skips without it), like `deterministic_forward`:
//! ```text
//! CAMELID_TINYLLAMA_Q8_GGUF=/path/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
//!   cargo test --release --test execution_trace -- --nocapture
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use camelid::gguf::read_metadata;
use camelid::inference::{LlamaInferenceSession, LlamaLoadedWeights, LlamaSampler};
use camelid::model::{LlamaModelConfig, LlamaTensorBinding};
use camelid::tensor::TensorStore;
use camelid::tokenizer::Tokenizer;

const STREAM_TOKENS: usize = 24;

/// Execution-trace rollup digest for prompt "hello", deterministic mode, Apple M4 (i8mm).
/// Tolerance is exactly zero.
const REF_HELLO_DIGEST: &str = "70649b4da0a1571a16deaa05c7b554a7256643e9f46b0f7a5c969668099f3f4c";

fn host_has_i8mm() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("i8mm")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

fn load(model: &PathBuf) -> (LlamaModelConfig, Arc<LlamaLoadedWeights>, Tokenizer) {
    let gguf = read_metadata(model).expect("read gguf metadata");
    let config = LlamaModelConfig::from_gguf(&gguf).expect("llama config");
    let binding = LlamaTensorBinding::bind(&gguf, &config).expect("tensor binding");
    let store = TensorStore::open(model, &gguf);
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let weights = Arc::new(LlamaLoadedWeights::load(&store, &binding, None).expect("weights"));
    (config, weights, tokenizer)
}

/// Arm the execution-trace rollup, greedily decode `STREAM_TOKENS` tokens, return the digest.
fn rollup_digest(
    config: &LlamaModelConfig,
    weights: &Arc<LlamaLoadedWeights>,
    prompt_ids: &[u32],
) -> String {
    let mut session = LlamaInferenceSession::new(config.clone(), weights.clone()).expect("session");
    assert!(
        session.enable_execution_trace(),
        "execution trace must arm in deterministic mode"
    );
    let step = session
        .generate_next_token(prompt_ids, LlamaSampler::Greedy)
        .expect("prefill step");
    let mut last = step.next_token_id;
    let mut count = 1;
    while count < STREAM_TOKENS {
        let step = session
            .generate_next_token(&[last], LlamaSampler::Greedy)
            .expect("decode step");
        last = step.next_token_id;
        count += 1;
    }
    session
        .take_execution_trace_digest()
        .expect("digest present after an armed deterministic run")
}

#[test]
fn tinyllama_q8_execution_trace_rollup_is_stable_and_pinned() {
    let Some(model) = std::env::var_os("CAMELID_TINYLLAMA_Q8_GGUF").map(PathBuf::from) else {
        eprintln!(
            "SKIP execution_trace: set CAMELID_TINYLLAMA_Q8_GGUF=/path/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
        );
        return;
    };

    std::env::set_var("CAMELID_DETERMINISTIC", "1");
    let (config, weights, tokenizer) = load(&model);
    let hello = tokenizer
        .encode("hello", true, false)
        .expect("encode hello");
    let other = tokenizer
        .encode("goodbye", true, false)
        .expect("encode goodbye");

    // Property 1a — two runs identical.
    let d1 = rollup_digest(&config, &weights, &hello);
    let d2 = rollup_digest(&config, &weights, &hello);
    eprintln!("execution_trace: hello digest = {d1}");
    assert_eq!(d1, d2, "rollup digest diverged between two runs");
    assert_eq!(d1.len(), 64, "digest must be 64 lowercase-hex chars");

    // Property 1b — thread-count invariant (1 worker vs the default pool).
    let d_single = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("1-thread pool")
        .install(|| rollup_digest(&config, &weights, &hello));
    assert_eq!(
        d1, d_single,
        "rollup digest changed with rayon thread count (must be output-partitioned/order-stable)"
    );

    // Property 2 — a different prompt yields a different digest.
    let d_other = rollup_digest(&config, &weights, &other);
    assert_ne!(
        d1, d_other,
        "different prompt must yield a different rollup"
    );

    // Property 3 — exact regression pin (M4 / i8mm). Skipped on other ISAs (internally
    // deterministic but rounds differently); properties 1–2 still hold there.
    if host_has_i8mm() {
        assert_eq!(
            d1, REF_HELLO_DIGEST,
            "rollup digest drifted from the committed M4/i8mm reference (tolerance is zero)"
        );
    } else {
        eprintln!("execution_trace: host lacks i8mm — skipped the M4 exact-digest pin");
    }

    // Property 4 — fail closed outside deterministic mode (reuses the loaded weights).
    std::env::remove_var("CAMELID_DETERMINISTIC");
    let mut plain = LlamaInferenceSession::new(config.clone(), weights.clone()).expect("session");
    assert!(
        !plain.enable_execution_trace(),
        "execution trace must NOT arm outside deterministic mode"
    );
    assert!(!plain.execution_trace_armed());
    assert!(plain.take_execution_trace_digest().is_none());
}
