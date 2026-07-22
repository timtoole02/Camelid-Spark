//! Pillar One regression — the supported **TinyLlama 1.1B Chat Q8_0** lane is bit-exact and
//! reduction-order-stable under deterministic mode (`--deterministic` /
//! `CAMELID_DETERMINISTIC=1`). See DECISIONS.md §D9 and `qa/determinism/determinism-baseline-*.md`.
//!
//! Two properties are asserted with a tolerance of **exactly zero**:
//!  1. Two consecutive deterministic runs produce byte-identical first-position logit vectors
//!     and identical token streams (portable: holds on any host, any rayon thread count).
//!  2. The first-position logits equal the committed reference floats (a regression pin).
//!
//! Determinism on the CPU forward pass is already structural (each output owns its full serial
//! K-dimension reduction; no cross-thread float combine), so deterministic mode adds no penalty
//! inside the computation — it only forgoes the GPU fast path. The reduction order mirrors the
//! llama.cpp reference block-wise Q8_0 dot layout the parity contract is gated against.
//!
//! Env-gated on the real GGUF (self-skips without it, like `gemma4_logit_probe`):
//! ```text
//! CAMELID_TINYLLAMA_Q8_GGUF=/path/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
//!   cargo test --release --test deterministic_forward -- --nocapture
//! ```
//! The committed reference floats (property 2) are the **Apple M4 / i8mm** host values; the
//! Q8_0 dot kernel is ISA-specific, so the exact-float pin is asserted only when the running
//! host has `i8mm` (other Apple Silicon is internally deterministic but rounds differently).
//! Property 1 (run-to-run byte identity) is asserted on every host.

use std::path::PathBuf;
use std::sync::Arc;

use camelid::gguf::read_metadata;
use camelid::inference::{LlamaInferenceSession, LlamaLoadedWeights, LlamaSampler};
use camelid::model::{LlamaModelConfig, LlamaTensorBinding};
use camelid::tensor::TensorStore;
use camelid::tokenizer::Tokenizer;

const PROMPT: &str = "hello";
const STREAM_TOKENS: usize = 50;

/// First 12 logits at the first generated position for `PROMPT` under deterministic mode,
/// captured bit-exactly on Apple M4 (i8mm). Tolerance is exactly zero.
const REF_FIRST12: [f32; 12] = [
    -10.165434, -10.214098, 6.0705123, -2.1841283, -4.0164623, -3.6804488, -4.492992, -6.3740983,
    -7.0855656, -7.563206, -7.9338403, -4.390971,
];
/// Argmax index + value at the first generated position (M4 / i8mm).
const REF_ARGMAX_INDEX: usize = 29892;
const REF_ARGMAX_VALUE: f32 = 9.757427;

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

/// One deterministic run: prefill the prompt, capture the first-position logit vector, then
/// greedily decode `STREAM_TOKENS` tokens. Returns (first_position_logits, token_stream).
fn run(
    config: &LlamaModelConfig,
    weights: &Arc<LlamaLoadedWeights>,
    prompt_ids: &[u32],
) -> (Vec<f32>, Vec<u32>) {
    let mut session = LlamaInferenceSession::new(config.clone(), weights.clone()).expect("session");
    let step = session
        .generate_next_token(prompt_ids, LlamaSampler::Greedy)
        .expect("prefill step");
    let first_logits = step.logits.data.clone();
    let mut stream = vec![step.next_token_id];
    let mut last = step.next_token_id;
    while stream.len() < STREAM_TOKENS {
        let step = session
            .generate_next_token(&[last], LlamaSampler::Greedy)
            .expect("decode step");
        last = step.next_token_id;
        stream.push(last);
    }
    (first_logits, stream)
}

#[test]
fn tinyllama_q8_deterministic_mode_is_bit_exact_and_pinned() {
    let Some(model) = std::env::var_os("CAMELID_TINYLLAMA_Q8_GGUF").map(PathBuf::from) else {
        eprintln!(
            "SKIP deterministic_forward: set CAMELID_TINYLLAMA_Q8_GGUF=/path/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
        );
        return;
    };

    // Exercise the real deterministic-mode engine gates (every Metal path fails closed to CPU).
    std::env::set_var("CAMELID_DETERMINISTIC", "1");
    assert!(
        camelid::inference::deterministic_mode_enabled(),
        "CAMELID_DETERMINISTIC=1 must enable deterministic mode"
    );

    let (config, weights, tokenizer) = load(&model);
    let prompt_ids = tokenizer
        .encode(PROMPT, true, false)
        .expect("encode prompt");

    // Property 1 — two consecutive runs are byte-identical (logits + token stream).
    let (logits_a, stream_a) = run(&config, &weights, &prompt_ids);
    let (logits_b, stream_b) = run(&config, &weights, &prompt_ids);

    assert_eq!(
        logits_a.len(),
        logits_b.len(),
        "logit vector length must be stable"
    );
    assert!(
        logits_a
            .iter()
            .zip(&logits_b)
            .all(|(x, y)| x.to_bits() == y.to_bits()),
        "deterministic mode: first-position logits diverged between two runs (tolerance is zero)"
    );
    assert_eq!(
        stream_a, stream_b,
        "deterministic mode: {STREAM_TOKENS}-token streams diverged between two runs"
    );

    // Always surface the captured window so the committed reference can be (re)derived.
    let argmax = logits_a
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, v)| (i, *v))
        .expect("non-empty logits");
    eprintln!(
        "deterministic_forward: vocab={} first12={:?} argmax_index={} argmax_value={} stream[..8]={:?}",
        logits_a.len(),
        &logits_a[..12.min(logits_a.len())],
        argmax.0,
        argmax.1,
        &stream_a[..8.min(stream_a.len())],
    );

    // Property 2 — exact-float regression pin (M4 / i8mm reference). Skipped (with a note) on
    // hosts that take a different Q8_0 dot kernel, which are internally deterministic but round
    // differently; Property 1 still guards run-to-run bit identity there.
    if host_has_i8mm() {
        for (i, expected) in REF_FIRST12.iter().enumerate() {
            assert_eq!(
                logits_a[i].to_bits(),
                expected.to_bits(),
                "first-position logit[{i}] = {} != committed reference {} (tolerance is zero)",
                logits_a[i],
                expected
            );
        }
        assert_eq!(
            argmax.0, REF_ARGMAX_INDEX,
            "first-position argmax index drifted from committed reference"
        );
        assert_eq!(
            argmax.1.to_bits(),
            REF_ARGMAX_VALUE.to_bits(),
            "first-position argmax value drifted from committed reference (tolerance is zero)"
        );
    } else {
        eprintln!(
            "deterministic_forward: host lacks i8mm — asserted run-to-run bit identity only, \
             skipped the M4 exact-float pin"
        );
    }
}
