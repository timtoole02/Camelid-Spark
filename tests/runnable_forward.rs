//! Phase 4 / Gate 4: the parametric f32 decoder produces logits and greedy tokens
//! for a llama Q8_0 model end-to-end, and is bit-for-bit deterministic across runs.
//!
//! This does NOT claim correctness vs a reference — that is Phase 5 (HF parity). It
//! asserts the forward path is structurally complete (real logits, real greedy decode)
//! and stable. Compute-heavy: run with `--release`.
//!
//! Skips (does not fail) when the model is absent.

use std::path::{Path, PathBuf};

use camelid::gguf::read_metadata;
use camelid::runnable::RunnableModel;
use camelid::tokenizer::Tokenizer;

fn model_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join("tinyllama-1.1b-chat-v1.0.Q8_0.gguf")
}

fn load() -> Option<(RunnableModel, Tokenizer)> {
    let path = model_path();
    if !path.exists() {
        eprintln!("SKIP: {} not present", path.display());
        return None;
    }
    let model = RunnableModel::load(path.to_str().unwrap()).expect("model loads");
    let tok = Tokenizer::from_gguf(&read_metadata(&path).unwrap()).expect("tokenizer");
    Some((model, tok))
}

#[test]
fn produces_logits_and_greedy_tokens_end_to_end() {
    let Some((model, tok)) = load() else { return };

    eprintln!(
        "loaded {}: layers={} d_model={} heads={}/{} head_dim={} rope_base={} vocab={}",
        model.architecture,
        model.n_layers,
        model.d_model,
        model.n_heads,
        model.n_kv_heads,
        model.head_dim,
        model.rope_base,
        model.vocab
    );

    // End-to-end: text -> tokens (with BOS) -> forward -> logits.
    let prompt = tok
        .encode("The capital of France is", true, false)
        .expect("encode");
    assert!(!prompt.is_empty());

    let logits = model.forward_logits(&prompt).expect("forward");
    assert_eq!(logits.len(), model.vocab, "logits must be vocab-sized");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "all logits must be finite"
    );

    // Greedy decode a few tokens and render them (visibility only, not asserted).
    let generated = model.generate(&prompt, 8).expect("generate");
    assert_eq!(generated.len(), 8);
    let text = tok.decode(&generated, true).unwrap_or_default();
    eprintln!("prompt ids = {prompt:?}");
    eprintln!("greedy gen = {generated:?}");
    eprintln!("greedy txt = {text:?}");
}

#[test]
fn forward_is_bit_exact_deterministic() {
    let Some((model, tok)) = load() else { return };
    let prompt = tok.encode("Hello, world!", true, false).expect("encode");

    // Same input, two runs -> bit-identical logits.
    let a = model.forward_logits(&prompt).expect("run a");
    let b = model.forward_logits(&prompt).expect("run b");
    assert_eq!(a.len(), b.len());
    let mut diffs = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        if x.to_bits() != y.to_bits() {
            diffs += 1;
        }
    }
    assert_eq!(diffs, 0, "logits must be bit-for-bit identical across runs");

    // Greedy decode is likewise reproducible.
    let g1 = model.generate(&prompt, 5).expect("gen 1");
    let g2 = model.generate(&prompt, 5).expect("gen 2");
    assert_eq!(g1, g2, "greedy decode must be reproducible");
    eprintln!("deterministic greedy = {g1:?}");
}
