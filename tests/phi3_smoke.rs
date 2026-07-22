//! phi3 runnable-lane coherence smoke. Phi-3-mini (3.8B) cannot be HF-parity'd on
//! this box (~15 GB f32), so the fused-QKV / fused-gate_up split + NEOX rope are
//! validated here by running greedy decode and checking the output is coherent
//! (not NaN/garbage/repetition). Run with `--release`. Skips if absent.

use camelid::gguf::read_metadata;
use camelid::runnable::RunnableModel;
use camelid::tokenizer::Tokenizer;
use std::path::Path;

#[test]
fn phi3_greedy_is_coherent() {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("models/Phi-3-mini-4k-instruct-Q8_0.gguf");
    if !path.exists() {
        eprintln!("SKIP: phi3 absent");
        return;
    }
    let model = RunnableModel::load(path.to_str().unwrap()).expect("load");
    eprintln!(
        "loaded {} layers={} d_model={} heads={}/{} head_dim={} vocab={}",
        model.architecture,
        model.n_layers,
        model.d_model,
        model.n_heads,
        model.n_kv_heads,
        model.head_dim,
        model.vocab
    );
    let tok = Tokenizer::from_gguf(&read_metadata(&path).unwrap()).expect("tok");

    let prompt = tok
        .encode("The capital of France is", true, false)
        .expect("encode");
    let logits = model.forward_logits(&prompt).expect("forward");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "logits must be finite (no NaN/Inf)"
    );
    let (maxv, minv) = logits
        .iter()
        .fold((f32::MIN, f32::MAX), |(mx, mn), &v| (mx.max(v), mn.min(v)));
    eprintln!("logit range = [{minv:.2}, {maxv:.2}]");
    assert!(maxv < 200.0 && minv > -200.0, "logit range must be sane");

    let gen = model.generate(&prompt, 12).expect("generate");
    let text = tok.decode(&gen, true).unwrap_or_default();
    eprintln!("greedy gen = {gen:?}");
    eprintln!("greedy txt = {text:?}");

    // Coherence: not a degenerate single-token repetition loop.
    let unique: std::collections::HashSet<_> = gen.iter().collect();
    assert!(
        unique.len() >= 4,
        "greedy output is degenerate (only {} unique tokens)",
        unique.len()
    );
}
