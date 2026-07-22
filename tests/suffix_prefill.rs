//! P0-4 (suffix prefill / multi-turn KV reuse) — session-level A/B regression.
//!
//! Drives two chat turns through the PUBLIC generate entry (prompt prefill +
//! greedy decode) against the process-global resident CUDA engine, with
//! `CAMELID_CUDA_SUFFIX_PREFILL` off (today's full-prefill path) and on (turn 2
//! prefills only the token-exact new suffix over the engine's live KV). The
//! token streams must be identical in every leg — the suffix lane is a pure
//! TTFT optimization, never a numerics change. Also exercises: a divergent
//! history (edited turn 1 → LCP lands mid-history → the tail is re-prefilled
//! from the divergence point), and a CPU-prefilled turn (no committed record →
//! suffix cannot engage → full path, proving the reseed/backdoor guard).
//!
//! Env mutation is confined to this one test fn (legs run sequentially in one
//! process), so there is no cross-test env race.
//!
//! Skips cleanly when the env var is unset (CI carries no model files); run locally:
//!   CAMELID_SUFFIX_PREFILL_GGUF=/path/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
//!     cargo test --release --test suffix_prefill -- --nocapture

use camelid::gguf::read_metadata;
use camelid::inference::{LlamaInferenceSession, LlamaLoadedWeights, LlamaSampler};
use camelid::model::{LlamaModelConfig, LlamaTensorBinding};
use camelid::tensor::TensorStore;
use camelid::tokenizer::Tokenizer;
use std::path::PathBuf;
use std::sync::Arc;

const GENERATE: usize = 8;

struct Model {
    config: LlamaModelConfig,
    weights: Arc<LlamaLoadedWeights>,
    prompt: Vec<u32>,
    followup: Vec<u32>,
}

fn load() -> Option<Model> {
    let model = std::env::var_os("CAMELID_SUFFIX_PREFILL_GGUF").map(PathBuf::from)?;
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
    // Appended at the TOKEN level for turn 2 (matches how the mechanism sees a
    // growing conversation; template re-tokenization effects are out of scope —
    // any token divergence simply shrinks the reused prefix).
    let followup = tokenizer
        .encode(" Next, the capital of Spain is", false, false)
        .expect("encode followup");
    assert!(prompt.len() > 4, "prompt must give the model real history");
    Some(Model {
        config,
        weights,
        prompt,
        followup,
    })
}

fn session(model: &Model) -> LlamaInferenceSession {
    LlamaInferenceSession::new(model.config.clone(), Arc::clone(&model.weights)).expect("session")
}

/// One chat turn on a FRESH session: feed the whole prompt (prefill + first
/// decode), then greedy single-token steps. Returns the generated tokens.
fn turn(model: &Model, prompt: &[u32], cpu_only: bool) -> Vec<u32> {
    let mut s = session(model);
    if cpu_only {
        s.set_resident_paths_disabled(true);
    }
    let mut out = Vec::with_capacity(GENERATE);
    let mut history = prompt.to_vec();
    let mut next = step(&mut s, prompt, &history);
    for _ in 0..GENERATE {
        out.push(next);
        history.push(next);
        next = step(&mut s, &[next], &history);
    }
    out
}

fn step(s: &mut LlamaInferenceSession, feed: &[u32], history: &[u32]) -> u32 {
    s.generate_next_token_with_history_diagnostics(feed, LlamaSampler::Greedy, history, false, None)
        .expect("generation step")
        .next_token_id
}

#[test]
fn suffix_prefill_is_token_identical_across_turns() {
    let Some(model) = load() else {
        eprintln!("SKIP suffix prefill regression: set CAMELID_SUFFIX_PREFILL_GGUF");
        return;
    };

    let two_turns = |model: &Model| -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        // Turn 1: fresh conversation.
        let g1 = turn(model, &model.prompt, false);
        // Turn 2: the conversation extended at the token level.
        let mut history2 = model.prompt.clone();
        history2.extend_from_slice(&g1);
        history2.extend_from_slice(&model.followup);
        let g2 = turn(model, &history2, false);
        // Divergent turn: same length, one mid-history token edited — the LCP
        // lands mid-history and the tail must be re-prefilled from there.
        let mut diverged = history2.clone();
        let mid = model.prompt.len() + 1;
        diverged[mid] = diverged[mid].wrapping_add(1);
        let g3 = turn(model, &diverged, false);
        (g1, g2, g3)
    };

    // Leg A: suffix OFF — today's full-prefill path (the reference).
    std::env::set_var("CAMELID_CUDA_SUFFIX_PREFILL", "0");
    let (a1, a2, a3) = two_turns(&model);
    eprintln!("suffix OFF: t1 {a1:?} t2 {a2:?} t2-div {a3:?}");

    // Leg B: suffix ON — turn 2 reuses the engine's live KV (turn 1 may even be
    // a pure hit on leg A's committed record; identity must hold regardless).
    std::env::set_var("CAMELID_CUDA_SUFFIX_PREFILL", "1");
    let (b1, b2, b3) = two_turns(&model);
    eprintln!("suffix ON : t1 {b1:?} t2 {b2:?} t2-div {b3:?}");

    assert_eq!(a1, b1, "turn 1 diverged under suffix prefill");
    assert_eq!(
        a2, b2,
        "turn 2 (suffix-extended) diverged under suffix prefill"
    );
    assert_eq!(
        a3, b3,
        "divergent-history turn diverged under suffix prefill"
    );

    // Leg C: CPU-prefilled turn 1 (resident paths off) leaves NO committed
    // record; a suffix-on turn 2 must take the full path and stay identical.
    std::env::set_var("CAMELID_CUDA_SUFFIX_PREFILL", "0");
    let c1 = turn(&model, &model.prompt, true);
    let mut history2c = model.prompt.clone();
    history2c.extend_from_slice(&c1);
    history2c.extend_from_slice(&model.followup);
    let c2_ref = turn(&model, &history2c, false);
    std::env::set_var("CAMELID_CUDA_SUFFIX_PREFILL", "1");
    let c1b = turn(&model, &model.prompt, true);
    assert_eq!(c1, c1b, "CPU turn diverged (suffix flag must not touch it)");
    let c2 = turn(&model, &history2c, false);
    assert_eq!(c2, c2_ref, "post-CPU-turn suffix turn diverged");
    std::env::remove_var("CAMELID_CUDA_SUFFIX_PREFILL");
}
