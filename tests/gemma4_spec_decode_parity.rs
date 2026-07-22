//! Speculative-decode losslessness gate for the gemma4 CPU runtime.
//!
//! N-gram speculative decode (`Gemma4Runtime::generate_greedy_speculative`) verifies
//! a batch of drafted tokens in one weight pass via `step_chunk`. It is a pure speed
//! optimization: every committed token is the target model's own greedy argmax, so the
//! emitted token stream MUST equal plain `generate_greedy` token-for-token. This test
//! asserts exactly that on the model named by `CAMELID_GEMMA4_GGUF`, across prompts
//! including highly repetitive ones (where draft acceptance — and thus the batched
//! `step_chunk` path — is heavily exercised).
//!
//! Skips cleanly when the env var is unset (CI has no GPU/large model); run locally:
//!   CAMELID_GEMMA4_GGUF=/path/gemma-4-E4B-it-Q8_0.gguf \
//!     cargo test --release --test gemma4_spec_decode_parity -- --nocapture

use camelid::gemma4_runtime::Gemma4Runtime;
use std::path::PathBuf;

#[test]
fn gemma4_speculative_decode_matches_greedy_token_for_token() {
    let Some(model) = std::env::var_os("CAMELID_GEMMA4_GGUF").map(PathBuf::from) else {
        eprintln!("SKIP gemma4 spec-decode parity: set CAMELID_GEMMA4_GGUF");
        return;
    };
    let rt = Gemma4Runtime::load(&model).expect("load gemma4 runtime");

    // Mix of novel prose (ngram rarely fires) and repetitive/structured text (high
    // draft acceptance → the batched verify path dominates). Parity must hold for both.
    let prompts = [
        "Explain the theory of relativity in simple terms.",
        "Once upon a time, in a land far away,",
        "Repeat exactly: the quick brown fox jumps over the lazy dog. \
         the quick brown fox jumps over the lazy dog. the quick brown fox",
        "List: apple, apple, apple, apple, apple, apple,",
    ];
    let max_new = 64;
    for prompt in prompts {
        let (_, greedy_ids) = rt
            .generate_greedy(prompt, max_new)
            .expect("greedy generation");
        let (_, spec_ids) = rt
            .generate_greedy_speculative(prompt, max_new)
            .expect("speculative generation");
        assert_eq!(
            greedy_ids, spec_ids,
            "speculative decode diverged from greedy for prompt {prompt:?}\n  greedy: {greedy_ids:?}\n  spec:   {spec_ids:?}"
        );
        eprintln!(
            "OK ({} tokens): {:?}",
            greedy_ids.len(),
            &prompt[..prompt.len().min(48)]
        );
    }
}
