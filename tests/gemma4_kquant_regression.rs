//! BASALT Phase 3 SHA_E3 regression — gemma4 files whose LAYER PROJECTION
//! matmuls are K-quants (Q4_K/Q6_K, e.g. the campaign's Q4K-mm and Q4_K_M
//! rows) must generate greedily WITHOUT panicking. Pre-fix, plain generation
//! panicked `unreachable!: K-quant matvec routes through matvec_q8k`
//! (src/gemma4_runtime.rs) because the per-layer projection call sites
//! pre-quantized the shared activation to Q8_0 and bypassed the top-level
//! matvec's K-quant routing — latent pre-BASALT, first reachable via the
//! BASALT S3 experimental rows.
//!
//! Gated on file presence (skip-if-absent): set `CAMELID_GEMMA4_KQUANT_GGUF`
//! to a gemma4 GGUF with K-quant projections (the campaign row is
//! `gemma-4-E4B-it-Q4K-mm.gguf`). Unset, or pointing at a missing file, the
//! test SKIPS ok — no raw model, no claim, no multi-GB load in the default
//! suite (the K-quant routing itself is pinned model-free by the
//! `kquant_projection_tests` unit suite in src/gemma4_runtime.rs).
//!
//! Run: `CAMELID_GEMMA4_KQUANT_GGUF=/path/gemma-4-E4B-it-Q4K-mm.gguf \
//!       cargo test --release --test gemma4_kquant_regression -- --nocapture`

use camelid::gemma4_runtime::Gemma4Runtime;

#[test]
fn kquant_projection_row_generates_greedy_tokens_without_panicking() {
    let Ok(path) = std::env::var("CAMELID_GEMMA4_KQUANT_GGUF") else {
        eprintln!("SKIP: CAMELID_GEMMA4_KQUANT_GGUF not set");
        return;
    };
    let path = std::path::PathBuf::from(path);
    if !path.is_file() {
        eprintln!("SKIP: {} not present", path.display());
        return;
    }
    let rt = Gemma4Runtime::load(&path).expect("load K-quant projection row");
    let (text, ids) = rt
        .generate_greedy("The capital of France is", 4)
        .expect("greedy generation must not fail");
    // The bug was a panic before the first token; any non-empty greedy
    // continuation proves the K-quant projection path executes end to end.
    assert!(!ids.is_empty(), "generated no tokens");
    eprintln!(
        "[kquant-regression] {}: ids={ids:?} text={text:?}",
        path.display()
    );
}
