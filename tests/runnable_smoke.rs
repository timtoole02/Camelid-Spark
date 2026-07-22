//! Part B: smoke-admission over real oracle-qualified models. Confirms the full
//! smoke pipeline (admission → load → forward sanity → coherence) passes and emits a
//! RUNNABLE receipt (lane=runnable, never copper, parity=not_compared). Run with
//! `--release`. Skips models that are absent.

use std::path::{Path, PathBuf};

use camelid::runnable::smoke_admit;
use serde_json::json;

fn models_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("models")
}

fn run_smoke(filename: &str) {
    let path = models_dir().join(filename);
    if !path.exists() {
        eprintln!("SKIP {filename}: absent");
        return;
    }
    let report = smoke_admit(path.to_str().unwrap()).expect("smoke-admission must pass");
    eprintln!(
        "=== smoke {filename} ===\n  arch={} quant={} tok={:?} prompt_tokens={} logits=[{:.1},{:.1}]\n  gen={:?}\n  txt={:?}",
        report.architecture,
        report.quant,
        report.tokenizer,
        report.prompt_token_count,
        report.logit_min,
        report.logit_max,
        report.generated,
        report.generated_text
    );

    // The receipt must be a RUNNABLE receipt: never copper, and honest that no
    // reference comparison happened (parity not_compared).
    let r = &report.receipt;
    assert!(r.is_runnable(), "smoke receipt must be lane=runnable");
    assert!(
        !r.parity.compared_against_reference,
        "smoke is not a parity check — parity must be not_compared"
    );
    r.verify_self_digest().expect("receipt digest must verify");

    // Persist the runnable smoke receipt for inspection.
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("qa/runnable/smoke");
    std::fs::create_dir_all(&out_dir).ok();
    let out = out_dir.join(format!("{}.json", filename.trim_end_matches(".gguf")));
    std::fs::write(
        &out,
        serde_json::to_string_pretty(&json!({
            "smoke": "runnable-smoke-admission",
            "receipt": r,
        }))
        .unwrap(),
    )
    .expect("write smoke receipt");
    eprintln!("  receipt -> {}", out.display());
}

#[test]
fn smoke_admits_tinyllama() {
    run_smoke("tinyllama-1.1b-chat-v1.0.Q8_0.gguf");
}

#[test]
fn smoke_admits_qwen3() {
    run_smoke("Qwen3-0.6B-Q8_0.gguf");
}
