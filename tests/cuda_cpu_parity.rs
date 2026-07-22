//! CPU-vs-CUDA greedy parity comparator (Task 5).
//!
//! Mirrors the repo's parity-diag pattern: it reads two token streams — one from
//! the **CPU** decode path and one from the **CUDA** path, generated over the
//! frozen fixtures — and gates them token-by-token, writing a
//! `camelid.cpu_cuda_parity/v1` artifact and reporting the first divergence.
//!
//! `#[ignore]` because it needs streams produced from BOTH paths over the same
//! fixtures (the default diag files in `qa/` compare against llama.cpp, not
//! CPU-vs-CUDA). Generate them on the CUDA host, then:
//!
//! ```text
//! CAMELID_PARITY_CPU_JSON=qa/cuda/cpu_tokens.json \
//! CAMELID_PARITY_CUDA_JSON=qa/cuda/cuda_tokens.json \
//!   cargo test --test cuda_cpu_parity -- --ignored --nocapture
//! ```
//!
//! Each input JSON may be a diag file (any of `generated_tokens`,
//! `backend_generated_tokens`, `tokens`, `cpu_tokens`, `cuda_tokens`).

use camelid::cuda_parity::{tokens_from_diag, ParityArtifact, ToleranceGate};

fn read_tokens(path: &str) -> Vec<u32> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let v: serde_json::Value =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {path}: {e}"));
    tokens_from_diag(&v).unwrap_or_else(|| panic!("no token array found in {path}"))
}

#[test]
#[ignore = "needs CPU and CUDA token streams over the frozen fixtures; run on a CUDA host"]
fn cpu_cuda_greedy_parity() {
    let cpu_path = std::env::var("CAMELID_PARITY_CPU_JSON")
        .unwrap_or_else(|_| "qa/parity_cpu_diag.json".into());
    let cuda_path = std::env::var("CAMELID_PARITY_CUDA_JSON")
        .unwrap_or_else(|_| "qa/parity_cuda_diag.json".into());
    let model = std::env::var("CAMELID_PARITY_MODEL").unwrap_or_else(|_| "validated-model".into());
    let fixture =
        std::env::var("CAMELID_PARITY_FIXTURE").unwrap_or_else(|_| "frozen-fixture".into());

    let cpu = read_tokens(&cpu_path);
    let cuda = read_tokens(&cuda_path);

    // The shipped Q8_0 kernel is bit-exact (--fmad=false, CPU-mirrored order), so
    // the default gate allows zero token divergences. Override the regime here if
    // comparing a non-bit-exact path.
    let artifact = ParityArtifact::evaluate(model, fixture, ToleranceGate::bit_exact(), cpu, cuda);

    let mut json = serde_json::to_string_pretty(&artifact.to_json()).unwrap();
    json.push('\n');
    eprintln!("{json}");
    let _ = std::fs::create_dir_all("qa/cuda");
    std::fs::write("qa/cuda/parity-latest.json", &json).expect("write parity artifact");

    assert!(
        artifact.passed(),
        "CPU↔CUDA parity FAILED: first divergence at {:?} ({} divergences over {} tokens)\n{}",
        artifact.comparison.first_divergence,
        artifact.comparison.divergences,
        artifact.comparison.compared,
        json
    );
}
