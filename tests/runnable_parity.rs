//! Phase 5 / Gate 5: external parity anchor — runnable lane vs HF transformers.
//!
//! HARD GATE: greedy token sequence matches HF transformers EXACTLY on frozen
//! fixtures. Logit max-abs-diff is reported as evidence (HF runs the same
//! GGUF-dequantized, Q/K-unpermuted weights, so the diff is pure graph fidelity).
//! On success a parity artifact (lane=runnable) is written and is traceable to the
//! exact GGUF bytes. Passing this is what makes the runnable lane an oracle for
//! (llama, Q8_0, SPM).
//!
//! Compute-heavy: run with `--release`. Skips if the model or HF fixtures are absent.

use std::path::{Path, PathBuf};

use camelid::receipt::sha256_file_hex;
use camelid::runnable::RunnableModel;
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
struct PromptFixture {
    prompt_text: String,
    prompt_ids: Vec<u32>,
    greedy_ids: Vec<u32>,
    first_step_logits_bits: Vec<String>,
}

#[derive(Deserialize)]
struct HfFixtures {
    reference: String,
    gguf: String,
    max_new: usize,
    fixtures: Vec<PromptFixture>,
}

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn ref_f32(bits: &[String]) -> Vec<f32> {
    bits.iter()
        .map(|s| f32::from_bits(u32::from_str_radix(s.trim_start_matches("0x"), 16).unwrap()))
        .collect()
}

#[test]
fn llama_matches_hf_transformers_greedy() {
    run_parity(
        "tinyllama.json",
        "tinyllama-parity.json",
        "llama",
        "llama_spm",
    );
}

#[test]
fn qwen3_matches_hf_transformers_greedy() {
    run_parity("qwen3.json", "qwen3-parity.json", "qwen3", "gpt2_bpe");
}

#[test]
fn gemma3_matches_hf_transformers_greedy() {
    run_parity("gemma3.json", "gemma3-parity.json", "gemma3", "llama_spm");
}

#[test]
fn phi3_matches_hf_transformers_greedy() {
    run_parity("phi3.json", "phi3-parity.json", "phi3", "llama_spm");
}

fn run_parity(fixture_file: &str, artifact_file: &str, arch: &str, tokenizer_kind: &str) {
    let fx_path = repo().join("tests/fixtures/hf_parity").join(fixture_file);
    if !fx_path.exists() {
        eprintln!("SKIP: HF parity fixtures absent ({})", fx_path.display());
        return;
    }
    let hf: HfFixtures =
        serde_json::from_str(&std::fs::read_to_string(&fx_path).unwrap()).expect("fixtures parse");

    let gguf_path = repo().join("models").join(&hf.gguf);
    if !gguf_path.exists() {
        eprintln!("SKIP: {} absent", gguf_path.display());
        return;
    }
    let model = RunnableModel::load(gguf_path.to_str().unwrap()).expect("model loads");
    let gguf_sha = sha256_file_hex(&gguf_path).expect("hash gguf");

    eprintln!("=== runnable vs {} ===", hf.reference);
    let mut all_match = true;
    let mut max_logit_abs_diff = 0.0f32;
    let mut prompt_records = Vec::new();

    for pf in &hf.fixtures {
        // Same prompt ids HF used → tokenizer is not a variable here.
        let cam_greedy = model
            .generate(&pf.prompt_ids, hf.max_new)
            .expect("generate");
        let matched = cam_greedy == pf.greedy_ids;
        all_match &= matched;

        // Logit evidence: camelid's last-position logits vs HF's first-step logits.
        let cam_logits = model.forward_logits(&pf.prompt_ids).expect("forward");
        let hf_logits = ref_f32(&pf.first_step_logits_bits);
        let mut diff = 0.0f32;
        if cam_logits.len() == hf_logits.len() {
            for (a, b) in cam_logits.iter().zip(hf_logits.iter()) {
                diff = diff.max((a - b).abs());
            }
        } else {
            diff = f32::INFINITY;
        }
        max_logit_abs_diff = max_logit_abs_diff.max(diff);

        // Both sides argmax the first step — they must pick the same next token.
        let cam_first = cam_greedy.first().copied();
        let hf_first = pf.greedy_ids.first().copied();

        eprintln!(
            "  {:?}\n    hf  greedy = {:?}\n    cam greedy = {:?}\n    match={} first_tok hf={:?} cam={:?} logit_max_abs_diff={:.3e}",
            pf.prompt_text, pf.greedy_ids, cam_greedy, matched, hf_first, cam_first, diff
        );

        prompt_records.push(json!({
            "prompt_text": pf.prompt_text,
            "prompt_ids": pf.prompt_ids,
            "hf_greedy": pf.greedy_ids,
            "cam_greedy": cam_greedy,
            "match": matched,
            "logit_max_abs_diff": diff,
        }));
    }

    eprintln!("  ALL greedy match = {all_match}   max logit_abs_diff = {max_logit_abs_diff:.3e}");

    // Parity artifact (lane=runnable), traceable to the exact GGUF bytes.
    assert_eq!(
        model.architecture, arch,
        "fixture/model architecture mismatch"
    );
    let artifact = json!({
        "lane": "runnable",
        "architecture": model.architecture,
        "quant": "Q8_0",
        "tokenizer": tokenizer_kind,
        "gguf_filename": hf.gguf,
        "gguf_sha256": gguf_sha,
        "reference": hf.reference,
        "fixture_count": hf.fixtures.len(),
        "decode": "greedy/argmax",
        "result": if all_match { "pass" } else { "fail" },
        "all_greedy_match": all_match,
        "max_logit_abs_diff": max_logit_abs_diff,
        "prompts": prompt_records,
    });
    let out_dir = repo().join("qa").join("runnable");
    std::fs::create_dir_all(&out_dir).ok();
    let out_path = out_dir.join(artifact_file);
    std::fs::write(&out_path, serde_json::to_string_pretty(&artifact).unwrap())
        .expect("write artifact");
    eprintln!("  parity artifact -> {}", out_path.display());

    // HARD GATE: greedy token sequence exact-match.
    assert!(
        all_match,
        "runnable greedy token sequence must match HF transformers exactly"
    );
}
