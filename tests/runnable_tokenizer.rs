//! Phase 3 / Gate 3: runnable-lane tokenizer (SPM + BPE) is exact-match vs HF
//! `tokenizers`, validated standalone (no model graph).
//!
//! Fixtures under tests/fixtures/tokenizer_hf/ are produced by
//! `scripts/gen-tokenizer-fixtures.py`, which loads each model's genuine
//! `tokenizer.json` from the HF Hub and records `encode(text, add_special_tokens=False)`
//! id sequences. The Rust side must reproduce them exactly from the SAME model's GGUF
//! metadata — proving the tokenizer is decoupled (built from a GgufFile alone, no
//! inference) and externally anchored to HF, not just to llama.cpp.

use std::path::{Path, PathBuf};

use camelid::gguf::read_metadata;
use camelid::tokenizer::Tokenizer;
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    text: String,
    ids: Vec<u32>,
}

#[derive(Deserialize)]
struct Fixture {
    hf_repo: String,
    gguf: String,
    corpus: Vec<Case>,
}

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tokenizer_hf")
}

fn models_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("models")
}

fn load_fixture(family: &str) -> Fixture {
    let path = fixtures_dir().join(format!("{family}.json"));
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing fixture {}: {e}", path.display()));
    serde_json::from_str(&raw).expect("fixture parses")
}

/// Build the tokenizer from GGUF metadata alone — this is the decoupling proof:
/// no model weights, no inference graph, just `Tokenizer::from_gguf`.
fn tokenizer_from_gguf(gguf_name: &str) -> Option<Tokenizer> {
    let path = models_dir().join(gguf_name);
    if !path.exists() {
        eprintln!("SKIP {gguf_name}: not present at {}", path.display());
        return None;
    }
    let file = read_metadata(&path).expect("gguf parses");
    Some(Tokenizer::from_gguf(&file).expect("tokenizer builds from gguf"))
}

fn check_family(family: &str) {
    let fx = load_fixture(family);
    let Some(tok) = tokenizer_from_gguf(&fx.gguf) else {
        return; // model absent — skip, don't fail
    };

    eprintln!(
        "=== {family} family: {} vs HF {} ({} cases) ===",
        fx.gguf,
        fx.hf_repo,
        fx.corpus.len()
    );

    let mut encode_mismatches = 0usize;
    let mut roundtrip_unstable = 0usize;

    for case in &fx.corpus {
        // core tokenization only: add_special=false, parse_special=false
        let got = tok
            .encode(&case.text, false, false)
            .expect("encode succeeds");

        if got != case.ids {
            encode_mismatches += 1;
            if encode_mismatches <= 5 {
                eprintln!(
                    "  ENCODE MISMATCH text={:?}\n    hf   = {:?}\n    cam  = {:?}",
                    case.text, case.ids, got
                );
            }
        }

        // Round-trip stability. camelid's `decode` is intentionally STATELESS (so it
        // can be called per-token during streaming generation), which means it retains
        // the single `add_space_prefix` dummy `▁` space that `encode` prepended. A
        // consumer recovers re-encodable text by stripping that one leading space —
        // exactly what HF's stateful Metaspace decoder does on the first token. We
        // model that here for SPM, then assert the ids re-encode identically.
        let decoded = tok.decode(&got, false).expect("decode succeeds");
        let normalized = if family == "spm" {
            decoded.strip_prefix(' ').unwrap_or(&decoded).to_string()
        } else {
            decoded
        };
        let reencoded = tok
            .encode(&normalized, false, false)
            .expect("re-encode succeeds");
        if reencoded != got {
            roundtrip_unstable += 1;
            if roundtrip_unstable <= 5 {
                eprintln!(
                    "  ROUNDTRIP UNSTABLE text={:?}\n    ids      = {:?}\n    decoded  = {:?}\n    reencode = {:?}",
                    case.text, got, normalized, reencoded
                );
            }
        }
    }

    eprintln!(
        "  encode_mismatches={}/{}  roundtrip_unstable={}/{}",
        encode_mismatches,
        fx.corpus.len(),
        roundtrip_unstable,
        fx.corpus.len()
    );

    assert_eq!(
        encode_mismatches, 0,
        "{family}: encode must match HF exactly on all cases"
    );
    assert_eq!(
        roundtrip_unstable, 0,
        "{family}: encode->decode->encode must be stable on all cases"
    );
}

#[test]
fn spm_tokenizer_matches_hf() {
    check_family("spm");
}

#[test]
fn bpe_tokenizer_matches_hf() {
    check_family("bpe");
}
