//! Phase 2 / Gate 2: runnable-lane dequant is bit-exact vs ggml reference fixtures.
//!
//! Fixtures under tests/fixtures/dequant/ are produced by `scripts/gen-dequant-fixtures.py`
//! using llama.cpp's `gguf` package (the numpy port of ggml's dequant). Each fixture
//! carries the exact wire bytes plus the reference f32 output as u32 bit patterns, so
//! the comparison here is bit-exact, not approximate.

use std::path::{Path, PathBuf};

use camelid::gguf::GgufTensorType;
use camelid::runnable::dequantize;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    qtype: String,
    block_bytes: usize,
    n_elements: usize,
    source: String,
    quant_hex: String,
    ref_f32_bits: Vec<String>,
}

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("dequant")
}

fn map_qtype(s: &str) -> GgufTensorType {
    match s {
        "F32" => GgufTensorType::F32,
        "F16" => GgufTensorType::F16,
        "Q8_0" => GgufTensorType::Q8_0,
        "Q4_0" => GgufTensorType::Q4_0,
        "Q4_K" => GgufTensorType::Q4K,
        "Q5_K" => GgufTensorType::Q5K,
        "Q6_K" => GgufTensorType::Q6K,
        other => panic!("unknown fixture qtype {other}"),
    }
}

fn hex_to_bytes(h: &str) -> Vec<u8> {
    assert!(h.len().is_multiple_of(2), "odd hex length");
    (0..h.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&h[i..i + 2], 16).expect("hex byte"))
        .collect()
}

fn ref_f32(bits: &[String]) -> Vec<f32> {
    bits.iter()
        .map(|s| {
            let v = u32::from_str_radix(s.trim_start_matches("0x"), 16).expect("u32 bits");
            f32::from_bits(v)
        })
        .collect()
}

/// f32 distance in ULPs (monotone total ordering of floats), for evidence reporting.
fn ulp_diff(a: f32, b: f32) -> u64 {
    // Canonical total-order key: flip non-sign bits for negatives so the i32
    // ordering matches numeric ordering. Keys are i32, so the diff fits in i64.
    let key = |x: f32| -> i64 {
        let bits = x.to_bits() as i32;
        (if bits < 0 { bits ^ 0x7fff_ffff } else { bits }) as i64
    };
    (key(a) - key(b)).unsigned_abs()
}

struct Report {
    qtype: String,
    source: String,
    n: usize,
    bit_mismatches: usize,
    max_abs_diff: f32,
    max_ulp: u64,
}

fn run_fixture(name: &str) -> Report {
    let path = fixtures_dir().join(format!("{name}.json"));
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing fixture {}: {e}", path.display()));
    let fx: Fixture = serde_json::from_str(&raw).expect("fixture parses");

    let bytes = hex_to_bytes(&fx.quant_hex);
    let reference = ref_f32(&fx.ref_f32_bits);
    assert_eq!(reference.len(), fx.n_elements, "{name}: ref len");

    let tt = map_qtype(&fx.qtype);
    let out = dequantize(tt, &bytes, fx.n_elements, &format!("fixture:{name}"))
        .unwrap_or_else(|e| panic!("{name}: dequant failed: {e}"));
    assert_eq!(out.len(), reference.len(), "{name}: output len");

    let mut bit_mismatches = 0usize;
    let mut max_abs_diff = 0.0f32;
    let mut max_ulp = 0u64;
    for (got, want) in out.iter().zip(reference.iter()) {
        if got.to_bits() != want.to_bits() {
            bit_mismatches += 1;
            max_abs_diff = max_abs_diff.max((got - want).abs());
            max_ulp = max_ulp.max(ulp_diff(*got, *want));
        }
    }
    let rep = Report {
        qtype: fx.qtype,
        source: fx.source,
        n: fx.n_elements,
        bit_mismatches,
        max_abs_diff,
        max_ulp,
    };
    eprintln!(
        "  {:5} n={:5} block_bytes={:3} src={:<40} bit_mismatch={:>4}/{:<5} max_abs={:.3e} max_ulp={}",
        rep.qtype, rep.n, fx.block_bytes, rep.source, rep.bit_mismatches, rep.n, rep.max_abs_diff, rep.max_ulp
    );
    rep
}

/// Integer-exact formats must match ggml bit-for-bit (max abs diff == 0).
const BIT_EXACT: &[&str] = &["F32", "F16", "Q8_0", "Q4_0"];

/// K-quant float-scale formats: bit-exact is the goal; any nonzero diff must be a
/// documented sub-ULP-class bound (see BACKEND_ASKS RA-3). Tightened from evidence.
const KQUANT_MAX_ABS: f32 = 0.0;

#[test]
fn dequant_matches_ggml_reference() {
    eprintln!("=== runnable dequant vs ggml reference (gguf) ===");
    let mut any_fail = false;

    for name in BIT_EXACT {
        let r = run_fixture(name);
        if r.bit_mismatches != 0 {
            any_fail = true;
            eprintln!(
                "  FAIL {}: {} bit mismatches (must be bit-exact)",
                r.qtype, r.bit_mismatches
            );
        }
    }

    for name in ["Q4_K", "Q5_K", "Q6_K"] {
        let r = run_fixture(name);
        if r.max_abs_diff > KQUANT_MAX_ABS {
            any_fail = true;
            eprintln!(
                "  FAIL {}: max_abs_diff {:.3e} exceeds bound {:.3e}",
                r.qtype, r.max_abs_diff, KQUANT_MAX_ABS
            );
        }
    }

    assert!(
        !any_fail,
        "dequant parity failed; see per-format lines above"
    );
}
