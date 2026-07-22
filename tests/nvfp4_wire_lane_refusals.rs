//! BASALT Amendment 3 §1.2 — wire-lane refusal trips on COMMITTED synthetic
//! fixtures (no model downloads; both files are tiny real GGUF v3 files built by
//! `scripts/basalt-nvfp4-golden/gen_sidecar_fixture.mjs`, byte-pinned below).
//!
//! Coverage cells (honest per the amendment):
//! - D-B2 sidecar refusal: tripped END-TO-END through `Gemma4Runtime::load` —
//!   the sidecar check fires immediately after `read_metadata`, BEFORE any
//!   config parsing, so the fixture needs no full gemma4 metadata (that
//!   ordering is what `sidecar_fixture_trips_d_b2_end_to_end` proves live).
//! - D17/T5 NaN-sentinel refusal: full-file tripping is NOT reachable in the
//!   wire lane — `LlamaModelConfig::from_gguf` runs before any
//!   `WireQuant::new` sentinel scan and fails first on the fixture's missing
//!   gemma4 config KVs — so the scan seam is driven directly on the fixture's
//!   actual wire bytes instead, and the non-reachability is asserted, not
//!   assumed (`nan_sentinel_fixture_trips_the_scan_seam`).

use camelid::gemma4_runtime::Gemma4Runtime;
use camelid::gguf::{read_metadata, GgufTensorType};
use camelid::tensor::{decode_nvfp4_tensor, nvfp4_find_nan_scale};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

const SIDECAR_FIXTURE: &str = "nvfp4_sidecar_trip.gguf";
const SIDECAR_SHA256: &str = "4220f812bfdc4cb7825241963604bf568963fad41e0bcba6a1c6e2b7e92b7d2d";
const NAN_FIXTURE: &str = "nvfp4_nan_sentinel_trip.gguf";
const NAN_SHA256: &str = "29dda31ac380982ccfa354a0f63963673cb7defa5221f17338df1361552ab2cc";

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("gguf")
        .join(name)
}

fn assert_fixture_pinned(name: &str, want_sha: &str) -> Vec<u8> {
    let bytes = std::fs::read(fixture_path(name)).expect("committed fixture must exist");
    let got = format!("{:x}", Sha256::digest(&bytes));
    assert_eq!(
        got, want_sha,
        "{name} drifted from the generator's pinned bytes; re-run \
         scripts/basalt-nvfp4-golden/gen_sidecar_fixture.mjs and re-pin deliberately"
    );
    bytes
}

#[test]
fn fixtures_are_byte_pinned() {
    assert_fixture_pinned(SIDECAR_FIXTURE, SIDECAR_SHA256);
    assert_fixture_pinned(NAN_FIXTURE, NAN_SHA256);
}

/// §1.2 primary cell: a sidecar-bearing NVFP4 GGUF trips the D-B2 refusal
/// END-TO-END through the wire lane's public entry point.
#[test]
fn sidecar_fixture_trips_d_b2_end_to_end() {
    let path = fixture_path(SIDECAR_FIXTURE);

    // The fixture is a real, fully parseable GGUF v3 file (not a parse error
    // masquerading as a refusal): metadata + tensor table read clean and the
    // sidecar pair is visible by name and type.
    let gguf = read_metadata(&path).expect("fixture must PARSE; the refusal is post-parse");
    assert_eq!(gguf.version, 3);
    assert_eq!(gguf.architecture(), Some("gemma4"));
    let names: Vec<&str> = gguf.tensors.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "blk.0.ffn_down.weight",
            "blk.0.ffn_down.weight.scale",
            "blk.0.ffn_down.weight.input_scale",
        ]
    );
    assert_eq!(gguf.tensors[0].tensor_type, GgufTensorType::NVFP4);

    // End-to-end trip: load refuses with the D-B2 message, naming a sidecar
    // tensor. This works on every platform because the sidecar check fires
    // before both the Amendment 3 §9 platform gate and config parsing.
    // (Gemma4Runtime has no Debug impl, so match instead of expect_err.)
    let msg = match Gemma4Runtime::load(&path) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("sidecar-bearing NVFP4 must refuse (D-B2)"),
    };
    assert!(msg.contains("BASALT D-B2"), "must cite D-B2: {msg}");
    assert!(
        msg.contains("blk.0.ffn_down.weight.scale"),
        "must name the offending sidecar tensor: {msg}"
    );
    assert!(
        msg.contains("sidecar"),
        "must say why (sidecar scales not applied): {msg}"
    );
}

/// §1.2 NaN cell — SEAM-ONLY, with the reason proven in-test: the wire lane's
/// sentinel scan lives in `WireQuant::new`, which `Gemma4Runtime::load` only
/// reaches AFTER `LlamaModelConfig::from_gguf` succeeds. A tiny fixture cannot
/// satisfy full gemma4 config + binding (that needs a real model's worth of
/// metadata and tensors), so config parsing fails first and the sentinel scan
/// is unreachable from the file boundary. The invariant is therefore tested at
/// its seam — `nvfp4_find_nan_scale` (the SAME function `WireQuant::new` calls)
/// and `decode_nvfp4_tensor` (the runnable-lane consumer) — on the fixture's
/// REAL on-disk wire bytes at the descriptor's offset.
#[test]
fn nan_sentinel_fixture_trips_the_scan_seam() {
    let bytes = assert_fixture_pinned(NAN_FIXTURE, NAN_SHA256);
    let path = fixture_path(NAN_FIXTURE);

    // Locate the NVFP4 tensor's wire bytes from the parsed descriptor — the
    // fixture has real tensor data at the correct aligned offset.
    let gguf = read_metadata(&path).expect("fixture must parse");
    let t = &gguf.tensors[0];
    assert_eq!(t.name, "blk.0.ffn_up.weight");
    assert_eq!(t.tensor_type, GgufTensorType::NVFP4);
    assert_eq!(t.n_bytes, 36, "one 64-element superblock");
    let wire = &bytes[t.absolute_offset as usize..(t.absolute_offset + t.n_bytes) as usize];
    assert_eq!(wire[0], 0x7F, "d[0] carries the raw NaN sentinel");

    // Seam 1: the shared scan (the exact fn `WireQuant::new` runs) flags block 0.
    assert_eq!(nvfp4_find_nan_scale(wire), Some(0));

    // Seam 2: the runnable-lane decoder refuses the same bytes, typed.
    let err = decode_nvfp4_tensor(&t.name, wire, 64)
        .expect_err("sentinel-bearing wire bytes must refuse (D17/T5)");
    let msg = err.to_string();
    assert!(msg.contains("NaN-sentinel"), "{msg}");
    assert!(msg.contains("D17/T5"), "{msg}");

    // Documented non-reachability: full-file load DOES error, but never with
    // the sentinel message — on Windows and macOS (both admit NVFP4 since GABBRO
    // M2) it dies in gemma4 config parsing, which precedes WireQuant
    // construction; on the remaining (unvalidated) platforms the Amendment 3 §9
    // platform gate fires even earlier. Either way the file boundary cannot
    // exercise the sentinel scan, which is why this cell is seam-only.
    let msg = match Gemma4Runtime::load(&path) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("skeleton gemma4 metadata cannot load"),
    };
    assert!(
        !msg.contains("NaN-sentinel"),
        "if load ever reaches the sentinel scan, promote this cell to end-to-end: {msg}"
    );
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    assert!(
        msg.contains("NVFP4 is Windows/macOS-only in this release"),
        "on unvalidated platforms the §9 platform gate fires first: {msg}"
    );
}
