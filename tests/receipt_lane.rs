use std::{fs, path::Path};

use camelid::gguf::read_metadata;
use camelid::receipt::{sha256_file_hex, sha256_hex, LaneIdentity};

#[test]
fn loading_a_gguf_yields_a_lane_identity_with_the_files_sha256() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lane-fixture.Q8_0.gguf");
    write_tiny_gguf(&path);

    // Independent expectation: hash the raw bytes directly, not through the
    // streaming file reader under test.
    let expected_sha256 = sha256_hex(&fs::read(&path).unwrap());

    let gguf = read_metadata(&path).unwrap();
    let gguf_sha256 = sha256_file_hex(&path).unwrap();
    assert_eq!(gguf_sha256, expected_sha256);

    let lane = LaneIdentity::capture("lane-fixture", &path, &gguf, None, gguf_sha256);

    assert_eq!(lane.gguf_sha256, expected_sha256);
    assert_eq!(lane.gguf_filename, "lane-fixture.Q8_0.gguf");
    // general.file_type = 7 (MOSTLY_Q8_0) drives the quantization label.
    assert_eq!(lane.quantization, "Q8_0");
    assert_eq!(lane.architecture, "llama");
    // No loader-reported kind; falls back to raw tokenizer.ggml.model.
    assert_eq!(lane.tokenizer_kind, "llama");
    let tokenizer_sha = lane.tokenizer_sha256.expect("tokenizer metadata present");
    assert_eq!(tokenizer_sha.len(), 64);
    assert!(!lane.camelid_version.is_empty());
    assert!(!lane.camelid_commit.is_empty());
}

// Minimal GGUF v3 writer, mirroring tests/gguf_metadata.rs.
fn write_tiny_gguf(path: &Path) {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    push_u32(&mut b, 3);
    push_i64(&mut b, 1); // tensor count
    push_i64(&mut b, 4); // metadata count

    push_kv_string(&mut b, "general.architecture", "llama");
    push_kv_string(&mut b, "general.name", "lane-fixture");
    push_string(&mut b, "general.file_type");
    push_u32(&mut b, 4); // u32 type
    push_u32(&mut b, 7); // MOSTLY_Q8_0
    push_kv_string(&mut b, "tokenizer.ggml.model", "llama");

    push_string(&mut b, "token_embd.weight");
    push_u32(&mut b, 2); // dims
    push_i64(&mut b, 4);
    push_i64(&mut b, 2);
    push_i32(&mut b, 0); // f32
    push_u64(&mut b, 0); // relative offset

    while !b.len().is_multiple_of(32) {
        b.push(0);
    }
    b.extend_from_slice(&[0u8; 4 * 2 * 4]);
    fs::write(path, b).unwrap();
}

fn push_kv_string(b: &mut Vec<u8>, key: &str, value: &str) {
    push_string(b, key);
    push_u32(b, 8); // string type
    push_string(b, value);
}

fn push_string(b: &mut Vec<u8>, value: &str) {
    push_u64(b, value.len() as u64);
    b.extend_from_slice(value.as_bytes());
}

fn push_u32(b: &mut Vec<u8>, value: u32) {
    b.extend_from_slice(&value.to_le_bytes());
}
fn push_i32(b: &mut Vec<u8>, value: i32) {
    b.extend_from_slice(&value.to_le_bytes());
}
fn push_u64(b: &mut Vec<u8>, value: u64) {
    b.extend_from_slice(&value.to_le_bytes());
}
fn push_i64(b: &mut Vec<u8>, value: i64) {
    b.extend_from_slice(&value.to_le_bytes());
}
