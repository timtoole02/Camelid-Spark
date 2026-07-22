//! FLINT split-GGUF round trip: `split_gguf` → shards parse → `merge_shards`
//! → the merged file is byte-identical to the original.
//!
//! The original is produced by this file's own canonical writer (same wire
//! serialization the merger emits), so whole-file byte identity is the bar —
//! not just per-tensor equality. Fills are distinct per tensor (a cross-tensor
//! swap breaks byte identity) and one f32 `[1]` scalar forces real alignment
//! padding between tensors (n_bytes = 4, not a 32-multiple).

use std::fs;
use std::path::{Path, PathBuf};

use camelid::gguf::merge::{discover_shard_set, merge_shards, parse_shard_name, split_gguf};
use camelid::gguf::read_metadata;

// ---- canonical little GGUF v3 writer (mirrors the repo's per-test helpers) --

fn push_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn push_kv_string(buf: &mut Vec<u8>, key: &str, value: &str) {
    push_string(buf, key);
    buf.extend_from_slice(&8i32.to_le_bytes());
    push_string(buf, value);
}

fn push_kv_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
    push_string(buf, key);
    buf.extend_from_slice(&4i32.to_le_bytes());
    buf.extend_from_slice(&value.to_le_bytes());
}

fn push_kv_f32(buf: &mut Vec<u8>, key: &str, value: f32) {
    push_string(buf, key);
    buf.extend_from_slice(&6i32.to_le_bytes());
    buf.extend_from_slice(&value.to_le_bytes());
}

fn push_kv_bool(buf: &mut Vec<u8>, key: &str, value: bool) {
    push_string(buf, key);
    buf.extend_from_slice(&7i32.to_le_bytes());
    buf.push(u8::from(value));
}

fn push_kv_u64(buf: &mut Vec<u8>, key: &str, value: u64) {
    push_string(buf, key);
    buf.extend_from_slice(&10i32.to_le_bytes());
    buf.extend_from_slice(&value.to_le_bytes());
}

fn push_kv_array_strings(buf: &mut Vec<u8>, key: &str, values: &[&str]) {
    push_string(buf, key);
    buf.extend_from_slice(&9i32.to_le_bytes());
    buf.extend_from_slice(&8i32.to_le_bytes());
    buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for v in values {
        push_string(buf, v);
    }
}

fn push_kv_array_i32(buf: &mut Vec<u8>, key: &str, values: &[i32]) {
    push_string(buf, key);
    buf.extend_from_slice(&9i32.to_le_bytes());
    buf.extend_from_slice(&5i32.to_le_bytes());
    buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

/// (name, wire type id, dims, n_bytes) — varied types AND a 4-byte scalar so
/// inter-tensor 32-alignment padding is genuinely exercised.
const TENSORS: &[(&str, i32, &[i64], usize)] = &[
    ("tok_embd.weight", 8, &[64, 4], 272), // Q8_0: 256 elems = 8 blocks * 34
    ("output_norm.weight", 0, &[1], 4),    // f32 scalar -> forces padding
    ("blk.0.attn_q.weight", 0, &[8, 2], 64), // f32 16 elems
    ("blk.0.attn_k.weight", 1, &[16], 32), // f16 16 elems
    ("blk.0.ffn_down.weight", 8, &[32, 3], 102), // Q8_0: 96 elems = 3 blocks * 34
    ("blk.0.ffn_up.weight", 0, &[4, 3], 48), // f32 12 elems
];

fn align_up(v: usize, a: usize) -> usize {
    v.div_ceil(a) * a
}

/// Distinct deterministic fill per tensor: byte j of tensor i = (i*31+j)%251.
fn fill(i: usize, n: usize) -> Vec<u8> {
    (0..n).map(|j| ((i * 31 + j) % 251) as u8).collect()
}

fn write_original(path: &Path) {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&(TENSORS.len() as u64).to_le_bytes());
    b.extend_from_slice(&8u64.to_le_bytes()); // kv count

    push_kv_string(&mut b, "general.architecture", "llama");
    push_kv_string(&mut b, "general.name", "split-merge-fixture");
    push_kv_u32(&mut b, "llama.block_count", 1);
    push_kv_f32(&mut b, "llama.rope.freq_base", 10000.0);
    push_kv_bool(&mut b, "fixture.flag", true);
    push_kv_u64(&mut b, "fixture.big", 1u64 << 40);
    push_kv_array_strings(&mut b, "tokenizer.ggml.tokens", &["a", "bb", "ccc"]);
    push_kv_array_i32(&mut b, "fixture.ids", &[3, -1, 4]);

    // Tensor table with the reader's align(prev_end) offset recurrence.
    let mut off = 0usize;
    for (name, ty, dims, n_bytes) in TENSORS {
        push_string(&mut b, name);
        b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in *dims {
            b.extend_from_slice(&d.to_le_bytes());
        }
        b.extend_from_slice(&ty.to_le_bytes());
        b.extend_from_slice(&(off as u64).to_le_bytes());
        off = align_up(off + n_bytes, 32);
    }

    // Aligned data section, zero padding, distinct fills.
    let data_start = align_up(b.len(), 32);
    b.resize(data_start, 0);
    for (i, (_, _, _, n_bytes)) in TENSORS.iter().enumerate() {
        let pos = b.len();
        b.resize(align_up(pos, 32).max(pos), 0);
        // relative offset must match the recurrence above
        b.extend_from_slice(&fill(i, *n_bytes));
        let end = b.len();
        b.resize(align_up(end, 32), 0);
    }
    // The loop above pads AFTER each tensor, which overshoots for the last one
    // only relative to the reader (harmless: reader checks end <= file_len).
    fs::write(path, &b).expect("write original fixture");
}

fn tmpdir(tag: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("camelid-split-merge-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create tmpdir");
    dir
}

fn round_trip(parts: usize) {
    let dir = tmpdir(&format!("rt{parts}"));
    let orig = dir.join("model.gguf");
    write_original(&orig);

    // The fixture must itself be a valid GGUF before we claim anything.
    let parsed = read_metadata(&orig).expect("original parses");
    assert_eq!(parsed.tensors.len(), TENSORS.len());

    let shards = split_gguf(&orig, parts, &dir).expect("split");
    assert_eq!(shards.len(), parts);
    for s in &shards {
        let f = read_metadata(s).expect("every shard parses");
        assert!(f.metadata.contains_key("split.no"));
        assert!(f.metadata.contains_key("split.count"));
        assert!(f.metadata.contains_key("split.tensors.count"));
    }

    let merged_path = dir.join("merged.gguf");
    let report = merge_shards(&shards, &merged_path).expect("merge");
    assert_eq!(report.tensors, TENSORS.len() as u64);
    assert_eq!(report.kv_dropped, 3);

    // Bar 1: whole-file byte identity with the original.
    let orig_bytes = fs::read(&orig).expect("read original");
    let merged_bytes = fs::read(&merged_path).expect("read merged");
    assert_eq!(
        orig_bytes.len(),
        merged_bytes.len(),
        "merged length differs from original"
    );
    assert!(
        orig_bytes == merged_bytes,
        "merged bytes differ from original"
    );

    // Bar 2 (semantic, in case bar 1 is ever relaxed): descriptors + metadata.
    let merged = read_metadata(&merged_path).expect("merged parses");
    assert_eq!(merged.metadata, parsed.metadata);
    let orig_names: Vec<_> = parsed.tensors.iter().map(|t| &t.name).collect();
    let merged_names: Vec<_> = merged.tensors.iter().map(|t| &t.name).collect();
    assert_eq!(orig_names, merged_names, "tensor order must be preserved");
    for (a, b) in parsed.tensors.iter().zip(&merged.tensors) {
        assert_eq!(a, b, "descriptor mismatch for {}", a.name);
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn split_merge_round_trip_two_shards_is_byte_identical() {
    round_trip(2);
}

#[test]
fn split_merge_round_trip_three_shards_is_byte_identical() {
    round_trip(3);
}

#[test]
fn split_merge_round_trip_one_tensor_per_shard_is_byte_identical() {
    // Regression: the group partitioner's must_break was off by one and could
    // not fill the tail groups of front-heavy inputs (e.g. n_parts == tensor
    // count), erroring on perfectly splittable files.
    round_trip(TENSORS.len());
}

#[test]
fn merge_refuses_output_aliasing_a_shard() {
    // Regression: File::create on an aliased output would truncate the shard
    // after validation but before its data is copied.
    let dir = tmpdir("alias");
    let orig = dir.join("model.gguf");
    write_original(&orig);
    let shards = split_gguf(&orig, 2, &dir).expect("split");
    let err = merge_shards(&shards, shards[1].as_path()).expect_err("aliased out must refuse");
    assert!(err.to_string().contains("aliases"), "{err}");
    // Both shards must be intact afterwards (nothing truncated).
    for s in &shards {
        read_metadata(s).expect("shard survives the refused merge");
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn merge_refuses_fewer_than_two_shards() {
    let dir = tmpdir("neg1");
    let orig = dir.join("model.gguf");
    write_original(&orig);
    let err = merge_shards(&[orig], dir.join("out.gguf").as_path())
        .expect_err("single shard must refuse");
    assert!(err.to_string().contains("at least 2"), "{err}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn merge_refuses_incoherent_split_stamps() {
    // Two copies of shard 1 presented as a 2-set: split.no says 0 twice.
    let dir = tmpdir("neg2");
    let orig = dir.join("model.gguf");
    write_original(&orig);
    let shards = split_gguf(&orig, 2, &dir).expect("split");
    let fake = dir.join("fake-00002-of-00002.gguf");
    fs::copy(&shards[0], &fake).expect("copy shard 1");
    let err = merge_shards(&[shards[0].clone(), fake], dir.join("out.gguf").as_path())
        .expect_err("split.no mismatch must refuse");
    assert!(err.to_string().contains("split.no"), "{err}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn discover_requires_every_sibling_on_disk() {
    let dir = tmpdir("neg3");
    let orig = dir.join("model.gguf");
    write_original(&orig);
    let shards = split_gguf(&orig, 3, &dir).expect("split");
    fs::remove_file(&shards[1]).expect("drop middle shard");
    let err = discover_shard_set(&shards[0]).expect_err("missing sibling must refuse");
    assert!(err.to_string().contains("missing shard"), "{err}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn discover_finds_all_siblings_and_merged_name() {
    let dir = tmpdir("disc");
    let orig = dir.join("model.gguf");
    write_original(&orig);
    let shards = split_gguf(&orig, 3, &dir).expect("split");
    // Discovery from the LAST shard, not the first.
    let set = discover_shard_set(&shards[2]).expect("discover");
    assert_eq!(set.paths, shards);
    assert_eq!(set.merged_name, "model.gguf");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn split_refuses_nesting_and_bad_part_counts() {
    let dir = tmpdir("neg4");
    let orig = dir.join("model.gguf");
    write_original(&orig);
    let shards = split_gguf(&orig, 2, &dir).expect("split");
    // Splitting a shard again must refuse (it carries split.* keys).
    let err = split_gguf(&shards[0], 2, &dir).expect_err("nested split must refuse");
    assert!(err.to_string().contains("split.*"), "{err}");
    let err = split_gguf(&orig, 1, &dir).expect_err("1 part must refuse");
    assert!(err.to_string().contains("at least 2"), "{err}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn shard_name_parsing_is_strict() {
    assert_eq!(
        parse_shard_name("m-00001-of-00002.gguf"),
        Some(("m".to_string(), 1, 2))
    );
    assert_eq!(
        parse_shard_name("Llama-3.3-70B-Instruct-Q8_0-00002-of-00002.gguf"),
        Some(("Llama-3.3-70B-Instruct-Q8_0".to_string(), 2, 2))
    );
    assert_eq!(parse_shard_name("m.gguf"), None);
    assert_eq!(parse_shard_name("m-1-of-2.gguf"), None); // not 5-digit
    assert_eq!(parse_shard_name("m-00000-of-00002.gguf"), None); // no == 0
    assert_eq!(parse_shard_name("m-00003-of-00002.gguf"), None); // no > total
    assert_eq!(parse_shard_name("m-00001-of-00002.bin"), None);
}
