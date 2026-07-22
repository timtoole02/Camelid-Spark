//! Gemma 4 metadata parsing — row-aware, exact-row checked.
//!
//! Two layers of coverage:
//! 1. Synthetic GGUF metadata (always run): the per-layer keys that vary across
//!    real rows — `attention.sliding_window_pattern` (E2B is 4:1, NOT the 5:1
//!    formula), per-layer `feed_forward_length` (E2B), per-layer
//!    `attention.head_count_kv` (12B) — plus the fail-closed blockers for the
//!    `gemma4-assistant` MTP architecture and gemma4 MoE rows.
//! 2. Real-file snapshots (skipped unless `CAMELID_GEMMA4_GGUF` is set): the
//!    full expected metadata for each known exact row, keyed by
//!    (block_count, embedding_length) so the wrong expectations can never be
//!    applied to the wrong file.

use std::collections::BTreeMap;
use std::path::PathBuf;

use camelid::gguf::{GgufFile, GgufMetadataValue, GgufTensorDescriptor, GgufTensorType};
use camelid::model::{Gemma4Binding, LlamaModelConfig};
use camelid::BackendError;

fn synthetic_gemma4_gguf(block_count: u32) -> GgufFile {
    let mut metadata: BTreeMap<String, GgufMetadataValue> = BTreeMap::new();
    let mut set = |k: &str, v: GgufMetadataValue| {
        metadata.insert(k.to_string(), v);
    };
    set(
        "general.architecture",
        GgufMetadataValue::String("gemma4".into()),
    );
    set("gemma4.block_count", GgufMetadataValue::U32(block_count));
    set("gemma4.context_length", GgufMetadataValue::U32(131072));
    set("gemma4.embedding_length", GgufMetadataValue::U32(1536));
    set("gemma4.feed_forward_length", GgufMetadataValue::U32(6144));
    set("gemma4.attention.head_count", GgufMetadataValue::U32(8));
    set("gemma4.attention.head_count_kv", GgufMetadataValue::U32(1));
    set("gemma4.attention.key_length", GgufMetadataValue::U32(512));
    set(
        "gemma4.attention.key_length_swa",
        GgufMetadataValue::U32(256),
    );
    set(
        "gemma4.attention.sliding_window",
        GgufMetadataValue::U32(512),
    );
    set("gemma4.vocab_size", GgufMetadataValue::U32(262144));
    GgufFile {
        path: PathBuf::from("synthetic-gemma4.gguf"),
        version: 3,
        tensor_count: 0,
        metadata_count: metadata.len() as i64,
        alignment: 32,
        data_start_offset: 0,
        metadata,
        tensors: Vec::new(),
    }
}

#[test]
fn sliding_window_pattern_array_is_authoritative() {
    // E2B's real schedule is 4:1 (full attention at indices 4, 9, 14, ...),
    // which the 5:1 fallback formula would get WRONG. The GGUF array must win.
    let mut gguf = synthetic_gemma4_gguf(10);
    let pattern = vec![true, true, true, true, false, true, true, true, true, false];
    gguf.metadata.insert(
        "gemma4.attention.sliding_window_pattern".into(),
        GgufMetadataValue::Array(
            pattern
                .iter()
                .map(|&b| GgufMetadataValue::Bool(b))
                .collect(),
        ),
    );
    let config = LlamaModelConfig::from_gguf(&gguf).expect("config");
    let g = config.gemma4.expect("gemma4 metadata");
    assert_eq!(g.layer_is_sliding, pattern);
    // The 5:1 formula would have made layer 4 sliding — prove we did not use it.
    assert!(!g.is_sliding_layer(4));
}

#[test]
fn sliding_window_pattern_wrong_length_falls_back_to_formula() {
    let mut gguf = synthetic_gemma4_gguf(10);
    gguf.metadata.insert(
        "gemma4.attention.sliding_window_pattern".into(),
        GgufMetadataValue::Array(vec![GgufMetadataValue::Bool(true); 4]),
    );
    let config = LlamaModelConfig::from_gguf(&gguf).expect("config");
    let g = config.gemma4.expect("gemma4 metadata");
    assert_eq!(g.layer_is_sliding.len(), 10);
    // Formula: every 6th layer full, final layer forced full.
    assert!(!g.is_sliding_layer(5));
    assert!(!g.is_sliding_layer(9));
}

#[test]
fn per_layer_feed_forward_length_array_parses() {
    // E2B shape: 15 early layers at 6144, the rest at 12288.
    let mut gguf = synthetic_gemma4_gguf(6);
    gguf.metadata.remove("gemma4.feed_forward_length");
    gguf.metadata.insert(
        "gemma4.feed_forward_length".into(),
        GgufMetadataValue::Array(
            [6144u32, 6144, 6144, 12288, 12288, 12288]
                .iter()
                .map(|&v| GgufMetadataValue::U32(v))
                .collect(),
        ),
    );
    let config = LlamaModelConfig::from_gguf(&gguf).expect("config");
    let g = config.gemma4.as_ref().expect("gemma4 metadata");
    assert_eq!(g.ffn_length_at(0), 6144);
    assert_eq!(g.ffn_length_at(5), 12288);
    assert_eq!(g.max_ffn_length(), 12288);
    // The config scalar holds the max for generic sizing.
    assert_eq!(config.feed_forward_length, 12288);
}

#[test]
fn per_layer_head_count_kv_array_parses() {
    // 12B shape: 8 KV heads on sliding layers, 1 on global layers.
    let mut gguf = synthetic_gemma4_gguf(6);
    gguf.metadata.remove("gemma4.attention.head_count_kv");
    gguf.metadata.insert(
        "gemma4.attention.head_count_kv".into(),
        GgufMetadataValue::Array(
            [8u32, 8, 8, 8, 8, 1]
                .iter()
                .map(|&v| GgufMetadataValue::U32(v))
                .collect(),
        ),
    );
    let config = LlamaModelConfig::from_gguf(&gguf).expect("config");
    let g = config.gemma4.as_ref().expect("gemma4 metadata");
    assert_eq!(g.kv_heads_at(0), 8);
    assert_eq!(g.kv_heads_at(5), 1);
    assert_eq!(config.attention_head_count_kv, 8);
}

#[test]
fn diffusiongemma_architecture_fails_closed() {
    // DiffusionGemma reuses the Gemma 4 26B-A4B MoE foundation but is a discrete
    // block-diffusion encoder-decoder, not an autoregressive model. Camelid must
    // fail closed by name on any `*diffusion*` architecture spelling rather than
    // mis-binding the shared gemma4 tensors.
    // "diffusion-gemma" is the ground-truth spelling: it is the actual
    // general.architecture value in the tracked unsloth Q4_K_M GGUF
    // (llama.cpp arch string, see docs/recon/DIFFUSIONGEMMA_RECON.md).
    for arch in [
        "diffusion-gemma",
        "diffusion_gemma",
        "diffusiongemma",
        "gemma-diffusion",
    ] {
        let mut gguf = synthetic_gemma4_gguf(4);
        gguf.metadata.insert(
            "general.architecture".into(),
            GgufMetadataValue::String(arch.into()),
        );
        let err = LlamaModelConfig::from_gguf(&gguf)
            .err()
            .unwrap_or_else(|| panic!("DiffusionGemma arch {arch} must fail closed"));
        match err {
            BackendError::UnsupportedModelArchitecture(msg) => {
                assert!(msg.contains("DiffusionGemma"), "names the model: {msg}");
                assert!(msg.contains("blocked"), "states it is blocked: {msg}");
                assert!(
                    msg.contains("diffusion") || msg.contains("autoregressive"),
                    "explains the paradigm mismatch: {msg}"
                );
            }
            other => panic!("expected UnsupportedModelArchitecture for {arch}, got {other:?}"),
        }
    }
}

#[test]
fn gemma4_assistant_mtp_architecture_fails_closed() {
    let mut gguf = synthetic_gemma4_gguf(4);
    gguf.metadata.insert(
        "general.architecture".into(),
        GgufMetadataValue::String("gemma4-assistant".into()),
    );
    let err = LlamaModelConfig::from_gguf(&gguf).expect_err("MTP head must fail closed");
    match err {
        BackendError::UnsupportedModelArchitecture(msg) => {
            assert!(msg.contains("gemma4-assistant"), "names the arch: {msg}");
            assert!(msg.contains("blocked"), "states it is blocked: {msg}");
            assert!(
                msg.contains("attn_k") || msg.contains("KV"),
                "names the missing tensor contract: {msg}"
            );
        }
        other => panic!("expected UnsupportedModelArchitecture, got {other:?}"),
    }
}

#[test]
fn gemma4_moe_row_fails_closed_with_typed_blocker() {
    // The 26B A4B row advertises gemma4.expert_count; the dense-only binding
    // must reject it by name, not with a generic missing-tensor error.
    let mut gguf = synthetic_gemma4_gguf(6);
    gguf.metadata
        .insert("gemma4.expert_count".into(), GgufMetadataValue::U32(64));
    gguf.metadata
        .insert("gemma4.expert_used_count".into(), GgufMetadataValue::U32(4));
    let config = LlamaModelConfig::from_gguf(&gguf).expect("config parses");
    let err = Gemma4Binding::bind(&gguf, &config).expect_err("MoE row must fail closed");
    match err {
        BackendError::UnsupportedModelArchitecture(msg) => {
            assert!(msg.contains("MoE"), "names MoE: {msg}");
            assert!(
                msg.contains("expert_count=64"),
                "names the row shape: {msg}"
            );
            assert!(msg.contains("blocked"), "states it is blocked: {msg}");
        }
        other => panic!("expected UnsupportedModelArchitecture, got {other:?}"),
    }
}

#[test]
fn gemma4_moe_tensors_without_metadata_fail_closed() {
    let mut gguf = synthetic_gemma4_gguf(6);
    gguf.tensors.push(GgufTensorDescriptor {
        name: "blk.0.ffn_gate_inp.weight".into(),
        dimensions: vec![1536, 64],
        tensor_type: GgufTensorType::F32,
        relative_offset: 0,
        absolute_offset: 0,
        n_bytes: 1536 * 64 * 4,
    });
    let config = LlamaModelConfig::from_gguf(&gguf).expect("config parses");
    let err = Gemma4Binding::bind(&gguf, &config).expect_err("router tensor must fail closed");
    match err {
        BackendError::UnsupportedModelArchitecture(msg) => {
            assert!(msg.contains("ffn_gate_inp"), "names the tensor: {msg}");
        }
        other => panic!("expected UnsupportedModelArchitecture, got {other:?}"),
    }
}

// --- real-file row snapshots -------------------------------------------------

struct RowExpectation {
    row: &'static str,
    block_count: u32,
    embedding_length: u32,
    heads: u32,
    context_length: u32,
    sliding_window: u32,
    shared_kv_layers: u32,
    ple_dim: u32,
    softcap: Option<f32>,
    head_dim_sliding: u32,
    head_dim_global: u32,
    /// (layer index, expected kv heads) probes.
    kv_probes: &'static [(usize, u32)],
    /// (layer index, expected ffn length) probes.
    ffn_probes: &'static [(usize, u32)],
    /// Indices of the first full-attention (global) layers.
    first_global_layers: &'static [usize],
}

const ROWS: &[RowExpectation] = &[
    RowExpectation {
        row: "gemma-4-E2B-it-Q8_0",
        block_count: 35,
        embedding_length: 1536,
        heads: 8,
        context_length: 131072,
        sliding_window: 512,
        shared_kv_layers: 20,
        ple_dim: 256,
        softcap: Some(30.0),
        head_dim_sliding: 256,
        head_dim_global: 512,
        kv_probes: &[(0, 1), (34, 1)],
        ffn_probes: &[(0, 6144), (14, 6144), (15, 12288), (34, 12288)],
        // E2B is 4:1 — the 5:1 formula would put the first global layer at 5.
        first_global_layers: &[4, 9, 14],
    },
    RowExpectation {
        row: "gemma-4-E4B-it-Q8_0",
        block_count: 42,
        embedding_length: 2560,
        heads: 8,
        context_length: 131072,
        sliding_window: 512,
        shared_kv_layers: 18,
        ple_dim: 256,
        softcap: Some(30.0),
        head_dim_sliding: 256,
        head_dim_global: 512,
        kv_probes: &[(0, 2), (41, 2)],
        ffn_probes: &[(0, 10240), (41, 10240)],
        first_global_layers: &[5, 11, 17],
    },
    RowExpectation {
        row: "gemma-4-12b-it-Q8_0",
        block_count: 48,
        embedding_length: 3840,
        heads: 16,
        context_length: 262144,
        sliding_window: 1024,
        shared_kv_layers: 0,
        ple_dim: 0,
        softcap: Some(30.0),
        head_dim_sliding: 256,
        head_dim_global: 512,
        // 12B: 8 kv heads on sliding layers, 1 on global layers.
        kv_probes: &[(0, 8), (5, 1), (11, 1), (47, 1)],
        ffn_probes: &[(0, 15360), (47, 15360)],
        first_global_layers: &[5, 11, 17],
    },
];

#[test]
fn real_row_metadata_snapshot_matches() {
    let Some(path) = std::env::var_os("CAMELID_GEMMA4_GGUF").map(PathBuf::from) else {
        eprintln!("SKIP real_row_metadata_snapshot: set CAMELID_GEMMA4_GGUF");
        return;
    };
    let gguf = camelid::gguf::read_metadata(&path).expect("read gguf");
    let config = LlamaModelConfig::from_gguf(&gguf).expect("config");
    let g = config.gemma4.as_ref().expect("gemma4 metadata");

    let row = ROWS
        .iter()
        .find(|r| {
            r.block_count == config.block_count && r.embedding_length == config.embedding_length
        })
        .unwrap_or_else(|| {
            panic!(
                "no known exact row for block_count={} embedding_length={} — add the \
                 row expectation before claiming anything about this file",
                config.block_count, config.embedding_length
            )
        });
    eprintln!("row identified: {}", row.row);

    assert_eq!(config.attention_head_count, row.heads, "{}", row.row);
    assert_eq!(config.context_length, row.context_length, "{}", row.row);
    assert_eq!(g.sliding_window, row.sliding_window, "{}", row.row);
    assert_eq!(g.num_kv_shared_layers, row.shared_kv_layers, "{}", row.row);
    assert_eq!(g.per_layer_input_dim, row.ple_dim, "{}", row.row);
    assert_eq!(g.final_logit_softcapping, row.softcap, "{}", row.row);
    assert_eq!(g.head_dim_sliding, row.head_dim_sliding, "{}", row.row);
    assert_eq!(g.head_dim_global, row.head_dim_global, "{}", row.row);
    assert_eq!(config.vocab_size, Some(262144), "{}", row.row);
    for &(l, kv) in row.kv_probes {
        assert_eq!(g.kv_heads_at(l), kv, "{} layer {l} kv heads", row.row);
    }
    for &(l, ffn) in row.ffn_probes {
        assert_eq!(g.ffn_length_at(l), ffn, "{} layer {l} ffn", row.row);
    }
    for &l in row.first_global_layers {
        assert!(
            !g.is_sliding_layer(l),
            "{} layer {l} must be global",
            row.row
        );
    }
    // Every non-global probe layer before the first global one is sliding.
    assert!(g.is_sliding_layer(0), "{} layer 0 must be sliding", row.row);
}

/// Recon probe: confirm the mixed-quant Q4_0 gemma4 file opens and dump the quant type
/// of every tensor class. Set CAMELID_GEMMA4_Q4_GGUF to the file. Read-only; no asserts.
#[test]
#[ignore = "set CAMELID_GEMMA4_Q4_GGUF to the mixed Q4_0 gemma4 gguf"]
fn dump_q4_0_tensor_types() {
    let path = match std::env::var("CAMELID_GEMMA4_Q4_GGUF") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("skip: set CAMELID_GEMMA4_Q4_GGUF");
            return;
        }
    };
    let gguf = camelid::gguf::read_metadata(std::path::Path::new(&path)).expect("read_metadata");
    let normalize = |name: &str| -> String {
        name.split('.')
            .map(|seg| {
                if !seg.is_empty() && seg.bytes().all(|b| b.is_ascii_digit()) {
                    "*"
                } else {
                    seg
                }
            })
            .collect::<Vec<_>>()
            .join(".")
    };
    let mut by_class: BTreeMap<String, std::collections::BTreeSet<String>> = BTreeMap::new();
    let mut type_counts: BTreeMap<String, usize> = BTreeMap::new();
    for t in &gguf.tensors {
        by_class
            .entry(normalize(&t.name))
            .or_default()
            .insert(format!("{:?}", t.tensor_type));
        *type_counts
            .entry(format!("{:?}", t.tensor_type))
            .or_default() += 1;
    }
    eprintln!("=== {} tensors; distinct types ===", gguf.tensors.len());
    for (ty, c) in &type_counts {
        eprintln!("  {ty}: {c}");
    }
    eprintln!("=== tensor class -> type(s) ===");
    for (cls, tys) in &by_class {
        eprintln!("  {cls} -> {tys:?}");
    }
}
