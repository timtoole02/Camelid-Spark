//! Gemma 4 tensor binding — synthetic descriptor maps (no model file needed).
//!
//! Builds a minimal but complete 2-layer gemma4 tensor map and proves:
//! 1. a correct per-layer-type map binds (sliding layer with small head_dim,
//!    global layer with large head_dim, per-layer kv heads and FFN widths),
//! 2. a map whose K projection uses the WRONG per-layer kv head count fails
//!    with a typed shape error naming the layer,
//! 3. PLE tensors are optional (dense rows bind without them) and detected
//!    when present.

use std::collections::BTreeMap;
use std::path::PathBuf;

use camelid::gguf::{GgufFile, GgufMetadataValue, GgufTensorDescriptor, GgufTensorType};
use camelid::model::{Gemma4Binding, LlamaModelConfig};

const EMB: u64 = 64;
const HEADS: u64 = 4;
// Layer 0: sliding, head_dim 8, kv 2, ffn 128. Layer 1: global, head_dim 16, kv 1, ffn 256.
const HD: [u64; 2] = [8, 16];
const KV: [u64; 2] = [2, 1];
const FFN: [u64; 2] = [128, 256];

fn desc(name: &str, dims: &[u64]) -> GgufTensorDescriptor {
    GgufTensorDescriptor {
        name: name.to_string(),
        dimensions: dims.to_vec(),
        tensor_type: GgufTensorType::Q8_0,
        relative_offset: 0,
        absolute_offset: 0,
        n_bytes: 0,
    }
}

fn synthetic_gemma4(with_ple: bool) -> GgufFile {
    let mut metadata: BTreeMap<String, GgufMetadataValue> = BTreeMap::new();
    let mut set = |k: &str, v: GgufMetadataValue| {
        metadata.insert(k.to_string(), v);
    };
    set(
        "general.architecture",
        GgufMetadataValue::String("gemma4".into()),
    );
    set("gemma4.block_count", GgufMetadataValue::U32(2));
    set("gemma4.context_length", GgufMetadataValue::U32(4096));
    set(
        "gemma4.embedding_length",
        GgufMetadataValue::U32(EMB as u32),
    );
    set(
        "gemma4.attention.head_count",
        GgufMetadataValue::U32(HEADS as u32),
    );
    set(
        "gemma4.attention.head_count_kv",
        GgufMetadataValue::Array(vec![
            GgufMetadataValue::U32(KV[0] as u32),
            GgufMetadataValue::U32(KV[1] as u32),
        ]),
    );
    set(
        "gemma4.feed_forward_length",
        GgufMetadataValue::Array(vec![
            GgufMetadataValue::U32(FFN[0] as u32),
            GgufMetadataValue::U32(FFN[1] as u32),
        ]),
    );
    set(
        "gemma4.attention.key_length_swa",
        GgufMetadataValue::U32(HD[0] as u32),
    );
    set(
        "gemma4.attention.key_length",
        GgufMetadataValue::U32(HD[1] as u32),
    );
    set(
        "gemma4.attention.sliding_window_pattern",
        GgufMetadataValue::Array(vec![
            GgufMetadataValue::Bool(true),
            GgufMetadataValue::Bool(false),
        ]),
    );
    set(
        "gemma4.attention.sliding_window",
        GgufMetadataValue::U32(32),
    );
    set("gemma4.vocab_size", GgufMetadataValue::U32(256));
    if with_ple {
        set(
            "gemma4.embedding_length_per_layer_input",
            GgufMetadataValue::U32(16),
        );
    }

    let mut tensors = vec![
        desc("token_embd.weight", &[EMB, 256]),
        desc("output_norm.weight", &[EMB]),
    ];
    if with_ple {
        tensors.push(desc("per_layer_token_embd.weight", &[16 * 2, 256]));
        tensors.push(desc("per_layer_model_proj.weight", &[EMB, 16 * 2]));
        tensors.push(desc("per_layer_proj_norm.weight", &[16]));
    }
    for l in 0..2u64 {
        let hd = HD[l as usize];
        let kv = KV[l as usize];
        let ffn = FFN[l as usize];
        let t = |suffix: &str, dims: &[u64]| desc(&format!("blk.{l}.{suffix}.weight"), dims);
        tensors.extend([
            t("attn_norm", &[EMB]),
            t("attn_q", &[EMB, HEADS * hd]),
            t("attn_k", &[EMB, kv * hd]),
            t("attn_v", &[EMB, kv * hd]),
            t("attn_output", &[HEADS * hd, EMB]),
            t("attn_q_norm", &[hd]),
            t("attn_k_norm", &[hd]),
            t("post_attention_norm", &[EMB]),
            t("ffn_norm", &[EMB]),
            t("post_ffw_norm", &[EMB]),
            t("ffn_gate", &[EMB, ffn]),
            t("ffn_up", &[EMB, ffn]),
            t("ffn_down", &[ffn, EMB]),
        ]);
        if with_ple {
            tensors.extend([
                t("inp_gate", &[EMB, 16]),
                t("proj", &[16, EMB]),
                t("layer_output_scale", &[1]),
                t("post_norm", &[EMB]),
            ]);
        }
    }

    GgufFile {
        path: PathBuf::from("synthetic-gemma4-binding.gguf"),
        version: 3,
        tensor_count: tensors.len() as i64,
        metadata_count: metadata.len() as i64,
        alignment: 32,
        data_start_offset: 0,
        metadata,
        tensors,
    }
}

#[test]
fn per_layer_type_map_binds_dense() {
    let gguf = synthetic_gemma4(false);
    let config = LlamaModelConfig::from_gguf(&gguf).expect("config");
    let binding = Gemma4Binding::bind(&gguf, &config).expect("dense map must bind");
    assert_eq!(binding.layers.len(), 2);
    assert!(!binding.has_per_layer_embeddings());
    assert!(binding.output_is_tied_embedding);
}

#[test]
fn per_layer_type_map_binds_with_ple() {
    let gguf = synthetic_gemma4(true);
    let config = LlamaModelConfig::from_gguf(&gguf).expect("config");
    let binding = Gemma4Binding::bind(&gguf, &config).expect("PLE map must bind");
    assert!(binding.has_per_layer_embeddings());
    assert!(binding.layers.iter().all(|l| l.ple_output_scale.is_some()));
}

#[test]
fn wrong_per_layer_kv_width_fails_with_typed_shape_error() {
    let mut gguf = synthetic_gemma4(false);
    // Corrupt layer 1's K projection to layer 0's kv geometry (2 heads x 8 dim
    // instead of 1 head x 16 dim — same element count class, wrong shape).
    let k = gguf
        .tensors
        .iter_mut()
        .find(|t| t.name == "blk.1.attn_k.weight")
        .unwrap();
    k.dimensions = vec![EMB, 2 * 8 * 2];
    let config = LlamaModelConfig::from_gguf(&gguf).expect("config");
    let err = Gemma4Binding::bind(&gguf, &config).expect_err("wrong kv width must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("layer 1") && msg.contains("attention k"),
        "names the exact tensor: {msg}"
    );
}

#[test]
fn wrong_per_layer_ffn_width_fails_with_typed_shape_error() {
    let mut gguf = synthetic_gemma4(false);
    // Give layer 0 the FFN width of layer 1 — per-layer validation must catch it.
    for name in [
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
    ] {
        let t = gguf.tensors.iter_mut().find(|t| t.name == name).unwrap();
        t.dimensions = if name.ends_with("ffn_down.weight") {
            vec![FFN[1], EMB]
        } else {
            vec![EMB, FFN[1]]
        };
    }
    let config = LlamaModelConfig::from_gguf(&gguf).expect("config");
    let err = Gemma4Binding::bind(&gguf, &config).expect_err("wrong ffn width must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("layer 0") && msg.contains("ffn"),
        "names the exact tensor: {msg}"
    );
}
