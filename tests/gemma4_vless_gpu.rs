//! V-less GPU layer vs CPU on REAL 12B weights (env-gated).
//!
//! Loads layer 5 of the 12B GGUF — a real V-less full-attention layer (no
//! `attn_v` tensor, kv_heads 1, head_dim 512, hidden 3840, ffn 15360) — onto
//! the GPU via `Gemma4ResidentLayer::from_wire` and compares one full layer
//! forward (attention + FFN, position 0) against an f32 CPU reference computed
//! from the dequantized tensors. Real weight magnitudes exercise the norm/tanh
//! ranges that synthetic tests cannot (the historical MSL tanh-overflow bug
//! only appeared at real scale).
//!
//! Run: `CAMELID_GEMMA4_GGUF=/path/gemma-4-12b-it-Q8_0.gguf \
//!       cargo test --release --test gemma4_vless_gpu -- --nocapture`

// The whole file is macOS-only: the imports feed a Metal-gated test, and on
// other targets they would trip -D unused-imports.
#![cfg(target_os = "macos")]
#![allow(clippy::needless_range_loop)]

use std::path::PathBuf;

use camelid::gguf::read_metadata;
use camelid::model::{Gemma4Binding, LlamaModelConfig};
use camelid::tensor::TensorStore;

#[test]
fn vless_gpu_layer_matches_cpu_on_real_12b_weights() {
    let Some(path) = std::env::var_os("CAMELID_GEMMA4_GGUF").map(PathBuf::from) else {
        eprintln!("SKIP vless gpu layer: set CAMELID_GEMMA4_GGUF to the 12B GGUF");
        return;
    };
    let gguf = read_metadata(&path).expect("gguf");
    let config = LlamaModelConfig::from_gguf(&gguf).expect("config");
    let g = config.gemma4.as_ref().expect("gemma4");
    let binding = Gemma4Binding::bind(&gguf, &config).expect("binding");
    // Find the first V-less layer; skip the row if there is none (this test is
    // specifically about the 12B-class geometry).
    let Some(l) = binding.layers.iter().position(|l| l.attn_v.is_none()) else {
        eprintln!("SKIP vless gpu layer: row has attn_v on every layer");
        return;
    };
    let lb = &binding.layers[l];
    let hidden = config.embedding_length as usize;
    let heads = config.attention_head_count as usize;
    let kv_heads = g.kv_heads_at(l) as usize;
    let head_dim = g.head_dim_at(l) as usize;
    let ffn_dim = g.ffn_length_at(l) as usize;
    let eps = config.rms_norm_epsilon;
    let half = head_dim / 2;
    eprintln!(
        "layer {l}: hidden {hidden}, heads {heads}, kv {kv_heads}, head_dim {head_dim}, ffn {ffn_dim}"
    );

    let store = TensorStore::open(&path, &gguf);
    let f32t = |name: &str| -> Vec<f32> {
        store
            .load_cpu_f32(name)
            .unwrap_or_else(|e| panic!("load {name}: {e}"))
            .data
    };
    // Wire bytes straight from the file for the GPU buffers.
    let mmap = camelid::wire_mmap::GgufWireMmap::map(&path).expect("mmap");
    let wire = |name: &str| -> Vec<u8> {
        let d = store.descriptor(name).expect("desc");
        mmap.bytes(d.absolute_offset, d.n_bytes as usize)
            .expect("wire bytes")
            .to_vec()
    };

    let attn_norm = f32t(&lb.attn_norm.name);
    let q_norm = f32t(&lb.attn_q_norm.name);
    let k_norm = f32t(
        &lb.attn_k_norm
            .as_ref()
            .expect("12B layers bind attn_k_norm")
            .name,
    );
    let post_attn = f32t(&lb.post_attention_norm.name);
    let ffn_norm = f32t(&lb.ffn_norm.name);
    let post_ffw = f32t(&lb.post_ffw_norm.name);
    // (attn_q is loaded for the GPU side only — at position 0 the CPU reference
    // never needs Q: softmax over one position makes the attention weight 1.0.)
    let k_w = f32t(&lb.attn_k.as_ref().expect("12B layers bind attn_k").name);
    let o_w = f32t(&lb.attn_output.name);
    let gate_w = f32t(&lb.ffn_gate.name);
    let up_w = f32t(&lb.ffn_up.name);
    let down_w = f32t(&lb.ffn_down.name);

    // A plausible input: the embedding row of token 2 (BOS), gemma-scaled.
    let embd = f32t("token_embd.weight");
    let h_in: Vec<f32> = embd[2 * hidden..3 * hidden]
        .iter()
        .map(|v| v * (hidden as f32).sqrt())
        .collect();

    // ---- CPU reference at position 0 (rope is identity; softmax over 1) ----
    let rms = |x: &[f32], w: Option<&[f32]>| -> Vec<f32> {
        let mss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let inv = (mss + eps).powf(-0.5);
        (0..x.len())
            .map(|i| x[i] * inv * w.map_or(1.0, |w| w[i]))
            .collect()
    };
    let matvec = |w: &[f32], x: &[f32], in_dim: usize, out_dim: usize| -> Vec<f32> {
        (0..out_dim)
            .map(|o| {
                w[o * in_dim..(o + 1) * in_dim]
                    .iter()
                    .zip(x)
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect()
    };
    let gelu = |x: f32| -> f32 {
        const C: f32 = 0.797_884_6;
        0.5 * x * (1.0 + (C * (x + 0.044_715 * x * x * x)).tanh())
    };
    let normf = rms(&h_in, Some(&attn_norm));
    let k_raw = matvec(&k_w, &normf, hidden, kv_heads * head_dim);
    let mut ctx = vec![0.0f32; heads * head_dim];
    {
        // Per-head normed q is irrelevant at pos 0 beyond producing weights of
        // 1.0 (softmax over a single position) — ctx = V for the mapped kv head.
        let mut v = vec![0.0f32; kv_heads * head_dim];
        for h in 0..kv_heads {
            let n = rms(&k_raw[h * head_dim..(h + 1) * head_dim], None);
            v[h * head_dim..(h + 1) * head_dim].copy_from_slice(&n);
        }
        let group = heads / kv_heads;
        for h in 0..heads {
            let kvh = h / group;
            ctx[h * head_dim..(h + 1) * head_dim]
                .copy_from_slice(&v[kvh * head_dim..(kvh + 1) * head_dim]);
        }
    }
    let o = matvec(&o_w, &ctx, heads * head_dim, hidden);
    let on = rms(&o, Some(&post_attn));
    let h_mid: Vec<f32> = h_in.iter().zip(&on).map(|(a, b)| a + b).collect();
    let xn = rms(&h_mid, Some(&ffn_norm));
    let gate = matvec(&gate_w, &xn, hidden, ffn_dim);
    let up = matvec(&up_w, &xn, hidden, ffn_dim);
    let act: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| gelu(*g) * u).collect();
    let down = matvec(&down_w, &act, ffn_dim, hidden);
    let dn = rms(&down, Some(&post_ffw));
    let want: Vec<f32> = h_mid.iter().zip(&dn).map(|(a, b)| a + b).collect();

    // ---- GPU ----
    let layer = camelid::metal::Gemma4ResidentLayer::from_wire(
        camelid::metal::GemmaWireFmt::Q8_0,
        attn_norm.clone(),
        q_norm.clone(),
        k_norm.clone(),
        post_attn.clone(),
        ffn_norm.clone(),
        post_ffw.clone(),
        &wire(&lb.attn_q.name),
        &wire(&lb.attn_k.as_ref().expect("12B layers bind attn_k").name),
        None, // V-less
        &wire(&lb.attn_output.name),
        &wire(&lb.ffn_gate.name),
        &wire(&lb.ffn_up.name),
        &wire(&lb.ffn_down.name),
        heads,
        kv_heads,
        head_dim,
        ffn_dim,
        eps,
    )
    .expect("metal layer");
    let cos_t = vec![1.0f32; half]; // position 0: identity rotation
    let sin_t = vec![0.0f32; half];
    let cache = vec![0.0f32; kv_heads * 8 * head_dim];
    let got = camelid::metal::try_gemma4_layer(
        &layer, &h_in, &cos_t, &sin_t, &cache, &cache, 8, 0, 1, 0, 1.0, true,
    )
    .expect("gpu layer");

    // GPU Q8-quantizes activations per GEMV (matching the engine); the CPU ref
    // here is plain f32 over dequantized weights, so allow a small tolerance
    // relative to the hidden-state magnitude (~70-200 L2 at this depth).
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let scale_ref = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    for (a, b) in got.iter().zip(&want) {
        max_abs = max_abs.max((a - b).abs());
        max_rel = max_rel.max((a - b).abs() / scale_ref.max(1.0));
    }
    eprintln!("max abs diff {max_abs:.5}, max rel-to-peak {max_rel:.6}");
    assert!(
        max_rel < 5.0e-3,
        "V-less GPU layer diverges from CPU reference: max abs {max_abs}, rel {max_rel}"
    );
}
