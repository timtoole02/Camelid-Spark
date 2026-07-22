//! Gemma 4 prefill forward pass — correctness-first reference, validated against
//! the llama.cpp oracle. Feeds the 6-token prompt "The capital of France is" and
//! checks the last-position argmax == 9079 (" Paris").
//!
//! Plain f32 math for transparency (no engine fast-paths). Skipped unless
//! `CAMELID_GEMMA4_GGUF` is set. RAM-aware: dequantizes only the 6 needed rows of
//! the giant embedding tables and loads per-layer weights on demand.
//!
//! Run: `CAMELID_GEMMA4_GGUF=/path/gemma-4-E4B-it-Q8_0.gguf \
//!       cargo test --test gemma4_forward -- --nocapture`

// Reference test: clarity over clippy idioms. The index loops mirror the math 1:1,
// the float constants are full-precision to match llama.cpp, and PROMPT_TOKENS
// documents the prompt even when only the count is used.
#![allow(clippy::needless_range_loop, clippy::excessive_precision, dead_code)]

use std::path::PathBuf;

use camelid::gguf::read_metadata;
use camelid::model::{Gemma4Binding, LlamaModelConfig};
use camelid::tensor::TensorStore;

const ORACLE_FIRST_TOKEN: u32 = 9079; // " Paris"
                                      // Prompt "The capital of France is" (6 tokens) + llama.cpp's greedy continuation.
                                      // Teacher-forcing this sequence proves multi-token generation: each position's
                                      // argmax must predict the next token (positions >= 5, the generated region).
const PROMPT_TOKENS: &[u32] = &[2, 818, 5279, 529, 7001, 563];
const FULL_SEQ: &[u32] = &[
    2, 818, 5279, 529, 7001, 563, // prompt
    9079, 236761, 108, 1018, 14977, 53121, 2900, 563, 506, 5279, 529, 7001, // continuation
];
const PROMPT_LEN: usize = 6;

// --- f32 helpers -----------------------------------------------------------

/// y[o] = sum_i W[o*in + i] * x[i]. GGUF stores a [in, out] descriptor as data
/// laid out row-major over (out, in) with `in` contiguous — i.e. W is `out` rows
/// of length `in`.
fn matvec(w: &[f32], x: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    assert_eq!(w.len(), in_dim * out_dim);
    assert_eq!(x.len(), in_dim);
    (0..out_dim)
        .map(|o| {
            let row = &w[o * in_dim..(o + 1) * in_dim];
            row.iter().zip(x).map(|(a, b)| a * b).sum()
        })
        .collect()
}

fn rms_norm(x: &[f32], weight: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let mss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = (mss + eps).powf(-0.5);
    match weight {
        Some(w) => x.iter().zip(w).map(|(v, ww)| v * inv * ww).collect(),
        None => x.iter().map(|v| v * inv).collect(),
    }
}

fn gelu_tanh(x: f32) -> f32 {
    const C: f32 = 0.797_884_56;
    0.5 * x * (1.0 + (C * (x + 0.044_715 * x * x * x)).tanh())
}

fn add_into(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += s;
    }
}

/// Apply RoPE in place to a [heads, head_dim] flattened vector at absolute
/// `position`, using the GPT-NeoX/HF pairing (first half / second half) with the
/// given theta. rope_dim == head_dim here (GGUF rope.dimension_count matches).
fn apply_rope(vec: &mut [f32], heads: usize, head_dim: usize, position: usize, theta: f32) {
    let half = head_dim / 2;
    for h in 0..heads {
        let base = h * head_dim;
        for i in 0..half {
            let freq = (theta).powf(-(2.0 * i as f32) / head_dim as f32);
            let angle = position as f32 * freq;
            let (s, c) = angle.sin_cos();
            let a = vec[base + i];
            let b = vec[base + half + i];
            vec[base + i] = a * c - b * s;
            vec[base + half + i] = b * c + a * s;
        }
    }
}

fn load_f32(store: &TensorStore, name: &str) -> (Vec<f32>, Vec<usize>) {
    let t = store
        .load_cpu_f32(name)
        .unwrap_or_else(|e| panic!("load {name}: {e}"));
    (t.data, t.shape.dims.clone())
}

#[test]
fn gemma4_prefill_matches_oracle() {
    let Some(path) = std::env::var_os("CAMELID_GEMMA4_GGUF").map(PathBuf::from) else {
        eprintln!("SKIP gemma4_prefill: set CAMELID_GEMMA4_GGUF");
        return;
    };
    let gguf = read_metadata(&path).expect("read_metadata");
    let config = LlamaModelConfig::from_gguf(&gguf).expect("config");
    let g = config.gemma4.as_ref().expect("gemma4 metadata").clone();
    let binding = Gemma4Binding::bind(&gguf, &config).expect("bind");
    let store = TensorStore::open(&path, &gguf);

    let hidden = config.embedding_length as usize; // 2560
    let n_layers = config.block_count as usize; // 42
    let heads = config.attention_head_count as usize; // 8
    let kv_heads = config.attention_head_count_kv as usize; // 2
    let ple_dim = g.per_layer_input_dim as usize; // 256
    let eps = config.rms_norm_epsilon;
    let seq = FULL_SEQ.len();
    let embed_scale = (hidden as f32).sqrt();
    let ple_embed_scale = (ple_dim as f32).sqrt();

    // --- embeddings: dequantize just the rows we need ----------------------
    let tok_blocks = store
        .load_q8_0_blocks(&binding.token_embedding.name)
        .expect("token_embd blocks");
    let ple_blocks = binding
        .per_layer_token_embd
        .as_ref()
        .map(|d| store.load_q8_0_blocks(&d.name).expect("ple embd blocks"));

    // hidden states [seq][hidden], scaled embeddings
    // GGUF token_embd is token-major in raw element order (each token's `hidden`
    // values are contiguous), so index by element offset, not dequantize_row.
    let ple_total = n_layers * ple_dim;
    let mut hs: Vec<Vec<f32>> = FULL_SEQ
        .iter()
        .map(|&t| {
            let row = tok_blocks
                .dequantize_elements(t as usize * hidden, hidden)
                .expect("tok row");
            row.iter().map(|v| v * embed_scale).collect()
        })
        .collect();

    // --- per-layer input embeddings (PLE), token-identity component --------
    // per_layer_token_embd[token] is [n_layers * ple_dim], scaled by sqrt(ple_dim).
    let token_identity: Vec<Vec<f32>> = FULL_SEQ
        .iter()
        .map(|&t| {
            ple_blocks
                .as_ref()
                .map(|b| {
                    b.dequantize_elements(t as usize * ple_total, ple_total)
                        .expect("ple row")
                        .iter()
                        .map(|v| v * ple_embed_scale)
                        .collect()
                })
                .unwrap_or_default()
        })
        .collect();

    // context component: per_layer_model_proj(embed_scaled) * hidden^-0.5, then
    // reshape [n_layers, ple_dim] and per_layer_proj_norm (RMSNorm).
    let per_layer_input: Vec<Vec<Vec<f32>>> = if let (Some(proj), Some(pnorm)) = (
        binding.per_layer_model_proj.as_ref(),
        binding.per_layer_proj_norm.as_ref(),
    ) {
        let (proj_w, proj_dims) = load_f32(&store, &proj.name); // [hidden, n_layers*ple_dim]
        let proj_out = proj_dims[1]; // n_layers * ple_dim
        let (pnorm_w, _) = load_f32(&store, &pnorm.name); // [ple_dim]
        let proj_scale = (hidden as f32).powf(-0.5);
        (0..seq)
            .map(|t| {
                let ctx = matvec(&proj_w, &hs[t], hidden, proj_out);
                (0..n_layers)
                    .map(|l| {
                        let ctx_l: Vec<f32> = (0..ple_dim)
                            .map(|d| ctx[l * ple_dim + d] * proj_scale)
                            .collect();
                        let ctx_n = rms_norm(&ctx_l, Some(&pnorm_w), eps);
                        let ti = &token_identity[t][l * ple_dim..(l + 1) * ple_dim];
                        ctx_n
                            .iter()
                            .zip(ti)
                            .map(|(a, b)| (a + b) * std::f32::consts::FRAC_1_SQRT_2)
                            .collect()
                    })
                    .collect()
            })
            .collect()
    } else {
        vec![]
    };

    // --- decoder layers ----------------------------------------------------
    let sum_last = |hs: &[Vec<f32>]| hs.iter().map(|r| r.iter().sum::<f32>()).sum::<f32>();
    eprintln!(
        "inp_scaled sum (all tokens) = {:.4}  (llama.cpp ref ~ -90.71 for last row)",
        sum_last(&hs)
    );

    // Gemma 4 passes scaling=1.0 to the softmax; the query scale is baked into the
    // learned q_norm/k_norm weights (no extra 1/sqrt(head_dim)).
    let q_scale = |_hd: usize| 1.0f32;
    let n_run = std::env::var("LAYERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(n_layers);
    // Cross-layer KV sharing: layers >= first_kv_shared reuse the K/V of the last
    // same-type (sliding/full) layer before first_kv_shared, instead of their own.
    let first_kv_shared = n_layers - g.num_kv_shared_layers as usize;
    let mut shared_k_sliding: Option<Vec<Vec<f32>>> = None;
    let mut shared_v_sliding: Option<Vec<Vec<f32>>> = None;
    let mut shared_k_full: Option<Vec<Vec<f32>>> = None;
    let mut shared_v_full: Option<Vec<Vec<f32>>> = None;
    for l in 0..n_run {
        let lt = &binding.layers[l];
        let sliding = g.is_sliding_layer(l);
        let head_dim = g.head_dim_at(l) as usize;
        let theta = g.rope_freq_base_at(l);
        let q_dim = heads * head_dim;
        let kv_dim = kv_heads * head_dim;

        let (attn_norm_w, _) = load_f32(&store, &lt.attn_norm.name);
        if l == 0 {
            let stat = |w: &[f32], n: &str| {
                let mean = w.iter().sum::<f32>() / w.len() as f32;
                let rms = (w.iter().map(|v| v * v).sum::<f32>() / w.len() as f32).sqrt();
                eprintln!("  {n} weight mean={mean:.4} rms={rms:.4}");
            };
            stat(&attn_norm_w, "attn_norm[0]");
            let (qn, _) = load_f32(&store, &lt.attn_q_norm.name);
            let (kn, _) = load_f32(
                &store,
                &lt.attn_k_norm
                    .as_ref()
                    .expect("owning layer binds attn_k_norm")
                    .name,
            );
            stat(&qn, "q_norm[0]");
            stat(&kn, "k_norm[0]");
        }
        let (q_w, _) = load_f32(&store, &lt.attn_q.name);
        let (k_w, _) = load_f32(
            &store,
            &lt.attn_k.as_ref().expect("owning layer binds attn_k").name,
        );
        // V-less layers (12B full attention) reuse the K projection as V; this
        // reference test exercises the E-series rows, which always carry V.
        let (v_w, _) = load_f32(
            &store,
            &lt.attn_v
                .as_ref()
                .or(lt.attn_k.as_ref())
                .expect("owning layer binds attn_k")
                .name,
        );
        let (o_w, _) = load_f32(&store, &lt.attn_output.name);
        let (qn_w, _) = load_f32(&store, &lt.attn_q_norm.name);
        let (kn_w, _) = load_f32(
            &store,
            &lt.attn_k_norm
                .as_ref()
                .expect("owning layer binds attn_k_norm")
                .name,
        );
        let (post_attn_w, _) = load_f32(&store, &lt.post_attention_norm.name);
        let (ffn_norm_w, _) = load_f32(&store, &lt.ffn_norm.name);
        let (gate_w, _) = load_f32(&store, &lt.ffn_gate.name);
        let (up_w, _) = load_f32(&store, &lt.ffn_up.name);
        let (down_w, _) = load_f32(&store, &lt.ffn_down.name);
        let (post_ffw_w, _) = load_f32(&store, &lt.post_ffw_norm.name);
        let ffn_dim = config.feed_forward_length as usize;

        // attention: compute q/k/v per position (own KV — GGUF carries all layers)
        let mut qs = vec![vec![0f32; q_dim]; seq];
        let mut ks = vec![vec![0f32; kv_dim]; seq];
        let mut vs = vec![vec![0f32; kv_dim]; seq];
        let compute_kv = l < first_kv_shared;
        for t in 0..seq {
            let xn = rms_norm(&hs[t], Some(&attn_norm_w), eps);
            let mut q = matvec(&q_w, &xn, hidden, q_dim);
            for h in 0..heads {
                let s = &mut q[h * head_dim..(h + 1) * head_dim];
                s.copy_from_slice(&rms_norm(s, Some(&qn_w), eps));
            }
            apply_rope(&mut q, heads, head_dim, t, theta);
            qs[t] = q;
            if compute_kv {
                let mut k = matvec(&k_w, &xn, hidden, kv_dim);
                let mut v = matvec(&v_w, &xn, hidden, kv_dim);
                for h in 0..kv_heads {
                    let s = &mut k[h * head_dim..(h + 1) * head_dim];
                    s.copy_from_slice(&rms_norm(s, Some(&kn_w), eps));
                    let sv = &mut v[h * head_dim..(h + 1) * head_dim];
                    sv.copy_from_slice(&rms_norm(sv, None, eps));
                }
                apply_rope(&mut k, kv_heads, head_dim, t, theta);
                ks[t] = k;
                vs[t] = v;
            }
        }
        // Resolve the K/V for this layer: store (non-shared) or reuse (shared).
        if compute_kv {
            if sliding {
                shared_k_sliding = Some(ks.clone());
                shared_v_sliding = Some(vs.clone());
            } else {
                shared_k_full = Some(ks.clone());
                shared_v_full = Some(vs.clone());
            }
        } else if sliding {
            ks = shared_k_sliding.clone().expect("sliding shared kv");
            vs = shared_v_sliding.clone().expect("sliding shared kv");
        } else {
            ks = shared_k_full.clone().expect("full shared kv");
            vs = shared_v_full.clone().expect("full shared kv");
        }
        // attention output per position
        let scale = q_scale(head_dim);
        let group = heads / kv_heads;
        let mut attn_out = vec![vec![0f32; q_dim]; seq];
        for t in 0..seq {
            for h in 0..heads {
                let kvh = h / group;
                let qh = &qs[t][h * head_dim..(h + 1) * head_dim];
                // causal + sliding-window range
                let lo = if sliding {
                    (t + 1).saturating_sub(g.sliding_window as usize)
                } else {
                    0
                };
                let mut scores = Vec::with_capacity(t - lo + 1);
                for p in lo..=t {
                    let kp = &ks[p][kvh * head_dim..(kvh + 1) * head_dim];
                    let dot: f32 = qh.iter().zip(kp).map(|(a, b)| a * b).sum();
                    scores.push(dot * scale);
                }
                let m = scores.iter().cloned().fold(f32::MIN, f32::max);
                let mut den = 0f32;
                for s in scores.iter_mut() {
                    *s = (*s - m).exp();
                    den += *s;
                }
                let out = &mut attn_out[t][h * head_dim..(h + 1) * head_dim];
                for (idx, p) in (lo..=t).enumerate() {
                    let w = scores[idx] / den;
                    let vp = &vs[p][kvh * head_dim..(kvh + 1) * head_dim];
                    for d in 0..head_dim {
                        out[d] += w * vp[d];
                    }
                }
            }
        }
        // o_proj, post_attention_norm, residual
        for t in 0..seq {
            let o = matvec(&o_w, &attn_out[t], q_dim, hidden);
            let on = rms_norm(&o, Some(&post_attn_w), eps);
            add_into(&mut hs[t], &on);
        }
        // FFN: GeGLU + pre/post norm + residual
        for t in 0..seq {
            let xn = rms_norm(&hs[t], Some(&ffn_norm_w), eps);
            let gate = matvec(&gate_w, &xn, hidden, ffn_dim);
            let up = matvec(&up_w, &xn, hidden, ffn_dim);
            let act: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(gt, u)| gelu_tanh(*gt) * u)
                .collect();
            let down = matvec(&down_w, &act, ffn_dim, hidden);
            let dn = rms_norm(&down, Some(&post_ffw_w), eps);
            add_into(&mut hs[t], &dn);
        }
        // PLE injection (E-series)
        let enable_ple = std::env::var("NO_PLE").is_err();
        if enable_ple && !per_layer_input.is_empty() {
            let (gate_w, _) = load_f32(&store, &lt.ple_inp_gate.as_ref().unwrap().name); // [hidden, ple_dim]
            let (proj_w, _) = load_f32(&store, &lt.ple_proj.as_ref().unwrap().name); // [ple_dim, hidden]
            let (postn_w, _) = load_f32(&store, &lt.post_norm.as_ref().unwrap().name);
            let scale = lt
                .ple_output_scale
                .as_ref()
                .map(|d| load_f32(&store, &d.name).0[0])
                .unwrap_or(1.0);
            if l == 0 {
                eprintln!("  layer 0 ple_output_scale = {scale}");
            }
            for t in 0..seq {
                let mut gated = matvec(&gate_w, &hs[t], hidden, ple_dim);
                for (gv, pv) in gated.iter_mut().zip(&per_layer_input[t][l]) {
                    *gv = gelu_tanh(*gv) * pv;
                }
                let proj = matvec(&proj_w, &gated, ple_dim, hidden);
                let pn = rms_norm(&proj, Some(&postn_w), eps);
                if l == 0 && t == seq - 1 {
                    let rms =
                        |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
                    eprintln!(
                        "  PLE[0]: per_layer_input_rms={:.4} post_norm_w_rms={:.4} pn_rms={:.4} h_rms={:.4}",
                        rms(&per_layer_input[t][l]), rms(&postn_w), rms(&pn), rms(&hs[t])
                    );
                }
                add_into(&mut hs[t], &pn);
                // llama.cpp: l_out = (residual + PLE) * layer_output_scale. (The HF
                // layer_scalar=1.0 no-op folds this scale into the GGUF tensor.)
                for v in hs[t].iter_mut() {
                    *v *= scale;
                }
            }
        }
        if matches!(l, 0 | 4 | 5 | 11 | 41) {
            let tot: f32 = hs.iter().map(|r| r.iter().sum::<f32>()).sum();
            let refs = [
                (0, -23.41),
                (4, 528.32),
                (5, 696.44),
                (11, -743.48),
                (41, 54.93),
            ];
            let r = refs
                .iter()
                .find(|(i, _)| *i == l)
                .map(|(_, v)| *v)
                .unwrap_or(0.0);
            eprintln!("  l_out-{l} sum = {tot:.4}  (llama.cpp ref = {r})");
        }
    }

    // final norm + tied logits, computed for every position (fast f32 matmul).
    let (out_norm_w, _) = load_f32(&store, &binding.output_norm.name);
    let vocab = config.vocab_size.unwrap() as usize;
    let cap = g.final_logit_softcapping.unwrap_or(0.0);
    // Load the (tied) output embedding once as f32: [hidden, vocab] token-major,
    // so row v = out_embd[v*hidden .. (v+1)*hidden].
    let (out_embd, _) = load_f32(&store, &binding.output.name);
    let argmax_at = |pos: usize| -> usize {
        let last = rms_norm(&hs[pos], Some(&out_norm_w), eps);
        let mut best = (0usize, f32::MIN);
        for v in 0..vocab {
            let row = &out_embd[v * hidden..(v + 1) * hidden];
            let mut logit: f32 = row.iter().zip(&last).map(|(a, b)| a * b).sum();
            if cap > 0.0 {
                logit = cap * (logit / cap).tanh();
            }
            if logit > best.1 {
                best = (v, logit);
            }
        }
        best.0
    };

    // Teacher-forced greedy check: every position in the generated region must
    // predict the next token of llama.cpp's greedy continuation.
    eprintln!("teacher-forced next-token predictions (generated region):");
    let mut all_match = true;
    for pos in (PROMPT_LEN - 1)..(seq - 1) {
        let pred = argmax_at(pos) as u32;
        let want = FULL_SEQ[pos + 1];
        let ok = pred == want;
        all_match &= ok;
        eprintln!(
            "  pos {pos}: pred {pred} want {want} {}",
            if ok { "✅" } else { "❌" }
        );
    }
    eprintln!(
        "RESULT: {}",
        if all_match {
            "ALL MATCH ✅ — Gemma 4 greedy generation reproduces llama.cpp"
        } else {
            "MISMATCH ❌"
        }
    );
    if std::env::var("LAYERS").is_err()
        && std::env::var("NO_PLE").is_err()
        && std::env::var("NO_VNORM").is_err()
    {
        assert_eq!(
            argmax_at(PROMPT_LEN - 1) as u32,
            ORACLE_FIRST_TOKEN,
            "first generated token must be the oracle 9079 ' Paris'"
        );
        assert!(all_match, "gemma4 greedy continuation must match llama.cpp");
    }
}
