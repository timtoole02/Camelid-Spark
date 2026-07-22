//! Phase 5 divergence ladder (NOT a gate): run camelid's story block-0
//! step-0 traced zero-SC forward (N=297) and compare every per-layer
//! checkpoint against the oracle unified dump to find the FIRST divergent
//! layer/checkpoint. Env: CAMELID_DG_GGUF, CAMELID_DG_DIAG_REF (oracle
//! unified dump dir), CAMELID_DG_DIAG_IDS (story prompt ids),
//! CAMELID_DG_DIAG_RNG (seed-0 canvas-ids dir).

use std::path::Path;

use camelid::diffusion_gemma::DgEncoderRuntime;

fn read_i32(p: &Path) -> Vec<i32> {
    std::fs::read(p)
        .unwrap()
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn cmp(name: &str, ours: &[f32], path: &Path) -> bool {
    if !path.exists() {
        eprintln!("  {name}: (no ref file, skipped)");
        return true;
    }
    let theirs = read_f32(path);
    let len = ours.len().min(theirs.len());
    let mut bad = 0usize;
    let mut maxabs = 0f32;
    let mut first = usize::MAX;
    for (idx, (a, b)) in ours[..len].iter().zip(&theirs[..len]).enumerate() {
        if a.to_bits() != b.to_bits() {
            bad += 1;
            if first == usize::MAX {
                first = idx;
            }
            maxabs = maxabs.max((a - b).abs());
        }
    }
    if bad > 0 {
        eprintln!(
            "  {name}: {bad}/{len} not bit-exact, maxabs={maxabs:.3e}, first idx {first} \
             (ours_len={} ref_len={})",
            ours.len(),
            theirs.len()
        );
        false
    } else {
        true
    }
}

#[test]
fn dg_mc_block0_step0_ladder() {
    let (Ok(g), Ok(r), Ok(i), Ok(rng)) = (
        std::env::var("CAMELID_DG_GGUF"),
        std::env::var("CAMELID_DG_DIAG_REF"),
        std::env::var("CAMELID_DG_DIAG_IDS"),
        std::env::var("CAMELID_DG_DIAG_RNG"),
    ) else {
        eprintln!("skipping: diag env not set");
        return;
    };
    let refd = Path::new(&r);
    let prompt: Vec<u32> = read_i32(Path::new(&i))
        .into_iter()
        .map(|v| v as u32)
        .collect();
    let canvas: Vec<u32> = read_i32(&Path::new(&rng).join("canvas-ids.i32"))
        .into_iter()
        .map(|v| v as u32)
        .collect();
    let rt = DgEncoderRuntime::load(Path::new(&g)).expect("load");
    eprintln!(
        "story block-0 traced forward (P={} C={} N={})...",
        prompt.len(),
        canvas.len(),
        prompt.len() + canvas.len()
    );
    // Optional SC-active mode: CAMELID_DG_DIAG_SC=<sc-logits.f32>,
    // CAMELID_DG_DIAG_SC_TINV=<temp_inv>. Without them, zero-SC (step-0 surface).
    let out = match (
        std::env::var("CAMELID_DG_DIAG_SC"),
        std::env::var("CAMELID_DG_DIAG_SC_TINV"),
    ) {
        (Ok(scp), Ok(tinv)) => {
            let sc_logits = read_f32(Path::new(&scp));
            let temp_inv: f32 = tinv.parse().expect("temp_inv");
            eprintln!(
                "SC-ACTIVE diag: sc_len={} temp_inv={temp_inv}",
                sc_logits.len()
            );
            let sc = camelid::diffusion_gemma::DgScInput {
                logits: &sc_logits,
                temp_inv,
                use_sc: 1.0,
            };
            rt.unified_forward_sc(&prompt, &canvas, Some(&sc), true)
                .expect("forward")
        }
        _ => rt.unified_forward(&prompt, &canvas, true).expect("forward"),
    };
    let logits = out.logits;
    let trace = out.trace.expect("trace");

    let mut first_divergent: Option<String> = None;
    let note = |name: String, ok: bool, first_divergent: &mut Option<String>| {
        if !ok && first_divergent.is_none() {
            *first_divergent = Some(name);
        }
    };

    let ok = cmp(
        "inp_region",
        &trace.inp_scaled,
        &refd.join("inp_region.bin"),
    );
    note("inp_region".into(), ok, &mut first_divergent);

    // per-position analysis of attn_out-0 (n_embd-fast, position-slow): WHICH
    // positions diverge tells prompt-vs-canvas + boundary effects
    {
        let h = rt.n_embd();
        let theirs = read_f32(&refd.join("attn_out-0.bin"));
        let ours = &trace.layers[0].attn_out;
        let npos = ours.len() / h;
        let mut div_pos = Vec::new();
        for p in 0..npos {
            let bad = (0..h)
                .filter(|&d| ours[p * h + d].to_bits() != theirs[p * h + d].to_bits())
                .count();
            if bad > 0 {
                div_pos.push((p, bad));
            }
        }
        eprintln!(
            "attn_out-0 diverging positions ({} of {npos}, P={}): {:?}",
            div_pos.len(),
            prompt.len(),
            div_pos
        );
    }

    // attention-internal split: softmax weights, then KQV value-mix
    if !trace.layers[0].kq_soft_max.is_empty() {
        let ok = cmp(
            "kq_soft_max-0",
            &trace.layers[0].kq_soft_max,
            &refd.join("kq_soft_max-0.bin"),
        );
        note("kq_soft_max-0".into(), ok, &mut first_divergent);
    }
    if !trace.layers[0].kqv.is_empty() {
        let ok = cmp("kqv-0", &trace.layers[0].kqv, &refd.join("kqv-0.bin"));
        note("kqv-0".into(), ok, &mut first_divergent);

        // ISOLATION: for the first diverging kqv element, dump the exact
        // V-column and softmax-row (both bit-exact inputs) so the dot can be
        // recomputed by a fresh ggml_vec_dot_f32 offline. Layer-0 layout:
        // head_dim=256, n_head=16, n_head_kv=8, group=2.
        let (head_dim, n_head, group) = (256usize, 16usize, 2usize);
        let np = prompt.len() + 256;
        let kqv_ref = read_f32(&refd.join("kqv-0.bin"));
        let ours = &trace.layers[0].kqv;
        let kv_dim = (n_head / group) * head_dim; // 8*256
        let vflat = &trace.layers[0].v; // [n_pos][kv_dim], Vcur_normed, bit-exact
        let sm = &trace.layers[0].kq_soft_max; // [k + q*np + h*np*np]
        if let Some(idx) = ours
            .iter()
            .zip(&kqv_ref)
            .position(|(a, b)| a.to_bits() != b.to_bits())
        {
            let h = idx / (head_dim * np);
            let q = (idx % (head_dim * np)) / head_dim;
            let d = idx % head_dim;
            let kvh = h / group;
            // V column: v_col[k] = V[k][kvh*head_dim + d]
            let v_col: Vec<f32> = (0..np)
                .map(|k| vflat[k * kv_dim + kvh * head_dim + d])
                .collect();
            // softmax row for (q, h)
            let s_row: Vec<f32> = (0..np).map(|k| sm[k + q * np + h * np * np]).collect();
            // camelid vec_dot of these exact inputs (should == ours[idx])
            let cam = v_col
                .iter()
                .zip(&s_row)
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum::<f64>() as f32;
            eprintln!(
                "KQV ISO idx {idx} (d={d},q={q},h={h},kvh={kvh}): ours={} ref={} naive_f64={cam} (ours_bits={:#010x} ref_bits={:#010x})",
                ours[idx], kqv_ref[idx], ours[idx].to_bits(), kqv_ref[idx].to_bits()
            );
            // dump v_col + softmax for offline ggml_vec_dot_f32 + the two targets
            let mut blob: Vec<u8> = Vec::new();
            blob.extend_from_slice(&(np as u32).to_le_bytes());
            for &v in &v_col {
                blob.extend_from_slice(&v.to_le_bytes());
            }
            for &v in &s_row {
                blob.extend_from_slice(&v.to_le_bytes());
            }
            blob.extend_from_slice(&ours[idx].to_le_bytes());
            blob.extend_from_slice(&kqv_ref[idx].to_le_bytes());
            std::fs::write("/Volumes/Untitled/dg-phase5-work/kqv-elem.bin", &blob).unwrap();
            eprintln!("wrote kqv-elem.bin (np, v_col[np], softmax[np], ours, ref)");
        }
    }

    for (l, lt) in trace.layers.iter().enumerate() {
        for (name, data) in [
            (format!("Kcur_pos-{l}"), &lt.k),
            (format!("Vcur_normed-{l}"), &lt.v),
            (format!("attn_out-{l}"), &lt.attn_out),
            (format!("ffn_moe_logits-{l}"), &lt.moe_logits),
            // MoE expert compute/combine chain (in graph order) — pinpoints
            // which sub-op first diverges (the l_out-17 seed lives here).
            (format!("ffn_moe_gate_up-{l}"), &lt.moe_gate_up),
            (format!("ffn_moe_geglu-{l}"), &lt.moe_geglu),
            (format!("ffn_moe_down-{l}"), &lt.moe_down),
            (format!("ffn_moe_down_scaled-{l}"), &lt.moe_down_scaled),
            (format!("ffn_moe_weights_norm-{l}"), &lt.moe_weights_norm),
            (format!("ffn_moe_out-{l}"), &lt.moe_pre_norm),
            // post-expert-sum tail: post_norm_2 MoE out, dense MLP branch,
            // pre-scalar FFN block output (residual + post_ffw_norm), l_out
            (format!("ffn_moe-{l}"), &lt.ffn_moe),
            (format!("ffn_mlp-{l}"), &lt.ffn_mlp),
            (format!("ffn_block_out-{l}"), &lt.ffn_block_out),
            (format!("l_out-{l}"), &lt.out_scaled),
        ] {
            let ok = cmp(&name, data, &refd.join(format!("{name}.bin")));
            note(name, ok, &mut first_divergent);
        }
    }
    let ok = cmp(
        "result_norm",
        &trace.result_norm_all,
        &refd.join("result_norm.bin"),
    );
    note("result_norm".into(), ok, &mut first_divergent);

    // Full canvas-row logits — the exact self-conditioning input for the next
    // step, and the most sensitive check (a sub-argmax/sub-entropy diff here
    // is what SC amplifies into the step-3 flip). Oracle result_output is
    // [n_vocab, N] position-slow; canvas rows are positions P..P+C. camelid's
    // out.logits is the C canvas rows, same vocab-fast layout.
    {
        let nvocab = logits.len() / 256;
        let p = prompt.len();
        let ro = read_f32(&refd.join("result_output.bin"));
        if ro.len() >= (p + 256) * nvocab {
            let oracle_canvas = &ro[p * nvocab..(p + 256) * nvocab];
            let mut bad = 0usize;
            let mut first = usize::MAX;
            let mut maxabs = 0f32;
            for (idx, (a, b)) in logits.iter().zip(oracle_canvas).enumerate() {
                if a.to_bits() != b.to_bits() {
                    bad += 1;
                    if first == usize::MAX {
                        first = idx;
                    }
                    maxabs = maxabs.max((a - b).abs());
                }
            }
            if bad > 0 {
                let (fp, fv) = (first / nvocab, first % nvocab);
                eprintln!(
                    "  result_output(canvas): {bad}/{} not bit-exact, maxabs={maxabs:.3e}, \
                     first idx {first} (canvas-pos {fp}, vocab {fv})",
                    logits.len()
                );
                note("result_output(canvas)".into(), false, &mut first_divergent);
            } else {
                eprintln!(
                    "  result_output(canvas): BIT-EXACT ({} logits)",
                    logits.len()
                );
            }
        } else {
            eprintln!(
                "  result_output: ref too small ({} floats), skipped",
                ro.len()
            );
        }
    }

    eprintln!(
        "FIRST DIVERGENT: {}",
        first_divergent.unwrap_or_else(|| "none (all bit-exact)".into())
    );
}
