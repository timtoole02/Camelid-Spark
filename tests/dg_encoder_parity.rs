//! DiffusionGemma lane Phase 2 gate: encoder (PREFILL) per-layer checkpoint
//! parity against the pinned llama.cpp reference for the exact tracked GGUF.
//!
//! Env-gated: skips unless `CAMELID_DG_GGUF`, `CAMELID_DG_ENC_REF` (a
//! directory produced by `scripts/dg-encoder-dump.cpp` at the pin:
//! manifest.json + raw tensors), and `CAMELID_DG_ENC_IDS` (the prompt id
//! file the dumper consumed) are set. Run via
//! `scripts/dg-encoder-parity.sh`.
//!
//! Comparison contract:
//! - every f32 checkpoint must satisfy |a-b| <= atol + rtol*|b| elementwise
//!   (atol/rtol from CAMELID_DG_ENC_ATOL / CAMELID_DG_ENC_RTOL; the chosen
//!   values are recorded in the gate artifact with their justification);
//! - MoE selected expert indices must match EXACTLY as per-position SETS
//!   (routing is discrete; "close" is a fail). Order agreement is also
//!   recorded but the gate is set equality, since the reference's top-k
//!   tie order is an implementation detail of its argsort.
//! - the first layer with any out-of-tolerance checkpoint is reported as
//!   `first_divergent_layer`.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use camelid::diffusion_gemma::DgEncoderRuntime;

struct RefTensor {
    name: String,
    type_name: String,
    ne: [i64; 4],
    bytes: Vec<u8>,
}

fn json_str(line: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\":\"");
    let start = line.find(&pat)? + pat.len();
    let end = line[start..].find('"')? + start;
    Some(line[start..end].to_string())
}

fn parse_manifest(dir: &Path) -> Vec<RefTensor> {
    let text = std::fs::read_to_string(dir.join("manifest.json")).expect("read manifest");
    text.lines()
        .filter(|l| l.trim_start().starts_with('{'))
        .map(|line| {
            let name = json_str(line, "name").expect("name");
            let type_name = json_str(line, "type").expect("type");
            let ne_pat = "\"ne\":[";
            let s = line.find(ne_pat).expect("ne") + ne_pat.len();
            let e = line[s..].find(']').expect("ne close") + s;
            let ne_vals: Vec<i64> = line[s..e]
                .split(',')
                .map(|t| t.trim().parse().expect("ne value"))
                .collect();
            let file = json_str(line, "file").expect("file");
            let bytes =
                std::fs::read(dir.join(&file)).unwrap_or_else(|e| panic!("read {file}: {e}"));
            RefTensor {
                name,
                type_name,
                ne: [ne_vals[0], ne_vals[1], ne_vals[2], ne_vals[3]],
                bytes,
            }
        })
        .collect()
}

fn as_f32(t: &RefTensor) -> Vec<f32> {
    assert_eq!(t.type_name, "f32", "{}: expected f32", t.name);
    t.bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn as_i32(t: &RefTensor) -> Vec<i32> {
    assert_eq!(t.type_name, "i32", "{}: expected i32", t.name);
    t.bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[derive(Default, Clone)]
struct CmpStats {
    values: usize,
    max_abs: f64,
    mean_abs: f64,
    max_at: usize,
    within: bool,
}

fn compare_f32(camelid: &[f32], reference: &[f32], atol: f64, rtol: f64) -> CmpStats {
    assert_eq!(camelid.len(), reference.len());
    let mut s = CmpStats {
        values: camelid.len(),
        within: true,
        ..Default::default()
    };
    let mut sum = 0f64;
    for (i, (&a, &b)) in camelid.iter().zip(reference.iter()).enumerate() {
        let d = (a as f64 - b as f64).abs();
        sum += d;
        if d > s.max_abs {
            s.max_abs = d;
            s.max_at = i;
        }
        if d > atol + rtol * (b as f64).abs() {
            s.within = false;
        }
    }
    s.mean_abs = sum / camelid.len().max(1) as f64;
    s
}

#[test]
fn dg_encoder_prefill_matches_pinned_llamacpp() {
    let (Ok(gguf_path), Ok(ref_dir), Ok(ids_path)) = (
        std::env::var("CAMELID_DG_GGUF"),
        std::env::var("CAMELID_DG_ENC_REF"),
        std::env::var("CAMELID_DG_ENC_IDS"),
    ) else {
        eprintln!(
            "skipping: CAMELID_DG_GGUF / CAMELID_DG_ENC_REF / CAMELID_DG_ENC_IDS not set \
             (run scripts/dg-encoder-parity.sh)"
        );
        return;
    };
    let atol: f64 = std::env::var("CAMELID_DG_ENC_ATOL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2e-2);
    let rtol: f64 = std::env::var("CAMELID_DG_ENC_RTOL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2e-3);

    let ids_bytes = std::fs::read(&ids_path).expect("read prompt ids");
    let prompt: Vec<u32> = ids_bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
        .collect();
    let n = prompt.len();
    assert!(n > 0, "empty prompt id file");

    let refs = parse_manifest(Path::new(&ref_dir));
    let by_name: BTreeMap<&str, &RefTensor> = refs.iter().map(|t| (t.name.as_str(), t)).collect();

    eprintln!("loading DiffusionGemma encoder runtime (lazy mmap)...");
    let rt = DgEncoderRuntime::load(Path::new(&gguf_path)).expect("load runtime");

    // Diagnostic mode: pin the MoE routing to the reference's expert choices
    // to isolate knife-edge router ties from every continuous checkpoint.
    let pin_routing = std::env::var("CAMELID_DG_ENC_PIN_ROUTING").is_ok_and(|v| v == "1");
    let routing: Option<Vec<Vec<i32>>> = pin_routing.then(|| {
        (0..rt.n_layer())
            .map(|l| {
                let t = by_name
                    .get(format!("ffn_moe_topk-{l}").as_str())
                    .unwrap_or_else(|| panic!("pinned routing needs ffn_moe_topk-{l}"));
                let raw = as_i32(t);
                let k = t.ne[0] as usize;
                let stride = if raw.len() == n * k {
                    k
                } else {
                    (raw.len() - k) / (n - 1)
                };
                (0..n)
                    .flat_map(|pos| raw[pos * stride..pos * stride + k].to_vec())
                    .collect()
            })
            .collect()
    });

    eprintln!(
        "running encoder prefill over {n} positions{}...",
        if pin_routing {
            " (routing pinned to reference)"
        } else {
            ""
        }
    );
    let t0 = std::time::Instant::now();
    let trace = match routing.as_ref() {
        Some(r) => rt
            .encoder_prefill_with_pinned_routing(&prompt, r)
            .expect("encoder prefill (pinned routing)"),
        None => rt.encoder_prefill(&prompt).expect("encoder prefill"),
    };
    eprintln!("encoder prefill done in {:.1}s", t0.elapsed().as_secs_f32());

    // optional: dump camelid's checkpoints for offline ulp forensics
    if let Ok(dump_dir) = std::env::var("CAMELID_DG_ENC_DUMP_DIR") {
        let dir = PathBuf::from(&dump_dir);
        std::fs::create_dir_all(&dir).expect("create camelid dump dir");
        let write_f32 = |name: &str, data: &[f32]| {
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(dir.join(format!("{name}.bin")), bytes).expect("write dump");
        };
        write_f32("inp_region", &trace.inp_scaled);
        write_f32("result_norm", &trace.result_norm_last);
        for (l, lt) in trace.layers.iter().enumerate() {
            write_f32(&format!("Kcur_pos-{l}"), &lt.k);
            write_f32(&format!("Vcur_normed-{l}"), &lt.v);
            write_f32(&format!("attn_out-{l}"), &lt.attn_out);
            write_f32(&format!("ffn_moe_logits-{l}"), &lt.moe_logits);
            write_f32(&format!("l_out-{l}"), &lt.out_scaled);
            write_f32(&format!("ffn_mlp-{l}"), &lt.ffn_mlp);
            write_f32(&format!("ffn_moe-{l}"), &lt.ffn_moe);
            write_f32(&format!("ffn_moe_weights-{l}"), &lt.moe_weights);
            write_f32(&format!("ffn_moe_gate_up-{l}"), &lt.moe_gate_up);
            write_f32(&format!("ffn_moe_geglu-{l}"), &lt.moe_geglu);
            write_f32(&format!("ffn_moe_down-{l}"), &lt.moe_down);
            write_f32(&format!("ffn_moe_down_scaled-{l}"), &lt.moe_down_scaled);
            write_f32(&format!("ffn_moe_weights_norm-{l}"), &lt.moe_weights_norm);
            write_f32(&format!("ffn_moe_out-{l}"), &lt.moe_pre_norm);
            let tk: Vec<u8> = lt.moe_topk.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(dir.join(format!("ffn_moe_topk-{l}.bin")), tk).expect("write topk");
        }
        eprintln!("camelid checkpoints dumped to {dump_dir}");
    }

    let mut rows: Vec<String> = Vec::new();
    let mut first_divergent_layer: Option<usize> = None;
    let mut failures: Vec<String> = Vec::new();
    let mut expert_sets_equal = true;
    let mut expert_order_equal = true;

    let check = |name: &str,
                 layer: Option<usize>,
                 camelid: &[f32],
                 rows: &mut Vec<String>,
                 failures: &mut Vec<String>,
                 first_div: &mut Option<usize>| {
        let Some(rt) = by_name.get(name) else {
            failures.push(format!("{name}: missing from reference dump"));
            return;
        };
        let reference = as_f32(rt);
        if reference.len() != camelid.len() {
            failures.push(format!(
                "{name}: length {} (camelid) vs {} (reference, ne {:?})",
                camelid.len(),
                reference.len(),
                rt.ne
            ));
            return;
        }
        let s = compare_f32(camelid, &reference, atol, rtol);
        rows.push(format!(
            "    {{\"name\": \"{name}\", \"values\": {}, \"max_abs\": {:.3e}, \"mean_abs\": {:.3e}, \"within_tolerance\": {}}}",
            s.values, s.max_abs, s.mean_abs, s.within
        ));
        if !s.within {
            failures.push(format!(
                "{name}: max_abs {:.3e} at flat index {} (camelid {} vs reference {})",
                s.max_abs, s.max_at, camelid[s.max_at], reference[s.max_at]
            ));
            if let Some(l) = layer {
                if first_div.is_none_or(|cur| l < cur) {
                    *first_div = Some(l);
                }
            }
        }
    };

    // input embeddings: cb() renames, so the scaled embedding survives only
    // under its last label "inp_region" (PREFILL leaves prompt rows unchanged
    // after scaling)
    check(
        "inp_region",
        None,
        &trace.inp_scaled,
        &mut rows,
        &mut failures,
        &mut first_divergent_layer,
    );

    for (l, lt) in trace.layers.iter().enumerate() {
        check(
            &format!("Kcur_pos-{l}"),
            Some(l),
            &lt.k,
            &mut rows,
            &mut failures,
            &mut first_divergent_layer,
        );
        check(
            &format!("Vcur_normed-{l}"),
            Some(l),
            &lt.v,
            &mut rows,
            &mut failures,
            &mut first_divergent_layer,
        );
        check(
            &format!("attn_out-{l}"),
            Some(l),
            &lt.attn_out,
            &mut rows,
            &mut failures,
            &mut first_divergent_layer,
        );
        check(
            &format!("ffn_moe_logits-{l}"),
            Some(l),
            &lt.moe_logits,
            &mut rows,
            &mut failures,
            &mut first_divergent_layer,
        );
        check(
            &format!("ffn_mlp-{l}"),
            Some(l),
            &lt.ffn_mlp,
            &mut rows,
            &mut failures,
            &mut first_divergent_layer,
        );
        check(
            &format!("ffn_moe-{l}"),
            Some(l),
            &lt.ffn_moe,
            &mut rows,
            &mut failures,
            &mut first_divergent_layer,
        );
        check(
            &format!("ffn_moe_weights-{l}"),
            Some(l),
            &lt.moe_weights,
            &mut rows,
            &mut failures,
            &mut first_divergent_layer,
        );
        // the scaled layer output's surviving label is "l_out-N" (cvec is a
        // pass-through, values identical to out_scaled)
        check(
            &format!("l_out-{l}"),
            Some(l),
            &lt.out_scaled,
            &mut rows,
            &mut failures,
            &mut first_divergent_layer,
        );

        // MoE gate sub-check: selected expert indices, exact. The reference
        // tensor is a VIEW of the full per-position argsort (ne[0]=k, but the
        // row stride is the parent's n_expert row), and the dump captures the
        // raw span — so each position's top-k is the first k of a strided row.
        let name = format!("ffn_moe_topk-{l}");
        if let Some(rt) = by_name.get(name.as_str()) {
            let reference = as_i32(rt);
            let k = rt.ne[0] as usize;
            let stride = if reference.len() == n * k {
                k
            } else {
                // spanned bytes of a strided view: (n-1)*stride + k
                (reference.len() - k) / (n - 1)
            };
            assert_eq!(
                (n - 1) * stride + k,
                reference.len(),
                "{name}: unexpected reference layout"
            );
            for pos in 0..n {
                let mut a: Vec<i32> = lt.moe_topk[pos * k..(pos + 1) * k].to_vec();
                let mut b: Vec<i32> = reference[pos * stride..pos * stride + k].to_vec();
                if a != b {
                    expert_order_equal = false;
                }
                a.sort_unstable();
                b.sort_unstable();
                if a != b {
                    expert_sets_equal = false;
                    // quantify the flip: the reference's softmax-prob margin
                    // between its rank-k and rank-k+1 experts at this position
                    let margin = by_name
                        .get(format!("ffn_moe_logits-{l}").as_str())
                        .map(|logits_ref| {
                            let logits = as_f32(logits_ref);
                            let n_e = logits.len() / n;
                            let row = &logits[pos * n_e..(pos + 1) * n_e];
                            let maxl = row.iter().cloned().fold(f32::MIN, f32::max);
                            let mut probs: Vec<f32> =
                                row.iter().map(|&v| (v - maxl).exp()).collect();
                            let s: f32 = probs.iter().sum();
                            for p in probs.iter_mut() {
                                *p /= s;
                            }
                            probs.sort_unstable_by(|x, y| y.partial_cmp(x).unwrap());
                            probs[k - 1] - probs[k]
                        })
                        .unwrap_or(f32::NAN);
                    let detail = format!(
                        "{name}: position {pos} expert set {:?} (camelid) vs {:?} (reference); \
                         reference rank-{k}/rank-{} prob margin {margin:e}",
                        &lt.moe_topk[pos * k..(pos + 1) * k],
                        &reference[pos * stride..pos * stride + k],
                        k + 1,
                    );
                    if pin_routing {
                        // pinned-routing diagnostic: camelid's own selection
                        // still flips the documented knife-edge ties; the run
                        // exists to isolate the continuous checkpoints.
                        eprintln!("documented tie flip: {detail}");
                    } else {
                        failures.push(detail);
                        if first_divergent_layer.is_none_or(|cur| l < cur) {
                            first_divergent_layer = Some(l);
                        }
                    }
                }
            }
        } else {
            failures.push(format!("{name}: missing from reference dump"));
        }
    }

    // final norm: reference dumps have carried one-row and all-rows shapes —
    // compare whichever extent the reference provides
    {
        let cam_full = &trace.result_norm_all;
        let ref_len = by_name
            .get("result_norm")
            .map(|t| t.bytes.len() / 4)
            .unwrap_or(0);
        let cam_slice: &[f32] = if ref_len == cam_full.len() {
            cam_full
        } else {
            &trace.result_norm_last
        };
        check(
            "result_norm",
            None,
            cam_slice,
            &mut rows,
            &mut failures,
            &mut first_divergent_layer,
        );
    }

    let report = format!(
        "{{\n  \"comparison\": \"camelid DgEncoderRuntime::encoder_prefill vs pinned llama.cpp PREFILL checkpoints (scripts/dg-encoder-dump.cpp, CPU backend, flash attention off)\",\n  \"mode\": \"{}\",\n  \"llamacpp_pinned_commit\": \"{}\",\n  \"gguf\": \"{}\",\n  \"prompt_ids\": {:?},\n  \"tolerance\": {{\"atol\": {atol:e}, \"rtol\": {rtol:e}, \"note\": \"same quantized-activation math in a different accumulation order (sequential per-position matvec vs batched reference); expert indices are exact-match, not tolerated\"}},\n  \"expert_index_sets_equal\": {expert_sets_equal},\n  \"expert_index_order_equal\": {expert_order_equal},\n  \"first_divergent_layer\": {},\n  \"checkpoints\": [\n{}\n  ],\n  \"pass\": {}\n}}\n",
        if pin_routing {
            "pinned_routing_diagnostic"
        } else {
            "free_running"
        },
        std::env::var("CAMELID_DG_PIN_SHA").unwrap_or_else(|_| "UNRECORDED".to_string()),
        Path::new(&gguf_path)
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default(),
        prompt,
        first_divergent_layer.map_or("null".to_string(), |l| l.to_string()),
        rows.join(",\n"),
        failures.is_empty(),
    );

    if let Ok(out_path) = std::env::var("CAMELID_DG_ENC_OUT") {
        let mut f = std::fs::File::create(PathBuf::from(&out_path)).expect("create gate report");
        f.write_all(report.as_bytes()).expect("write gate report");
        eprintln!("gate report written to {out_path}");
    }
    eprintln!("{report}");

    assert!(
        failures.is_empty(),
        "encoder parity failures ({}):\n{}",
        failures.len(),
        failures.join("\n")
    );
}
