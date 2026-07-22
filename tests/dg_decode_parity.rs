//! DiffusionGemma lane Phase 3 gate: single denoise step parity against the
//! pinned llama.cpp reference — host RNG (canvas init + step-0 draws), the
//! unified zero-SC `[prompt | canvas]` forward's canvas logits, and one
//! Entropy-Bound sampler step's outputs.
//!
//! Env-gated: skips unless `CAMELID_DG_GGUF`, `CAMELID_DG_DEC_REF` (a
//! directory produced by `scripts/dg-encoder-dump.cpp` in UNIFIED mode:
//! manifest.json + raw tensors including `result_output`),
//! `CAMELID_DG_DEC_IDS` (prompt ids), `CAMELID_DG_DEC_RNG` (a directory from
//! `scripts/dg-rng-dump.cpp`: canvas-ids.i32 + u-step0.f32 +
//! renoise-step0.i32) and `CAMELID_DG_DEC_EB` (a directory from
//! `scripts/dg-eb-step.cpp`: eb-*.i32/f32/u8) are set. Run via
//! `scripts/dg-decode-parity.sh`.
//!
//! Comparison contract (same shape as the Phase 2 gate):
//! - RNG streams and every discrete output (canvas ids, renoise ids, argmax,
//!   denoiser, accepted set, next canvas) must match EXACTLY;
//! - every f32 surface (canvas logits, entropies, trace checkpoints when
//!   present) must satisfy |a-b| <= atol + rtol*|b| elementwise, with
//!   atol/rtol from CAMELID_DG_DEC_ATOL / CAMELID_DG_DEC_RTOL (default 0 —
//!   the bit-exactness bar Phase 2 set).

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use camelid::diffusion_gemma::refrng;
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

fn read_i32_file(path: &Path) -> Vec<i32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_f32_file(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[derive(Default, Clone)]
struct CmpStats {
    values: usize,
    max_abs: f64,
    mean_abs: f64,
    max_at: usize,
    non_bit_exact: usize,
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
        if a.to_bits() != b.to_bits() {
            s.non_bit_exact += 1;
        }
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
fn dg_single_denoise_step_matches_pinned_llamacpp() {
    let (Ok(gguf_path), Ok(ref_dir), Ok(ids_path), Ok(rng_dir), Ok(eb_dir)) = (
        std::env::var("CAMELID_DG_GGUF"),
        std::env::var("CAMELID_DG_DEC_REF"),
        std::env::var("CAMELID_DG_DEC_IDS"),
        std::env::var("CAMELID_DG_DEC_RNG"),
        std::env::var("CAMELID_DG_DEC_EB"),
    ) else {
        eprintln!(
            "skipping: CAMELID_DG_GGUF / CAMELID_DG_DEC_REF / CAMELID_DG_DEC_IDS / \
             CAMELID_DG_DEC_RNG / CAMELID_DG_DEC_EB not set (run scripts/dg-decode-parity.sh)"
        );
        return;
    };
    let atol: f64 = std::env::var("CAMELID_DG_DEC_ATOL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let rtol: f64 = std::env::var("CAMELID_DG_DEC_RTOL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let seed: u32 = std::env::var("CAMELID_DG_DEC_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    // EB sampler defaults at the pin (diffusion_eb_params)
    let s_steps: i32 = 48;
    let (t_min, t_max, bound) = (0.4f32, 0.8f32, 0.1f32);

    let prompt: Vec<u32> = read_i32_file(Path::new(&ids_path))
        .into_iter()
        .map(|v| v as u32)
        .collect();
    assert!(!prompt.is_empty(), "empty prompt id file");
    let p = prompt.len();

    let rng_dir = PathBuf::from(rng_dir);
    let ref_canvas = read_i32_file(&rng_dir.join("canvas-ids.i32"));
    let ref_u = read_f32_file(&rng_dir.join("u-step0.f32"));
    let ref_renoise = read_i32_file(&rng_dir.join("renoise-step0.i32"));
    let c = ref_canvas.len();
    assert_eq!(ref_u.len(), c);
    assert_eq!(ref_renoise.len(), c);

    eprintln!("loading DiffusionGemma runtime (lazy mmap)...");
    let rt = DgEncoderRuntime::load(Path::new(&gguf_path)).expect("load runtime");
    let n_vocab = rt.n_vocab();

    // ---- gate 1: host RNG parity (libc++ distribution ports) ----
    let draws = refrng::eb_draws(seed, n_vocab as i32, c, 1);
    let rng_canvas_ok = draws.canvas_init == ref_canvas;
    let rng_u_ok = draws.u[0]
        .iter()
        .zip(&ref_u)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    let rng_renoise_ok = draws.renoise[0] == ref_renoise;
    eprintln!(
        "RNG parity: canvas_init={rng_canvas_ok} u={rng_u_ok} renoise={rng_renoise_ok} (C={c})"
    );

    // ---- gate 2: unified zero-SC forward, canvas logits ----
    let canvas: Vec<u32> = ref_canvas.iter().map(|&v| v as u32).collect();
    let want_trace = std::env::var("CAMELID_DG_DEC_TRACE").map_or(true, |v| v != "0");
    eprintln!("running unified forward over [{p} | {c}] positions...");
    let t0 = std::time::Instant::now();
    let out = rt
        .unified_forward(&prompt, &canvas, want_trace)
        .expect("unified forward");
    eprintln!("unified forward done in {:.1}s", t0.elapsed().as_secs_f32());
    assert_eq!(out.n_vocab, n_vocab);
    assert_eq!(out.logits.len(), c * n_vocab);

    if let Ok(dump_dir) = std::env::var("CAMELID_DG_DEC_DUMP_DIR") {
        let dir = PathBuf::from(&dump_dir);
        std::fs::create_dir_all(&dir).expect("create camelid dump dir");
        let write_f32 = |name: &str, data: &[f32]| {
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(dir.join(format!("{name}.bin")), bytes).expect("write dump");
        };
        write_f32("logits", &out.logits);
        if let Some(trace) = &out.trace {
            write_f32("inp_region", &trace.inp_scaled);
            write_f32("result_norm", &trace.result_norm_all);
            for (l, lt) in trace.layers.iter().enumerate() {
                write_f32(&format!("Kcur_pos-{l}"), &lt.k);
                write_f32(&format!("Vcur_normed-{l}"), &lt.v);
                write_f32(&format!("attn_out-{l}"), &lt.attn_out);
                write_f32(&format!("l_out-{l}"), &lt.out_scaled);
                write_f32(&format!("ffn_moe_logits-{l}"), &lt.moe_logits);
            }
        }
    }

    let refs = parse_manifest(Path::new(&ref_dir));
    let by_name: BTreeMap<&str, &RefTensor> = refs.iter().map(|t| (t.name.as_str(), t)).collect();

    let mut sections: Vec<String> = Vec::new();
    let mut all_within = true;
    let mut first_divergent: Option<String> = None;
    let note = |name: &str, st: &CmpStats, first_divergent: &mut Option<String>| {
        if !st.within && first_divergent.is_none() {
            *first_divergent = Some(name.to_string());
        }
        format!(
            "{{\"name\":\"{name}\",\"values\":{},\"max_abs\":{:.6e},\"mean_abs\":{:.6e},\
             \"max_at\":{},\"non_bit_exact\":{},\"within\":{}}}",
            st.values, st.max_abs, st.mean_abs, st.max_at, st.non_bit_exact, st.within
        )
    };

    // canvas logits vs the reference's result_output (all-rows; slice the
    // canvas rows = the last C)
    let ro = by_name
        .get("result_output")
        .expect("reference dump must contain result_output (unified-mode dumper)");
    let ro_f32 = as_f32(ro);
    assert_eq!(ro.ne[0] as usize, n_vocab, "result_output vocab width");
    let ro_rows = ro_f32.len() / n_vocab;
    assert!(
        ro_rows >= c,
        "result_output has {ro_rows} rows, need >= {c}"
    );
    let ro_canvas = &ro_f32[(ro_rows - c) * n_vocab..];
    let st_logits = compare_f32(&out.logits, ro_canvas, atol, rtol);
    eprintln!(
        "canvas logits: max_abs={:.3e} non_bit_exact={}/{} within={}",
        st_logits.max_abs, st_logits.non_bit_exact, st_logits.values, st_logits.within
    );
    all_within &= st_logits.within;
    sections.push(note("canvas_logits", &st_logits, &mut first_divergent));

    // optional per-layer ladder (localizes any logits divergence)
    let mut topk_sets_equal = true;
    if let Some(trace) = &out.trace {
        let n = p + c;
        let mut cmp_named = |name: String, ours: &[f32]| {
            if let Some(t) = by_name.get(name.as_str()) {
                let theirs = as_f32(t);
                // shape-agnostic: compare the common prefix when the ref
                // carries fewer rows (e.g. one-row result_norm dumps)
                let len = ours.len().min(theirs.len());
                let st = compare_f32(&ours[..len], &theirs[..len], atol, rtol);
                all_within &= st.within;
                sections.push(note(&name, &st, &mut first_divergent));
            }
        };
        cmp_named("inp_region".into(), &trace.inp_scaled);
        for (l, lt) in trace.layers.iter().enumerate() {
            cmp_named(format!("Kcur_pos-{l}"), &lt.k);
            cmp_named(format!("Vcur_normed-{l}"), &lt.v);
            cmp_named(format!("attn_out-{l}"), &lt.attn_out);
            cmp_named(format!("ffn_moe_logits-{l}"), &lt.moe_logits);
            cmp_named(format!("l_out-{l}"), &lt.out_scaled);
        }
        cmp_named("result_norm".into(), &trace.result_norm_all);

        // expert selections: per-position SET equality (ref dumps argsort
        // views with a row stride; first k entries are the selection)
        let k = rt.n_expert_used();
        for (l, lt) in trace.layers.iter().enumerate() {
            if let Some(t) = by_name.get(format!("ffn_moe_topk-{l}").as_str()) {
                let raw = as_i32(t);
                let stride = if raw.len() == n * k {
                    k
                } else {
                    (raw.len() - k) / (n - 1)
                };
                for pos in 0..n {
                    let mut a: Vec<i32> = raw[pos * stride..pos * stride + k].to_vec();
                    let mut b: Vec<i32> = lt.moe_topk[pos * k..(pos + 1) * k].to_vec();
                    a.sort_unstable();
                    b.sort_unstable();
                    if a != b {
                        topk_sets_equal = false;
                        eprintln!("topk set mismatch at layer {l} pos {pos}: {a:?} vs {b:?}");
                    }
                }
            }
        }
    }

    // ---- gate 3: one EB sampler step from the SAME logits ----
    let eb_dir = PathBuf::from(eb_dir);
    let eb_argmax = read_i32_file(&eb_dir.join("eb-argmax.i32"));
    let eb_entropy = read_f32_file(&eb_dir.join("eb-entropy.f32"));
    let eb_denoiser = read_i32_file(&eb_dir.join("eb-denoiser.i32"));
    let eb_accepted: Vec<u8> = std::fs::read(eb_dir.join("eb-accepted.u8")).expect("eb-accepted");
    let eb_next = read_i32_file(&eb_dir.join("eb-next-canvas.i32"));

    let step = DgEncoderRuntime::eb_step(
        &out.logits,
        n_vocab,
        0,
        s_steps,
        t_min,
        t_max,
        bound,
        &draws.u[0],
        &draws.renoise[0],
    );
    let eb_argmax_ok = step.argmax == eb_argmax;
    let eb_entropy_st = compare_f32(&step.entropy, &eb_entropy, atol, rtol);
    let eb_denoiser_ok = step.denoiser == eb_denoiser;
    let eb_accepted_ok = step
        .accepted
        .iter()
        .zip(&eb_accepted)
        .all(|(&a, &b)| a == (b != 0));
    let eb_next_ok = step.next_canvas == eb_next;
    let n_accepted = step.accepted.iter().filter(|&&a| a).count();
    eprintln!(
        "EB step 0: argmax={eb_argmax_ok} entropy(max_abs={:.3e}, non_bit_exact={}) \
         denoiser={eb_denoiser_ok} accepted={eb_accepted_ok} (n={n_accepted}) next={eb_next_ok}",
        eb_entropy_st.max_abs, eb_entropy_st.non_bit_exact
    );
    all_within &= eb_entropy_st.within;
    sections.push(note("eb_entropy", &eb_entropy_st, &mut first_divergent));

    let discrete_ok = rng_canvas_ok
        && rng_u_ok
        && rng_renoise_ok
        && topk_sets_equal
        && eb_argmax_ok
        && eb_denoiser_ok
        && eb_accepted_ok
        && eb_next_ok;
    let pass = all_within && discrete_ok;

    let report = format!
        (
        "{{\n  \"gate\": \"dg-phase3-single-denoise-step\",\n  \"pin\": \"{}\",\n  \"gguf\": \"{}\",\n  \
         \"seed\": {seed},\n  \"P\": {p},\n  \"C\": {c},\n  \"n_vocab\": {n_vocab},\n  \
         \"eb_params\": {{\"S\": {s_steps}, \"t_min\": {t_min}, \"t_max\": {t_max}, \"entropy_bound\": {bound}}},\n  \
         \"atol\": {atol},\n  \"rtol\": {rtol},\n  \
         \"rng\": {{\"canvas_init\": {rng_canvas_ok}, \"u\": {rng_u_ok}, \"renoise\": {rng_renoise_ok}}},\n  \
         \"eb\": {{\"argmax\": {eb_argmax_ok}, \"denoiser\": {eb_denoiser_ok}, \"accepted\": {eb_accepted_ok}, \
         \"next_canvas\": {eb_next_ok}, \"n_accepted\": {n_accepted}}},\n  \
         \"topk_sets_equal\": {topk_sets_equal},\n  \"first_divergent\": \"{}\",\n  \
         \"sections\": [\n    {}\n  ],\n  \"pass\": {pass}\n}}\n",
        std::env::var("CAMELID_DG_PIN_SHA").unwrap_or_else(|_| "UNRECORDED".to_string()),
        Path::new(&gguf_path)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
        first_divergent.clone().unwrap_or_else(|| "none".to_string()),
        sections.join(",\n    "),
    );
    if let Ok(out_path) = std::env::var("CAMELID_DG_DEC_OUT") {
        let mut f = std::fs::File::create(&out_path).expect("create compare.json");
        f.write_all(report.as_bytes()).expect("write compare.json");
        eprintln!("wrote {out_path}");
    }
    eprintln!("{report}");
    assert!(
        pass,
        "Phase 3 gate FAILED (first divergent: {})",
        first_divergent.unwrap_or_else(|| "discrete output".to_string())
    );
}
