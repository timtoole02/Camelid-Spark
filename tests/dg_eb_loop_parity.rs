//! DiffusionGemma lane Phase 4 gate: full Entropy-Bound denoise-loop parity
//! against the pinned llama.cpp reference — every executed step's discrete
//! outputs (canvas-in, argmax, denoiser, accepted set, next canvas), the
//! per-step entropies, the raw canvas logits on the dumped steps, the
//! adaptive-stop trajectory, and the final canvas.
//!
//! Env-gated: skips unless `CAMELID_DG_GGUF` and `CAMELID_DG_EB_REF` (a
//! directory produced by `scripts/dg-eb-loop.cpp`: loop-meta.json +
//! per-step files) are set. `CAMELID_DG_EB_SEED` / `CAMELID_DG_EB_S`
//! default to 0 / the reference meta's S. Run via
//! `scripts/dg-eb-loop-parity.sh`.
//!
//! Contract: discrete outputs match EXACTLY; every compared f32 surface
//! must be bit-exact (atol=0/rtol=0 — the bar Phases 2 and 3 set).

use std::io::Write;
use std::path::{Path, PathBuf};

use camelid::diffusion_gemma::{DgEbParams, DgEncoderRuntime};

fn read_i32(path: &Path) -> Vec<i32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn meta_i64(meta: &str, key: &str) -> i64 {
    let pat = format!("\"{key}\":");
    let s = meta.find(&pat).map(|i| i + pat.len()).expect("meta key");
    meta[s..]
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '-')
        .collect::<String>()
        .parse()
        .expect("meta int")
}

#[test]
fn dg_eb_loop_matches_pinned_llamacpp() {
    let (Ok(gguf_path), Ok(ref_dir), Ok(ids_path)) = (
        std::env::var("CAMELID_DG_GGUF"),
        std::env::var("CAMELID_DG_EB_REF"),
        std::env::var("CAMELID_DG_EB_IDS"),
    ) else {
        eprintln!(
            "skipping: CAMELID_DG_GGUF / CAMELID_DG_EB_REF / CAMELID_DG_EB_IDS not set \
             (run scripts/dg-eb-loop-parity.sh)"
        );
        return;
    };
    let ref_dir = PathBuf::from(ref_dir);
    let meta = std::fs::read_to_string(ref_dir.join("loop-meta.json")).expect("loop-meta.json");
    let seed = meta_i64(&meta, "seed") as u32;
    let s_steps = meta_i64(&meta, "S") as i32;
    let executed_ref = meta_i64(&meta, "executed") as usize;
    let ref_finished = meta.contains("\"finished\":true");

    let prompt: Vec<u32> = read_i32(Path::new(&ids_path))
        .into_iter()
        .map(|v| v as u32)
        .collect();

    eprintln!("loading runtime (lazy mmap)...");
    let rt = DgEncoderRuntime::load(Path::new(&gguf_path)).expect("load runtime");
    let params = DgEbParams {
        seed,
        max_steps: s_steps,
        ..DgEbParams::default()
    };

    // collect per-step comparisons as we go (logits compared only on steps
    // the oracle dumped)
    let mut failures: Vec<String> = Vec::new();
    let mut steps_run = 0usize;
    let mut logits_bits_checked = 0usize;
    let t0 = std::time::Instant::now();
    let records = rt
        .eb_generate(&prompt, &params, |rec, logits| {
            let si = rec.step_idx;
            let ss = si.to_string();
            eprintln!(
                "camelid step {si}: t={} n_accepted={} entropy_sum={} finished={} ({:.0}s)",
                rec.step.t,
                rec.step.accepted.iter().filter(|&&a| a).count(),
                rec.step.entropy_sum,
                rec.finished,
                t0.elapsed().as_secs_f32()
            );
            let mut chk_i32 = |name: &str, ours: &[i32]| {
                let theirs = read_i32(&ref_dir.join(format!("{name}-step{ss}.i32")));
                if ours != theirs.as_slice() {
                    let bad = ours
                        .iter()
                        .zip(&theirs)
                        .position(|(a, b)| a != b)
                        .unwrap_or(0);
                    failures.push(format!(
                        "step {si} {name}: first mismatch at pos {bad} ({} vs {})",
                        ours[bad], theirs[bad]
                    ));
                }
            };
            chk_i32("canvas-in", &rec.canvas_in);
            chk_i32("argmax", &rec.step.argmax);
            chk_i32("denoiser", &rec.step.denoiser);
            chk_i32("next", &rec.step.next_canvas);
            let acc_ref: Vec<u8> =
                std::fs::read(ref_dir.join(format!("accepted-step{ss}.u8"))).expect("accepted");
            if !rec
                .step
                .accepted
                .iter()
                .zip(&acc_ref)
                .all(|(&a, &b)| a == (b != 0))
            {
                failures.push(format!("step {si} accepted set mismatch"));
            }
            let ent_ref = read_f32(&ref_dir.join(format!("entropy-step{ss}.f32")));
            let ent_bad = rec
                .step
                .entropy
                .iter()
                .zip(&ent_ref)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            if ent_bad > 0 {
                failures.push(format!("step {si} entropy: {ent_bad} values not bit-exact"));
            }
            let lf = ref_dir.join(format!("logits-step{ss}.f32"));
            if lf.exists() {
                let lref = read_f32(&lf);
                let bad = logits
                    .iter()
                    .zip(&lref)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                logits_bits_checked += lref.len();
                if bad > 0 {
                    failures.push(format!(
                        "step {si} canvas logits: {bad}/{} values not bit-exact",
                        lref.len()
                    ));
                }
            }
            steps_run += 1;
        })
        .expect("eb_generate");

    let camelid_finished = records.last().map(|r| r.finished).unwrap_or(false);
    if steps_run != executed_ref {
        failures.push(format!(
            "executed steps differ: camelid {steps_run} vs reference {executed_ref}"
        ));
    }
    if camelid_finished != ref_finished {
        failures.push(format!(
            "adaptive stop differs: camelid finished={camelid_finished} vs reference {ref_finished}"
        ));
    }

    let pass = failures.is_empty();
    let report = format!(
        "{{\n  \"gate\": \"dg-phase4-eb-loop\",\n  \"pin\": \"{}\",\n  \"seed\": {seed},\n  \
         \"S\": {s_steps},\n  \"executed\": {steps_run},\n  \"finished\": {camelid_finished},\n  \
         \"logits_values_bit_checked\": {logits_bits_checked},\n  \"failures\": [{}],\n  \
         \"pass\": {pass}\n}}\n",
        std::env::var("CAMELID_DG_PIN_SHA").unwrap_or_else(|_| "UNRECORDED".to_string()),
        failures
            .iter()
            .map(|f| format!("\"{f}\""))
            .collect::<Vec<_>>()
            .join(", "),
    );
    if let Ok(out_path) = std::env::var("CAMELID_DG_EB_OUT") {
        let mut f = std::fs::File::create(&out_path).expect("create compare.json");
        f.write_all(report.as_bytes()).expect("write compare.json");
        eprintln!("wrote {out_path}");
    }
    eprintln!("{report}");
    assert!(pass, "Phase 4 gate FAILED: {failures:?}");
}
