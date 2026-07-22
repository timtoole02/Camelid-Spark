//! Phase 5 block-1 logit ladder (NOT a gate). Reproduces block 1 in isolation
//! — an EB loop on the committed block-1 prefix with seed 0 — and compares
//! camelid's FULL per-step canvas logits to the oracle's (dg-eb-loop.cpp dumps
//! `logits-step{k}.f32`). The Phase 5 gate only checks per-step argmax/entropy;
//! this catches a SUB-threshold logit diff (the self-conditioning input) that
//! the gate misses and that amplifies into the step-3 argmax flip. Finds the
//! FIRST step whose full logits diverge and its in-step location.
//!
//! Env: CAMELID_DG_GGUF, CAMELID_DG_B1_IDS (block1-prefix.i32),
//! CAMELID_DG_B1_REF (a dg-eb-loop dump dir). Set CAMELID_DG_EB_CAP=<k> to cap
//! executed steps (must match the oracle's DG_EB_CAP).

use std::path::Path;

use camelid::diffusion_gemma::{DgEbParams, DgEncoderRuntime};

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

#[test]
fn dg_block1_logit_ladder() {
    let (Ok(g), Ok(i), Ok(r)) = (
        std::env::var("CAMELID_DG_GGUF"),
        std::env::var("CAMELID_DG_B1_IDS"),
        std::env::var("CAMELID_DG_B1_REF"),
    ) else {
        eprintln!("skipping: CAMELID_DG_GGUF / CAMELID_DG_B1_IDS / CAMELID_DG_B1_REF not set");
        return;
    };
    let refd = Path::new(&r);
    let prompt: Vec<u32> = read_i32(Path::new(&i))
        .into_iter()
        .map(|v| v as u32)
        .collect();
    eprintln!(
        "loading runtime (lazy mmap); block-1 prefix P={}...",
        prompt.len()
    );
    let rt = DgEncoderRuntime::load(Path::new(&g)).expect("load");
    let params = DgEbParams::default(); // seed 0, S=48 — exactly block 1

    let mut first_div: Option<i32> = None;
    rt.eb_generate(&prompt, &params, |rec, logits| {
        let k = rec.step_idx;
        let nvocab = logits.len() / rec.step.argmax.len().max(1);
        // full canvas logits — the self-conditioning input for the next step
        let lp = refd.join(format!("logits-step{k}.f32"));
        if lp.exists() {
            let oracle = read_f32(&lp);
            let n = logits.len().min(oracle.len());
            let mut bad = 0usize;
            let mut first = usize::MAX;
            let mut maxabs = 0f32;
            for (idx, (a, b)) in logits[..n].iter().zip(&oracle[..n]).enumerate() {
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
                    "step {k}: LOGITS DIVERGE {bad}/{n} maxabs={maxabs:.3e} first(canvas-pos {fp}, vocab {fv})"
                );
                if first_div.is_none() {
                    first_div = Some(k);
                }
            } else {
                eprintln!("step {k}: logits BIT-EXACT ({n})");
            }
        } else {
            eprintln!("step {k}: (no oracle logits dump for this step)");
        }
        // cross-check the gate-visible discretes against the oracle per-step
        let ap = refd.join(format!("argmax-step{k}.i32"));
        let ep = refd.join(format!("entropy-step{k}.f32"));
        if ap.exists() && ep.exists() {
            let amax_eq = rec.step.argmax == read_i32(&ap);
            let ent_ref = read_f32(&ep);
            let ent_bad = rec
                .step
                .entropy
                .iter()
                .zip(&ent_ref)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            eprintln!(
                "  step {k}: argmax {}, entropy diffs {ent_bad}",
                if amax_eq { "EQ" } else { "NE" }
            );
        }
    })
    .expect("eb_generate");

    eprintln!(
        "FIRST DIVERGENT STEP (full logits): {}",
        first_div
            .map(|k| k.to_string())
            .unwrap_or_else(|| "none in capped range".into())
    );
}
