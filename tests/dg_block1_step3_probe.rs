//! Phase 5 block-1 step-3 probe (NOT a gate). The logit ladder showed block-1
//! steps 0-2 bit-exact then step 3 diverging catastrophically (maxabs ~5),
//! despite step-3's inputs (canvas-in, SC buffer = step-2 logits, temperature)
//! all matching the oracle per the gate. This feeds camelid the ORACLE's EXACT
//! step-3 inputs and runs the forward TWICE to settle the cause:
//!   run1 != run2            -> NON-DETERMINISM in the forward
//!   run1 == run2 != oracle  -> deterministic forward bug on this input
//!   run1 == run2 == oracle  -> camelid's OWN step-3 input differed (eb_step bug)
//!
//! Env: CAMELID_DG_GGUF, CAMELID_DG_B1_IDS (block1-prefix), CAMELID_DG_B1_REF
//! (the dg-eb-loop ladder dump: canvas-in-step3.i32, logits-step2.f32 [SC
//! buffer], logits-step3.f32 [expected]).

use std::path::Path;

use camelid::diffusion_gemma::{DgEncoderRuntime, DgScInput};

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

fn diff(a: &[f32], b: &[f32]) -> (usize, usize, f32) {
    let n = a.len().min(b.len());
    let (mut bad, mut first, mut maxabs) = (0usize, usize::MAX, 0f32);
    for (i, (x, y)) in a[..n].iter().zip(&b[..n]).enumerate() {
        if x.to_bits() != y.to_bits() {
            bad += 1;
            if first == usize::MAX {
                first = i;
            }
            maxabs = maxabs.max((x - y).abs());
        }
    }
    (bad, first, maxabs)
}

#[test]
fn dg_block1_step3_probe() {
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
    let canvas: Vec<u32> = read_i32(&refd.join("canvas-in-step3.i32"))
        .into_iter()
        .map(|v| v as u32)
        .collect();
    let sc_logits = read_f32(&refd.join("logits-step2.f32")); // step-2 logits = step-3 SC buffer
    let oracle_step3 = read_f32(&refd.join("logits-step3.f32"));

    // prev_temp_inv for step 3 = step-2's temp_inv — computed EXACTLY as
    // eb_step does (FUSED mul_add; the reference's `t_min + (t_max-t_min)*ratio`
    // contracts to fma under clang -ffp-contract=on, so camelid mirrors it with
    // mul_add). Using the plain `*` here gives a 1-ULP-different temp_inv that
    // does NOT match the real loop.
    let (t_min, t_max, s) = (0.4f32, 0.8f32, 48f32);
    let cur_step = s - 2.0;
    let t2 = (t_max - t_min).mul_add(cur_step / s, t_min); // fused, == eb_step
    let temp_inv = 1.0f32 / t2;
    eprintln!(
        "P={} C={} N={} sc_len={} temp_inv(step2)={temp_inv}",
        prompt.len(),
        canvas.len(),
        prompt.len() + canvas.len(),
        sc_logits.len()
    );

    let rt = DgEncoderRuntime::load(Path::new(&g)).expect("load");
    let sc = DgScInput {
        logits: &sc_logits,
        temp_inv,
        use_sc: 1.0,
    };

    eprintln!("forward run 1...");
    let r1 = rt
        .unified_forward_sc(&prompt, &canvas, Some(&sc), false)
        .expect("fwd1")
        .logits;
    eprintln!("forward run 2 (same inputs)...");
    let r2 = rt
        .unified_forward_sc(&prompt, &canvas, Some(&sc), false)
        .expect("fwd2")
        .logits;

    let (d12, f12, m12) = diff(&r1, &r2);
    let (d1o, f1o, m1o) = diff(&r1, &oracle_step3);
    eprintln!(
        "run1 vs run2 (DETERMINISM): {d12}/{} differ, first {f12}, maxabs {m12:.3e}",
        r1.len()
    );
    eprintln!(
        "run1 vs oracle step3:       {d1o}/{} differ, first {f1o}, maxabs {m1o:.3e}",
        r1.len()
    );

    let verdict = if d12 != 0 {
        "NON-DETERMINISTIC forward (run1 != run2)"
    } else if d1o != 0 {
        "DETERMINISTIC forward differs from oracle on this exact input"
    } else {
        "forward matches oracle on oracle inputs -> camelid's own step-3 input differs (eb_step)"
    };
    eprintln!("VERDICT: {verdict}");
}
