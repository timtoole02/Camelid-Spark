//! Diagnostic (env-gated): teacher-force a token sequence through the gemma4
//! CPU runtime and print the top-2 logits at each step — used to qualify
//! whether a cross-runtime argmax divergence is a knife-edge near-tie or a
//! real numeric break. Not a parity gate.
//!
//! Run: `CAMELID_GEMMA4_GGUF=/path/row.gguf \
//!       CAMELID_GEMMA4_PROBE_TOKENS=2,27832,236787,1535,5597,107,51423,236787 \
//!       cargo test --release --test gemma4_logit_probe -- --nocapture`

use std::path::PathBuf;

use camelid::gemma4_runtime::{Gemma4Runtime, Gemma4StepOutput};

#[test]
fn gemma4_teacher_forced_top2_logits() {
    let Some(model) = std::env::var_os("CAMELID_GEMMA4_GGUF").map(PathBuf::from) else {
        eprintln!("SKIP logit probe: set CAMELID_GEMMA4_GGUF");
        return;
    };
    let Ok(tokens) = std::env::var("CAMELID_GEMMA4_PROBE_TOKENS") else {
        eprintln!("SKIP logit probe: set CAMELID_GEMMA4_PROBE_TOKENS=id,id,...");
        return;
    };
    let tokens: Vec<u32> = tokens
        .split(',')
        .map(|t| t.trim().parse().expect("token id"))
        .collect();
    let runtime = Gemma4Runtime::load(&model).expect("load");
    let (mut kc, mut vc) = runtime.empty_kv_caches();
    for (pos, &tok) in tokens.iter().enumerate() {
        let logits = match runtime
            .step_range(tok, pos, None, &mut kc, &mut vc)
            .expect("step")
        {
            Gemma4StepOutput::Logits(l) => l,
            Gemma4StepOutput::Hidden(_) => unreachable!("full runtime"),
        };
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        let (a, b) = (idx[0], idx[1]);
        eprintln!(
            "pos {pos} fed {tok}: top1 {a} ({:.6}) top2 {b} ({:.6}) gap {:.6}",
            logits[a],
            logits[b],
            logits[a] - logits[b]
        );
        // Optional: full top-10 + specific requested ids, for cross-runtime
        // logit-structure comparison at a divergence position.
        if std::env::var("CAMELID_GEMMA4_PROBE_VERBOSE").is_ok_and(|v| v == "1") {
            let top: Vec<String> = idx[..10]
                .iter()
                .map(|&i| format!("{i}:{:.4}", logits[i]))
                .collect();
            eprintln!("  top10: {}", top.join(" "));
            if let Ok(ids) = std::env::var("CAMELID_GEMMA4_PROBE_REPORT_IDS") {
                let report: Vec<String> = ids
                    .split(',')
                    .filter_map(|t| t.trim().parse::<usize>().ok())
                    .map(|i| format!("{i}:{:.4}", logits[i]))
                    .collect();
                eprintln!("  report: {}", report.join(" "));
            }
        }
    }
}
