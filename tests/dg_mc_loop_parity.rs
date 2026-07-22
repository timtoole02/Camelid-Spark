//! DiffusionGemma lane Phase 5 gate: multi-canvas (block-autoregressive)
//! generation parity against the pinned llama.cpp reference — the vocab's
//! end-of-generation id set, every block's prefix and per-step EB outputs,
//! each block's final canvas and trim cut, the committed-prefix chaining,
//! the stop reason, and the final response tokens + text.
//!
//! Env-gated: skips unless `CAMELID_DG_GGUF`, `CAMELID_DG_MC_REF` (a
//! directory produced by `scripts/dg-mc-loop.cpp`) and `CAMELID_DG_MC_IDS`
//! are set.
//!
//! Contract: discrete outputs match EXACTLY; entropies bit-exact
//! (atol=0/rtol=0); the EOG set camelid derives from the GGUF must equal
//! the reference vocab's set (fail closed on any difference).

use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};

use camelid::diffusion_gemma::{DgEbParams, DgEncoderRuntime};
use camelid::gguf::read_metadata;
use camelid::tokenizer::Tokenizer;

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

fn meta_str(meta: &str, key: &str) -> String {
    let pat = format!("\"{key}\":\"");
    let s = meta
        .find(&pat)
        .map(|i| i + pat.len())
        .expect("meta str key");
    let e = meta[s..].find('"').expect("meta str close") + s;
    meta[s..e].to_string()
}

/// The reference's end-of-generation set, mirrored from llama-vocab.cpp at
/// the pin: GGUF special eos/eot/eom + fim pad/rep/sep ids, plus the
/// text-matched EOG list (the gemma4 entries and the generic ones).
fn camelid_eog_set(gguf: &camelid::gguf::GgufFile, tok: &Tokenizer) -> HashSet<i32> {
    let mut set = HashSet::new();
    for key in [
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.eot_token_id",
        "tokenizer.ggml.eom_token_id",
        "tokenizer.ggml.fim_pad_token_id",
        "tokenizer.ggml.fim_rep_token_id",
        "tokenizer.ggml.fim_sep_token_id",
    ] {
        if let Some(id) = gguf.metadata_u32(key) {
            set.insert(id as i32);
        }
    }
    for text in [
        "<|eot_id|>",
        "<|im_end|>",
        "<|end|>",
        "<|return|>",
        "<|call|>",
        "<|flush|>",
        "<|calls|>",
        "<end_of_turn>",
        "<|endoftext|>",
        "</s>",
        "<|eom_id|>",
        "<EOT>",
        "_<EOT>",
        "[EOT]",
        "[EOS]",
        "<|end_of_text|>",
        "<end_of_utterance>",
        "<eos>",
        "<turn|>",
        "<|tool_response>",
    ] {
        if let Some(id) = tok.token_id(text) {
            set.insert(id as i32);
        }
    }
    // gemma4/paddleocr workaround at the pin (llama-vocab.cpp): when the set
    // contains BOTH "<|tool_response>" and "</s>", "</s>" is removed
    if let (Some(tool_resp), Some(s_tok)) = (tok.token_id("<|tool_response>"), tok.token_id("</s>"))
    {
        if set.contains(&(tool_resp as i32)) {
            set.remove(&(s_tok as i32));
        }
    }
    set
}

#[test]
fn dg_multi_canvas_loop_matches_pinned_llamacpp() {
    let (Ok(gguf_path), Ok(ref_dir), Ok(ids_path)) = (
        std::env::var("CAMELID_DG_GGUF"),
        std::env::var("CAMELID_DG_MC_REF"),
        std::env::var("CAMELID_DG_MC_IDS"),
    ) else {
        eprintln!("skipping: CAMELID_DG_GGUF / CAMELID_DG_MC_REF / CAMELID_DG_MC_IDS not set");
        return;
    };
    let ref_dir = PathBuf::from(ref_dir);
    let meta = std::fs::read_to_string(ref_dir.join("mc-meta.json")).expect("mc-meta.json");
    let seed = meta_i64(&meta, "seed") as u32;
    let s_steps = meta_i64(&meta, "S") as i32;
    let n_blocks = meta_i64(&meta, "n_blocks") as i32;
    let max_ub = meta_i64(&meta, "max_ub") as i32;
    let ref_blocks = meta_i64(&meta, "executed_blocks") as usize;
    let ref_stop = meta_str(&meta, "stop");

    let prompt: Vec<u32> = read_i32(Path::new(&ids_path))
        .into_iter()
        .map(|v| v as u32)
        .collect();

    let gguf = read_metadata(Path::new(&gguf_path)).expect("gguf metadata");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let eog = camelid_eog_set(&gguf, &tok);
    let ref_eog: HashSet<i32> = read_i32(&ref_dir.join("eog-ids.i32")).into_iter().collect();
    assert_eq!(
        eog, ref_eog,
        "camelid EOG set != reference vocab EOG set (fail closed)"
    );
    eprintln!("EOG set verified: {} ids", eog.len());

    eprintln!("loading runtime (lazy mmap)...");
    let rt = DgEncoderRuntime::load(Path::new(&gguf_path)).expect("load runtime");
    let params = DgEbParams {
        seed,
        max_steps: s_steps,
        ..DgEbParams::default()
    };

    let mut failures: Vec<String> = Vec::new();
    let t0 = std::time::Instant::now();
    let (blocks, response, stop_reason) = rt
        .mc_generate(
            &prompt,
            &params,
            n_blocks,
            max_ub,
            &eog,
            |b, prefix, records, canvas, cut| {
                eprintln!(
                    "camelid block {b}: prefix={} steps={} cut={cut} ({:.0}s)",
                    prefix.len(),
                    records.len(),
                    t0.elapsed().as_secs_f32()
                );
                let fail_start = failures.len();
                let pre_ref = read_i32(&ref_dir.join(format!("block{b}-prefix.i32")));
                if prefix.iter().map(|&t| t as i32).collect::<Vec<_>>() != pre_ref {
                    failures.push(format!("block {b}: prefix differs"));
                }
                let bmeta = std::fs::read_to_string(ref_dir.join(format!("block{b}-meta.json")))
                    .expect("block meta");
                let ref_exec = meta_i64(&bmeta, "executed") as usize;
                if records.len() != ref_exec {
                    failures.push(format!(
                        "block {b}: executed steps {} vs reference {ref_exec}",
                        records.len()
                    ));
                }
                for rec in records {
                    let ss = rec.step_idx.to_string();
                    let chk = |name: &str, ours: &[i32], failures: &mut Vec<String>| {
                        let theirs =
                            read_i32(&ref_dir.join(format!("block{b}-{name}-step{ss}.i32")));
                        if ours != theirs.as_slice() {
                            failures.push(format!("block {b} step {ss} {name} differs"));
                        }
                    };
                    chk("argmax", &rec.step.argmax, &mut failures);
                    chk("denoiser", &rec.step.denoiser, &mut failures);
                    chk("next", &rec.step.next_canvas, &mut failures);
                    let ent_ref = read_f32(&ref_dir.join(format!("block{b}-entropy-step{ss}.f32")));
                    if rec
                        .step
                        .entropy
                        .iter()
                        .zip(&ent_ref)
                        .any(|(a, c)| a.to_bits() != c.to_bits())
                    {
                        failures.push(format!("block {b} step {ss} entropy not bit-exact"));
                    }
                    let acc_ref: Vec<u8> =
                        std::fs::read(ref_dir.join(format!("block{b}-accepted-step{ss}.u8")))
                            .expect("accepted");
                    if !rec
                        .step
                        .accepted
                        .iter()
                        .zip(&acc_ref)
                        .all(|(&a, &r)| a == (r != 0))
                    {
                        failures.push(format!("block {b} step {ss} accepted differs"));
                    }
                }
                let canvas_ref = read_i32(&ref_dir.join(format!("block{b}-final-canvas.i32")));
                if canvas != canvas_ref.as_slice() {
                    failures.push(format!("block {b}: final canvas differs"));
                }
                let cut_meta = std::fs::read_to_string(ref_dir.join(format!("block{b}-cut.json")))
                    .expect("cut meta");
                let ref_cut = meta_i64(&cut_meta, "cut") as usize;
                if cut != ref_cut {
                    failures.push(format!("block {b}: cut {cut} vs reference {ref_cut}"));
                }
                // Durable per-block verdict: written the moment this block's
                // comparison completes, so a long run interrupted in a later
                // block still leaves proof of the blocks that did finish.
                let block_fails = &failures[fail_start..];
                let block_pass = block_fails.is_empty();
                eprintln!(
                    "BLOCK {b} VERDICT: {} ({} failures)",
                    if block_pass { "PASS" } else { "FAIL" },
                    block_fails.len()
                );
                if let Ok(out_path) = std::env::var("CAMELID_DG_MC_OUT") {
                    let vp = format!("{out_path}.block{b}.json");
                    let body = format!(
                        "{{\"block\": {b}, \"executed\": {}, \"cut\": {cut}, \
                         \"prefix_len\": {}, \"pass\": {block_pass}, \"failures\": [{}]}}\n",
                        records.len(),
                        prefix.len(),
                        block_fails
                            .iter()
                            .map(|f| format!("{f:?}"))
                            .collect::<Vec<_>>()
                            .join(", "),
                    );
                    let _ = std::fs::write(&vp, body);
                    eprintln!("wrote {vp}");
                }
            },
        )
        .expect("mc_generate");

    if blocks.len() != ref_blocks {
        failures.push(format!(
            "executed blocks differ: camelid {} vs reference {ref_blocks}",
            blocks.len()
        ));
    }
    if stop_reason != ref_stop {
        failures.push(format!(
            "stop reason differs: camelid {stop_reason} vs reference {ref_stop}"
        ));
    }
    let resp_ref = read_i32(&ref_dir.join("response.i32"));
    if response != resp_ref {
        failures.push("response token ids differ".into());
    }
    let text_ref = std::fs::read_to_string(ref_dir.join("response.txt")).expect("response.txt");
    let text = tok
        .decode(
            &response.iter().map(|&t| t as u32).collect::<Vec<_>>(),
            false,
        )
        .expect("decode response");
    if text != text_ref {
        failures.push(format!(
            "response text differs:\n camelid: {text:?}\n reference: {text_ref:?}"
        ));
    }

    let pass = failures.is_empty();
    let report = format!(
        "{{\n  \"gate\": \"dg-phase5-multi-canvas\",\n  \"pin\": \"{}\",\n  \"seed\": {seed},\n  \
         \"S\": {s_steps},\n  \"n_blocks\": {n_blocks},\n  \"max_ub\": {max_ub},\n  \
         \"executed_blocks\": {},\n  \"stop\": \"{stop_reason}\",\n  \"response_len\": {},\n  \
         \"response_text\": {:?},\n  \"failures\": [{}],\n  \"pass\": {pass}\n}}\n",
        std::env::var("CAMELID_DG_PIN_SHA").unwrap_or_else(|_| "UNRECORDED".to_string()),
        blocks.len(),
        response.len(),
        text,
        failures
            .iter()
            .map(|f| format!("{f:?}"))
            .collect::<Vec<_>>()
            .join(", "),
    );
    if let Ok(out_path) = std::env::var("CAMELID_DG_MC_OUT") {
        let mut f = std::fs::File::create(&out_path).expect("create compare.json");
        f.write_all(report.as_bytes()).expect("write compare.json");
        eprintln!("wrote {out_path}");
    }
    eprintln!("{report}");
    assert!(pass, "Phase 5 gate FAILED: {failures:?}");
}
