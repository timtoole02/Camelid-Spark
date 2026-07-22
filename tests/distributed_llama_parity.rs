//! Phase 1 of the distributed parity lane: prove the dense-Llama decoder layer stack can
//! be partitioned into contiguous blocks and still produce **bitwise-identical** final
//! logits, in a single process, before any sockets exist (see DISTRIBUTED_RECON.md /
//! DECISIONS.md). The gate is token-identity to a single-node reference, not "it ran".
//!
//! Reference and split BOTH drive `forward_layer_range_from_hidden` +
//! `forward_final_norm_and_logits` so the only difference is *where the layer loop is cut*,
//! not which per-layer kernel runs (DECISIONS D4: the decode fast-path is a different
//! implementation than the chunk path; comparing across them would test two engines). All
//! sessions pin the CPU lane via `set_resident_paths_disabled(true)` so the comparison is
//! deterministic regardless of ambient `CAMELID_METAL_RESIDENT_DECODE`.
//!
//! Run: `CAMELID_LLAMA_GGUF=/path/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
//!       cargo test --test distributed_llama_parity -- --nocapture`
//! Skips (does not fail) when `CAMELID_LLAMA_GGUF` is unset, matching the Gemma 4 parity
//! tests' convention.

use std::path::PathBuf;

use camelid::{
    gguf::read_metadata,
    inference::{LlamaInferenceSession, LlamaLoadedWeights},
    model::{LlamaModelConfig, LlamaTensorBinding},
    tensor::{CpuTensor, TensorStore},
    tokenizer::Tokenizer,
};

const PROMPT: &str = "hello";
const STEPS: usize = 8;

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i as u32;
        }
    }
    best
}

/// The final position's logits row out of a `[seq, vocab]` logits tensor.
fn last_row(logits: &CpuTensor) -> Vec<f32> {
    let vocab = *logits.shape.dims.last().expect("logits have a vocab dim");
    let seq = logits.data.len() / vocab;
    logits.data[(seq - 1) * vocab..seq * vocab].to_vec()
}

fn open(
    path: &PathBuf,
) -> (
    camelid::gguf::GgufFile,
    LlamaModelConfig,
    LlamaTensorBinding,
) {
    let gguf = read_metadata(path).expect("read gguf metadata");
    let config = LlamaModelConfig::from_gguf(&gguf).expect("llama config");
    let binding = LlamaTensorBinding::bind(&gguf, &config).expect("tensor binding");
    (gguf, config, binding)
}

fn session(
    path: &PathBuf,
    gguf: &camelid::gguf::GgufFile,
    binding: &LlamaTensorBinding,
    range: Option<std::ops::Range<usize>>,
) -> LlamaInferenceSession {
    let store = TensorStore::open(path, gguf);
    let weights = LlamaLoadedWeights::load(&store, binding, range).expect("load weights");
    let config = LlamaModelConfig::from_gguf(gguf).expect("llama config");
    let mut s = LlamaInferenceSession::new(config, weights).expect("build inference session");
    // Pin the deterministic CPU lane (DECISIONS D4).
    s.set_resident_paths_disabled(true);
    s
}

/// Greedy generation through the FULL stack (single node). Returns per-step final logits
/// rows and the per-step argmax token ids.
fn run_full(full: &mut LlamaInferenceSession, prompt_ids: &[u32]) -> (Vec<Vec<f32>>, Vec<u32>) {
    let mut rows = Vec::new();
    let mut toks = Vec::new();

    let h = full
        .weights
        .token_embedding
        .embedding_lookup(prompt_ids, "emb")
        .unwrap();
    let out = full
        .forward_layer_range_from_hidden(&h, 0, prompt_ids.len())
        .unwrap();
    let logits = full.forward_final_norm_and_logits(&out).unwrap();
    let mut row = last_row(&logits);
    let mut next = argmax(&row);
    rows.push(row);
    toks.push(next);

    for step in 1..STEPS {
        let pos = prompt_ids.len() + (step - 1);
        let h = full
            .weights
            .token_embedding
            .embedding_lookup(&[next], "emb")
            .unwrap();
        let out = full.forward_layer_range_from_hidden(&h, pos, 1).unwrap();
        let logits = full.forward_final_norm_and_logits(&out).unwrap();
        row = last_row(&logits);
        next = argmax(&row);
        rows.push(row);
        toks.push(next);
    }
    (rows, toks)
}

/// Greedy generation through a TWO-shard split (coordinator owns embedding + [0,k);
/// shard owns [k,L) + final norm/output) entirely in-process, no sockets.
fn run_split(
    coord: &mut LlamaInferenceSession,
    shard: &mut LlamaInferenceSession,
    prompt_ids: &[u32],
) -> (Vec<Vec<f32>>, Vec<u32>) {
    let mut rows = Vec::new();
    let mut toks = Vec::new();

    let h = coord
        .weights
        .token_embedding
        .embedding_lookup(prompt_ids, "emb")
        .unwrap();
    let h = coord
        .forward_layer_range_from_hidden(&h, 0, prompt_ids.len())
        .unwrap();
    let h = shard
        .forward_layer_range_from_hidden(&h, 0, prompt_ids.len())
        .unwrap();
    let logits = shard.forward_final_norm_and_logits(&h).unwrap();
    let mut row = last_row(&logits);
    let mut next = argmax(&row);
    rows.push(row);
    toks.push(next);

    for step in 1..STEPS {
        let pos = prompt_ids.len() + (step - 1);
        let h = coord
            .weights
            .token_embedding
            .embedding_lookup(&[next], "emb")
            .unwrap();
        let h = coord.forward_layer_range_from_hidden(&h, pos, 1).unwrap();
        let h = shard.forward_layer_range_from_hidden(&h, pos, 1).unwrap();
        let logits = shard.forward_final_norm_and_logits(&h).unwrap();
        row = last_row(&logits);
        next = argmax(&row);
        rows.push(row);
        toks.push(next);
    }
    (rows, toks)
}

#[test]
fn inprocess_chained_partition_matches_full_stack_bitwise() {
    let Some(path) = std::env::var_os("CAMELID_LLAMA_GGUF").map(PathBuf::from) else {
        eprintln!("skipping: set CAMELID_LLAMA_GGUF to a dense-Llama Q8_0 GGUF to run");
        return;
    };

    let (gguf, config, binding) = open(&path);
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let prompt_ids: Vec<u32> = tokenizer
        .encode(PROMPT, true, false)
        .expect("encode prompt");
    assert!(!prompt_ids.is_empty(), "prompt tokenized to nothing");

    let layers = config.block_count as usize;
    assert!(layers >= 2, "need at least 2 layers to split");
    let k = layers / 2;

    // Reference: full stack on one node.
    let mut full = session(&path, &gguf, &binding, None);
    let (full_rows, full_toks) = run_full(&mut full, &prompt_ids);

    // Split: [0,k) coordinator + [k,L) shard, in-process.
    let mut coord = session(&path, &gguf, &binding, Some(0..k));
    let mut shard = session(&path, &gguf, &binding, Some(k..layers));
    let (split_rows, split_toks) = run_split(&mut coord, &mut shard, &prompt_ids);

    // Bitwise final-logits identity at every step. A single differing f32 bit is a finding,
    // not noise to smooth over (operating rule #6) — report the first divergence loudly.
    for (step, (fr, sr)) in full_rows.iter().zip(split_rows.iter()).enumerate() {
        assert_eq!(fr.len(), sr.len(), "step {step}: vocab width differs");
        for (i, (&a, &b)) in fr.iter().zip(sr.iter()).enumerate() {
            assert!(
                a.to_bits() == b.to_bits(),
                "step {step}: logit[{i}] diverged at the first differing bit: \
                 full={a} (0x{:08x}) split={b} (0x{:08x}) — split at layer k={k}/{layers}",
                a.to_bits(),
                b.to_bits()
            );
        }
    }

    assert_eq!(
        full_toks, split_toks,
        "greedy token trajectory must be identical across the layer split"
    );

    eprintln!(
        "OK: {} steps bitwise-identical across split k={}/{} (prompt {:?} -> tokens {:?})",
        STEPS, k, layers, prompt_ids, full_toks
    );
}
