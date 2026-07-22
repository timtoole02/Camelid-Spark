//! Phase 2 of the distributed parity lane: a loopback two-shard pipeline over a REAL TCP
//! hop (coordinator + one worker thread on 127.0.0.1, the production `cluster.rs` wire
//! protocol) must be token-identical to the single-node reference, and emit a sealed
//! [`DistributedParityReceipt`] with `first_divergent_generated_token_index == -1`
//! (see DISTRIBUTED_RECON.md / DECISIONS.md).
//!
//! Reference and both shards pin the CPU lane (`set_resident_paths_disabled(true)`,
//! DECISIONS D4) so this builds directly on the Phase 1 bitwise result: the wire carries
//! the same little-endian f32 activations losslessly, so the only question Phase 2 adds is
//! whether framing/positions survive the hop. The gate is the receipt, not "it ran".
//!
//! Run: `CAMELID_LLAMA_GGUF=/path/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
//!       cargo test --test distributed_llama_loopback_parity -- --nocapture`
//! Skips (does not fail) when `CAMELID_LLAMA_GGUF` is unset.

use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::mpsc;

use camelid::{
    cluster::{
        recv_activation_packet, recv_token_feedback, send_activation_packet, send_token_feedback,
    },
    gguf::read_metadata,
    inference::{LlamaInferenceSession, LlamaLoadedWeights},
    model::{LlamaModelConfig, LlamaTensorBinding},
    receipt::{
        distributed::{
            DistributedParityReceipt, DistributedRunRecord, ParityVerdict, TopologyNode,
        },
        sha256_file_hex, LaneIdentity,
    },
    tensor::{CpuTensor, TensorStore},
    tokenizer::Tokenizer,
};

const PROMPT: &str = "hello";
const MAX_TOKENS: usize = 50;

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

fn last_row_argmax(logits: &CpuTensor) -> u32 {
    let vocab = *logits.shape.dims.last().expect("vocab dim");
    let seq = logits.data.len() / vocab;
    argmax(&logits.data[(seq - 1) * vocab..seq * vocab])
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
    let mut s = LlamaInferenceSession::new(config, weights).expect("build session");
    s.set_resident_paths_disabled(true); // DECISIONS D4: deterministic CPU lane
    s
}

/// Single-node greedy reference: returns generated token ids (length MAX_TOKENS).
fn reference_generation(full: &mut LlamaInferenceSession, prompt_ids: &[u32]) -> Vec<u32> {
    let mut generated = Vec::with_capacity(MAX_TOKENS);

    let h = full
        .weights
        .token_embedding
        .embedding_lookup(prompt_ids, "emb")
        .unwrap();
    let out = full
        .forward_layer_range_from_hidden(&h, 0, prompt_ids.len())
        .unwrap();
    let logits = full.forward_final_norm_and_logits(&out).unwrap();
    generated.push(last_row_argmax(&logits));

    for step in 1..MAX_TOKENS {
        let pos = prompt_ids.len() + (step - 1);
        let last = *generated.last().unwrap();
        let h = full
            .weights
            .token_embedding
            .embedding_lookup(&[last], "emb")
            .unwrap();
        let out = full.forward_layer_range_from_hidden(&h, pos, 1).unwrap();
        let logits = full.forward_final_norm_and_logits(&out).unwrap();
        generated.push(last_row_argmax(&logits));
    }
    generated
}

/// The shard worker: owns layers `[k,L)` + the output head. Reconstructs each activation
/// from the wire, runs its layer block, computes the greedy next token, and ships it back.
/// Ends when the coordinator closes the connection.
fn run_worker(mut shard: LlamaInferenceSession, listener: TcpListener) {
    let (mut stream, _) = listener.accept().expect("worker accept");
    loop {
        let mut floats = Vec::new();
        let header = match recv_activation_packet(&mut stream, &mut floats) {
            Ok(h) => h,
            Err(_) => break, // coordinator hung up: run complete
        };
        let seq = header.seq_len as usize;
        let hidden_w = floats.len() / seq;
        let h = CpuTensor::from_f32("wire_activation", vec![seq, hidden_w], floats)
            .expect("rebuild activation tensor");
        let out = shard
            .forward_layer_range_from_hidden(&h, header.pos as usize, seq)
            .expect("shard forward");
        let logits = shard
            .forward_final_norm_and_logits(&out)
            .expect("shard final norm/logits");
        let token = last_row_argmax(&logits);
        send_token_feedback(&mut stream, token, false).expect("send token feedback");
    }
}

/// The coordinator: owns the embedding + layers `[0,k)`. Drives MAX_TOKENS greedy steps,
/// shipping activations to the worker and collecting fed-back tokens.
fn coordinator_generation(
    coord: &mut LlamaInferenceSession,
    port: u16,
    prompt_ids: &[u32],
) -> Vec<u32> {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect worker");
    let mut generated = Vec::with_capacity(MAX_TOKENS);

    let h = coord
        .weights
        .token_embedding
        .embedding_lookup(prompt_ids, "emb")
        .unwrap();
    let out = coord
        .forward_layer_range_from_hidden(&h, 0, prompt_ids.len())
        .unwrap();
    send_activation_packet(&mut stream, 0, prompt_ids.len() as u32, &out.data).unwrap();
    generated.push(recv_token_feedback(&mut stream).unwrap().token_id);

    for step in 1..MAX_TOKENS {
        let pos = prompt_ids.len() + (step - 1);
        let last = *generated.last().unwrap();
        let h = coord
            .weights
            .token_embedding
            .embedding_lookup(&[last], "emb")
            .unwrap();
        let out = coord.forward_layer_range_from_hidden(&h, pos, 1).unwrap();
        send_activation_packet(&mut stream, pos as u32, 1, &out.data).unwrap();
        generated.push(recv_token_feedback(&mut stream).unwrap().token_id);
    }

    // Closing the stream ends the worker loop.
    drop(stream);
    generated
}

#[test]
fn loopback_two_shard_is_token_identical_with_receipt() {
    let Some(path) = std::env::var_os("CAMELID_LLAMA_GGUF").map(PathBuf::from) else {
        eprintln!("skipping: set CAMELID_LLAMA_GGUF to a dense-Llama Q8_0 GGUF to run");
        return;
    };

    let gguf = read_metadata(&path).expect("gguf");
    let config = LlamaModelConfig::from_gguf(&gguf).expect("config");
    let binding = LlamaTensorBinding::bind(&gguf, &config).expect("binding");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let prompt_ids: Vec<u32> = tokenizer.encode(PROMPT, true, false).expect("encode");
    assert!(!prompt_ids.is_empty());

    let layers = config.block_count as usize;
    assert!(layers >= 2, "need >= 2 layers to split");
    let k = layers / 2;

    // Single-node reference.
    let mut full = session(&path, &gguf, &binding, None);
    let ref_gen = reference_generation(&mut full, &prompt_ids);
    let ref_text = tokenizer.decode(&ref_gen, true).expect("decode ref");

    // Distributed loopback: worker owns [k,L) on a background thread; coordinator owns
    // embedding + [0,k) and drives the pipeline over a real TCP hop.
    let worker_shard = session(&path, &gguf, &binding, Some(k..layers));
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind worker");
    let port = listener.local_addr().unwrap().port();
    let (tx, rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        tx.send(()).ok();
        run_worker(worker_shard, listener);
    });
    rx.recv().ok(); // worker is about to accept

    let mut coord = session(&path, &gguf, &binding, Some(0..k));
    let dist_gen = coordinator_generation(&mut coord, port, &prompt_ids);
    worker.join().expect("worker thread");
    let dist_text = tokenizer.decode(&dist_gen, true).expect("decode dist");

    // Spec gate: token-identical at 1, 5, and 50 generated tokens.
    for n in [1usize, 5, MAX_TOKENS] {
        assert_eq!(
            &ref_gen[..n],
            &dist_gen[..n],
            "distributed run diverged from single-node within the first {n} generated tokens"
        );
    }

    // Build + seal the parity receipt from the computed verdict (never hand-set).
    let verdict = ParityVerdict::compare(
        &prompt_ids,
        &ref_gen,
        &ref_text,
        &prompt_ids,
        &dist_gen,
        &dist_text,
    );
    assert!(
        verdict.is_token_identical(),
        "verdict not token-identical: {verdict:?}"
    );

    let sha = sha256_file_hex(&path).expect("gguf sha256");
    let lane = LaneIdentity::capture("tinyllama-1.1b-chat-q8", &path, &gguf, None, sha);
    let record = DistributedRunRecord {
        config_id: format!("loopback-2shard-llama-q8-k{k}-L{layers}"),
        lane,
        reference: "single-node-camelid".to_string(),
        prompt: PROMPT.to_string(),
        seed: None,
        temperature: 0.0,
        max_tokens: MAX_TOKENS as u32,
        topology: vec![
            TopologyNode::coordinator("coordinator", "127.0.0.1", Some([0, k as u32])),
            TopologyNode::shard(
                "shard-b",
                &format!("127.0.0.1:{port}"),
                [k as u32, layers as u32],
            ),
        ],
        prompt_token_ids: prompt_ids.clone(),
        generated_token_ids: dist_gen.clone(),
        generated_text: dist_text.clone(),
    };
    let receipt =
        DistributedParityReceipt::build(record, &verdict, "1970-01-01T00:00:00Z".to_string())
            .expect("build receipt");

    assert!(receipt.verify_self_digest().is_ok(), "receipt self-digest");
    assert!(receipt.is_validated(), "receipt must be validated");
    assert_eq!(receipt.first_divergent_generated_token_index, -1);
    assert_eq!(receipt.completion_tokens, MAX_TOKENS as u32);

    // Emit the artifact (spec: emit on every gated distributed run).
    let out_dir = std::env::temp_dir().join("camelid-distributed-receipts");
    std::fs::create_dir_all(&out_dir).ok();
    let out_path = out_dir.join(format!("{}.json", receipt.config_id));
    let json = serde_json::to_string_pretty(&receipt).expect("serialize receipt");
    if let Ok(mut f) = std::fs::File::create(&out_path) {
        f.write_all(json.as_bytes()).ok();
    }

    eprintln!(
        "OK: loopback 2-shard token-identical over TCP, {} tokens, k={}/{}\n  receipt_id={}\n  artifact={}\n{}",
        MAX_TOKENS,
        k,
        layers,
        receipt.receipt_id,
        out_path.display(),
        json
    );
}
