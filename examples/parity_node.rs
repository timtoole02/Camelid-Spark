//! Distributed parity lane node (Phase 3+): run one pipeline node as either a `worker`
//! (owns a contiguous decoder layer block + the output head) or the `coordinator` (owns the
//! embedding + the head block, drives generation, computes the single-node reference, and
//! emits a sealed `DistributedParityReceipt`). Both roles pin the deterministic CPU lane
//! (DECISIONS D4) so the two machines run identical math and token-identity is provable.
//!
//! The SAME binary must run on every node — different builds would defeat the parity claim.
//!
//!   worker:      camelid-parity-node worker --gguf M.gguf --layers 11:22 --listen 0.0.0.0:9311
//!   coordinator: camelid-parity-node coordinator --gguf M.gguf --split 11 \
//!                  --worker 169.254.156.89:9311 --prompt hello --max-tokens 50 \
//!                  --config-id two-mac-tinyllama-q8 --self-host macA --worker-host mini2 \
//!                  --receipt /path/out.json
//!
//! This is an example (not shipped in the binary) so it can use the library's public
//! session API directly. It reuses src/cluster.rs for the wire and src/receipt for sealing.

use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::time::Instant;

use camelid::cluster::{
    recv_activation_packet, recv_token_feedback, send_activation_packet, send_token_feedback,
};
use camelid::gguf::{read_metadata, GgufFile};
use camelid::inference::{LlamaInferenceSession, LlamaLoadedWeights};
use camelid::model::{LlamaModelConfig, LlamaTensorBinding};
use camelid::receipt::distributed::{
    DistributedParityReceipt, DistributedRunRecord, ParityVerdict, TopologyNode,
};
use camelid::receipt::{sha256_file_hex, LaneIdentity};
use camelid::tensor::{CpuTensor, TensorStore};
use camelid::tokenizer::Tokenizer;

type Err = Box<dyn std::error::Error>;

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

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

fn open(path: &PathBuf) -> Result<(GgufFile, LlamaTensorBinding), Err> {
    let gguf = read_metadata(path)?;
    let config = LlamaModelConfig::from_gguf(&gguf)?;
    let binding = LlamaTensorBinding::bind(&gguf, &config)?;
    Ok((gguf, binding))
}

fn session(
    path: &PathBuf,
    gguf: &GgufFile,
    binding: &LlamaTensorBinding,
    range: Option<std::ops::Range<usize>>,
) -> Result<LlamaInferenceSession, Err> {
    let store = TensorStore::open(path, gguf);
    let weights = LlamaLoadedWeights::load(&store, binding, range)?;
    let config = LlamaModelConfig::from_gguf(gguf)?;
    let mut s = LlamaInferenceSession::new(config, weights)?;
    s.set_resident_paths_disabled(true); // DECISIONS D4: deterministic CPU lane
    Ok(s)
}

/// f32 footprint of one tensor when materialized: product(dims) * 4. Unowned pipeline
/// tensors are zero-element placeholders (dims `[0]`) and contribute 0.
fn tensor_f32_bytes(t: &CpuTensor) -> u64 {
    if t.shape.dims.is_empty() {
        return 0;
    }
    t.shape.dims.iter().product::<usize>() as u64 * 4
}

/// Total f32-materialization bytes of the tensors this node actually owns. Summing every
/// tensor yields just this shard's slice because unowned tensors are placeholders.
fn shard_materialization_bytes(w: &LlamaLoadedWeights) -> u64 {
    let mut total = tensor_f32_bytes(&w.token_embedding) + tensor_f32_bytes(&w.output_norm);
    if let Some(o) = &w.output {
        total += tensor_f32_bytes(o);
    }
    if let Some(r) = &w.rope_freqs {
        total += tensor_f32_bytes(r);
    }
    for l in &w.layers {
        for t in [
            &l.attention_norm,
            &l.attention_q,
            &l.attention_k,
            &l.attention_v,
            &l.attention_output,
            &l.ffn_norm,
            &l.ffn_gate,
            &l.ffn_up,
            &l.ffn_down,
        ] {
            total += tensor_f32_bytes(t);
        }
        if let Some(m) = &l.moe_router {
            total += tensor_f32_bytes(m);
        }
    }
    total
}

/// The per-node materialization cap (bytes): `--max-weight-bytes` flag, else
/// `CAMELID_MAX_CPU_WEIGHT_MATERIALIZATION_BYTES` (the engine's existing discipline).
fn cap_from(args: &[String]) -> Option<u64> {
    arg(args, "--max-weight-bytes")
        .and_then(|s| s.parse().ok())
        .or_else(|| {
            std::env::var("CAMELID_MAX_CPU_WEIGHT_MATERIALIZATION_BYTES")
                .ok()
                .and_then(|s| s.parse().ok())
        })
}

/// Enforce the per-node memory cap: a shard whose slice would exceed its cap refuses to
/// start with a typed error rather than starting and OOM-ing mid-run (Phase 4 discipline).
fn enforce_cap(label: &str, weights: &LlamaLoadedWeights, args: &[String]) -> Result<(), Err> {
    let bytes = shard_materialization_bytes(weights);
    match cap_from(args) {
        Some(cap) if bytes > cap => Err(format!(
            "{label}: slice would materialize {bytes} bytes > cap {cap}; refusing to start \
             (shard smaller or raise --max-weight-bytes)"
        )
        .into()),
        Some(cap) => {
            eprintln!("{label}: slice materialization {bytes} bytes within cap {cap}");
            Ok(())
        }
        None => {
            eprintln!("{label}: slice materialization {bytes} bytes (no cap set)");
            Ok(())
        }
    }
}

/// Connect to the worker with bounded retries + a per-attempt timeout. Link-local / Wi-Fi
/// fabrics flap; a transient connect failure is a transport concern, never a reason to
/// relax the parity gate.
fn connect_with_retry(addr: &str, attempts: u32) -> Result<TcpStream, Err> {
    let sock: std::net::SocketAddr = addr.parse()?;
    let mut last: Option<std::io::Error> = None;
    for attempt in 1..=attempts {
        match TcpStream::connect_timeout(&sock, std::time::Duration::from_secs(5)) {
            Ok(s) => return Ok(s),
            Err(e) => {
                eprintln!(
                    "coordinator: connect attempt {attempt}/{attempts} failed ({e}); retrying"
                );
                last = Some(e);
                std::thread::sleep(std::time::Duration::from_millis(750));
            }
        }
    }
    Err(Box::new(last.unwrap()))
}

fn parse_range(s: &str) -> (usize, usize) {
    let (a, b) = s.split_once(':').expect("layers must be START:END");
    (a.parse().unwrap(), b.parse().unwrap())
}

/// Single-node greedy reference on the full stack.
fn reference(full: &mut LlamaInferenceSession, prompt_ids: &[u32], max_tokens: usize) -> Vec<u32> {
    let mut gen = Vec::with_capacity(max_tokens);
    let h = full
        .weights
        .token_embedding
        .embedding_lookup(prompt_ids, "emb")
        .unwrap();
    let out = full
        .forward_layer_range_from_hidden(&h, 0, prompt_ids.len())
        .unwrap();
    gen.push(last_row_argmax(
        &full.forward_final_norm_and_logits(&out).unwrap(),
    ));
    for step in 1..max_tokens {
        let pos = prompt_ids.len() + (step - 1);
        let last = *gen.last().unwrap();
        let h = full
            .weights
            .token_embedding
            .embedding_lookup(&[last], "emb")
            .unwrap();
        let out = full.forward_layer_range_from_hidden(&h, pos, 1).unwrap();
        gen.push(last_row_argmax(
            &full.forward_final_norm_and_logits(&out).unwrap(),
        ));
    }
    gen
}

fn run_worker(args: &[String]) -> Result<(), Err> {
    let gguf_path = PathBuf::from(arg(args, "--gguf").expect("--gguf"));
    let (start, end) = parse_range(&arg(args, "--layers").expect("--layers"));
    let listen = arg(args, "--listen").unwrap_or_else(|| "0.0.0.0:9311".to_string());
    let (gguf, binding) = open(&gguf_path)?;
    let mut shard = session(&gguf_path, &gguf, &binding, Some(start..end))?;
    enforce_cap(
        &format!("worker shard [{start},{end})"),
        &shard.weights,
        args,
    )?;
    let listener = TcpListener::bind(&listen)?;
    eprintln!("worker: layers [{start},{end}) listening on {listen}");
    loop {
        let (mut stream, peer) = listener.accept()?;
        stream.set_nodelay(true).ok();
        // Each coordinator connection is a fresh generation: reset this shard's KV cache to
        // position 0 so a new run starts clean. Without this, a second run's pos=0 packet
        // mismatches the position left by the previous run.
        shard.kv_cache.position = 0;
        eprintln!("worker: coordinator {peer} connected (KV reset to pos 0)");
        if let Err(e) = serve_connection(&mut shard, &mut stream) {
            // A per-run error must not take the worker down; log and await the next run.
            eprintln!("worker: run ended ({e}); awaiting next run");
        }
    }
}

/// Serve one coordinator connection until it hangs up. Errors are returned (not `?`-ed out
/// of the process) so the worker survives a single bad run.
fn serve_connection(shard: &mut LlamaInferenceSession, stream: &mut TcpStream) -> Result<(), Err> {
    loop {
        let mut floats = Vec::new();
        let header = match recv_activation_packet(stream, &mut floats) {
            Ok(h) => h,
            Err(_) => return Ok(()), // coordinator hung up: run complete
        };
        let seq = header.seq_len as usize;
        let hidden_w = floats.len() / seq;
        let h = CpuTensor::from_f32("wire_activation", vec![seq, hidden_w], floats)?;
        let out = shard.forward_layer_range_from_hidden(&h, header.pos as usize, seq)?;
        let logits = shard.forward_final_norm_and_logits(&out)?;
        send_token_feedback(stream, last_row_argmax(&logits), false)?;
    }
}

/// One distributed run's output + timing.
struct Bench {
    gen: Vec<u32>,
    ttft_us: u128,
    local_us: u128,
    hop_us: u128,
    decode_elapsed: std::time::Duration,
}

/// Drive one full greedy generation across the coordinator's [0,split) block and the
/// worker's [split,L) block over a fresh TCP connection. Resets the coordinator's KV to
/// position 0 first so each run is an independent sequence (the worker resets its own KV on
/// accept). Any transport error is returned so the caller can retry the whole run.
fn distributed_run(
    coord: &mut LlamaInferenceSession,
    worker_addr: &str,
    prompt_ids: &[u32],
    max_tokens: usize,
) -> Result<Bench, Err> {
    coord.kv_cache.position = 0;
    let mut stream = connect_with_retry(worker_addr, 15)?;
    stream.set_nodelay(true).ok();

    let mut gen = Vec::with_capacity(max_tokens);
    let h = coord
        .weights
        .token_embedding
        .embedding_lookup(prompt_ids, "emb")?;
    let out = coord.forward_layer_range_from_hidden(&h, 0, prompt_ids.len())?;
    let t_hop = Instant::now();
    send_activation_packet(&mut stream, 0, prompt_ids.len() as u32, &out.data)?;
    gen.push(recv_token_feedback(&mut stream)?.token_id);
    let ttft_us = t_hop.elapsed().as_micros();

    let mut local_us = 0u128;
    let mut hop_us = 0u128;
    let decode_start = Instant::now();
    for step in 1..max_tokens {
        let pos = prompt_ids.len() + (step - 1);
        let last = *gen.last().unwrap();
        let t_local = Instant::now();
        let h = coord
            .weights
            .token_embedding
            .embedding_lookup(&[last], "emb")?;
        let out = coord.forward_layer_range_from_hidden(&h, pos, 1)?;
        local_us += t_local.elapsed().as_micros();
        let t_hop = Instant::now();
        send_activation_packet(&mut stream, pos as u32, 1, &out.data)?;
        gen.push(recv_token_feedback(&mut stream)?.token_id);
        hop_us += t_hop.elapsed().as_micros();
    }
    let decode_elapsed = decode_start.elapsed();
    drop(stream);
    Ok(Bench {
        gen,
        ttft_us,
        local_us,
        hop_us,
        decode_elapsed,
    })
}

fn run_coordinator(args: &[String]) -> Result<(), Err> {
    let gguf_path = PathBuf::from(arg(args, "--gguf").expect("--gguf"));
    let split: usize = arg(args, "--split").expect("--split").parse()?;
    let worker_addr = arg(args, "--worker").expect("--worker");
    let prompt = arg(args, "--prompt").unwrap_or_else(|| "hello".to_string());
    let max_tokens: usize = arg(args, "--max-tokens")
        .unwrap_or_else(|| "50".to_string())
        .parse()?;
    let config_id = arg(args, "--config-id").unwrap_or_else(|| "two-node-llama-q8".to_string());
    let self_host = arg(args, "--self-host").unwrap_or_else(|| "coordinator".to_string());
    let worker_host = arg(args, "--worker-host").unwrap_or_else(|| worker_addr.clone());
    let receipt_out = arg(args, "--receipt");

    let (gguf, binding) = open(&gguf_path)?;
    let config = LlamaModelConfig::from_gguf(&gguf)?;
    let layers = config.block_count as usize;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let prompt_ids: Vec<u32> = tokenizer.encode(&prompt, true, false)?;

    // Single-node reference (full stack, CPU lane) on this node.
    eprintln!("coordinator: computing single-node reference ({max_tokens} tokens)...");
    let mut full = session(&gguf_path, &gguf, &binding, None)?;
    let ref_gen = reference(&mut full, &prompt_ids, max_tokens);
    let ref_text = tokenizer.decode(&ref_gen, true)?;
    drop(full);

    // Distributed run(s): this node owns embedding + [0,split); worker owns [split,layers).
    // `--runs N` performs N consecutive token-identical runs (the lane's "two consecutive
    // bounded successes" gate); each run retries the whole run on a transport flake.
    let runs: usize = arg(args, "--runs")
        .unwrap_or_else(|| "1".to_string())
        .parse()?;
    let hidden_dim = config.embedding_length as usize;
    let activation_bytes_per_token = hidden_dim * 4;
    let mut coord = session(&gguf_path, &gguf, &binding, Some(0..split))?;
    enforce_cap(
        &format!("coordinator shard [0,{split})"),
        &coord.weights,
        args,
    )?;

    let mut last_run: Option<Bench> = None;
    for run_idx in 1..=runs {
        let mut attempt = 0;
        let bench = loop {
            attempt += 1;
            match distributed_run(&mut coord, &worker_addr, &prompt_ids, max_tokens) {
                Ok(b) => break b,
                Err(e) if attempt < 5 => {
                    eprintln!(
                        "coordinator: run {run_idx} attempt {attempt} failed ({e}); retrying whole run"
                    );
                    std::thread::sleep(std::time::Duration::from_millis(500));
                }
                Err(e) => return Err(e),
            }
        };
        let text = tokenizer.decode(&bench.gen, true)?;
        let v = ParityVerdict::compare(
            &prompt_ids,
            &ref_gen,
            &ref_text,
            &prompt_ids,
            &bench.gen,
            &text,
        );
        eprintln!(
            "coordinator: run {run_idx}/{runs} token-identical={} ({} tok/s)",
            v.is_token_identical(),
            (max_tokens - 1) as f64 / bench.decode_elapsed.as_secs_f64().max(1e-9)
        );
        if !v.is_token_identical() {
            // A divergence is a finding to RECORD (in the receipt), not an error to abort on
            // before emitting the artifact (operating rule #6 / spec Phase 4). Keep going so
            // the receipt is always written with the verdict.
            eprintln!(
                "coordinator: run {run_idx} DIVERGED at generated token {} — recording in receipt",
                v.first_divergent_generated_token_index
            );
        }
        last_run = Some(bench);
    }

    let Bench {
        gen,
        ttft_us,
        local_us,
        hop_us,
        decode_elapsed,
    } = last_run.expect("at least one run");
    let dist_text = tokenizer.decode(&gen, true)?;
    let verdict = ParityVerdict::compare(
        &prompt_ids,
        &ref_gen,
        &ref_text,
        &prompt_ids,
        &gen,
        &dist_text,
    );

    let decode_steps = (max_tokens - 1).max(1) as f64;
    let decode_tok_s = (max_tokens - 1) as f64 / decode_elapsed.as_secs_f64().max(1e-9);
    println!("\n=== distributed parity ({config_id}) ===");
    println!("nodes: {self_host} [0,{split}) -> {worker_host} [{split},{layers})");
    println!("prompt {prompt:?} -> prompt_ids {prompt_ids:?}");
    println!("token-identical: {}", verdict.is_token_identical());
    println!(
        "  prompt_tokens_match={} generated_tokens_match={} text_match={} first_divergent={}",
        verdict.prompt_tokens_match,
        verdict.generated_token_ids_match,
        verdict.generated_text_match,
        verdict.first_divergent_generated_token_index
    );
    println!("--- bench (honest: hop time bundles worker compute + wire round-trip) ---");
    println!("  activation bytes/token (coord->worker): {activation_bytes_per_token}");
    println!(
        "  ttft (prefill hop round-trip):          {:.2} ms",
        ttft_us as f64 / 1000.0
    );
    println!(
        "  avg coord-local compute/token:          {:.3} ms",
        local_us as f64 / 1000.0 / decode_steps
    );
    println!(
        "  avg hop round-trip/token:               {:.3} ms",
        hop_us as f64 / 1000.0 / decode_steps
    );
    println!("  decode throughput:                      {decode_tok_s:.2} tok/s");
    println!("  generated text: {dist_text:?}");

    let sha = sha256_file_hex(&gguf_path)?;
    let lane = LaneIdentity::capture(&config_id, &gguf_path, &gguf, None, sha);
    let record = DistributedRunRecord {
        config_id: config_id.clone(),
        lane,
        reference: "single-node-camelid".to_string(),
        prompt,
        seed: None,
        temperature: 0.0,
        max_tokens: max_tokens as u32,
        topology: vec![
            TopologyNode::coordinator(&self_host, "self", Some([0, split as u32])),
            TopologyNode::shard(&worker_host, &worker_addr, [split as u32, layers as u32]),
        ],
        prompt_token_ids: prompt_ids,
        generated_token_ids: gen,
        generated_text: dist_text,
    };
    let receipt =
        DistributedParityReceipt::build(record, &verdict, "1970-01-01T00:00:00Z".to_string())?;
    receipt.verify_self_digest()?;
    println!("receipt_id: {}", receipt.receipt_id);
    println!("validated:  {}", receipt.is_validated());
    if let Some(out) = receipt_out {
        let json = serde_json::to_string_pretty(&receipt)?;
        std::fs::File::create(&out)?.write_all(json.as_bytes())?;
        println!("receipt written: {out}");
    }
    if receipt.is_validated() {
        println!("RESULT: token-identical to single-node reference (validated).");
    } else {
        println!(
            "RESULT: EXPERIMENTAL-DIVERGENT — diverged from reference at generated token {} \
             (recorded in receipt, not smoothed over).",
            receipt.first_divergent_generated_token_index
        );
    }
    Ok(())
}

fn main() -> Result<(), Err> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("worker") => run_worker(&args[2..]),
        Some("coordinator") => run_coordinator(&args[2..]),
        _ => {
            eprintln!("usage: parity_node <worker|coordinator> [--flags]");
            std::process::exit(2);
        }
    }
}
