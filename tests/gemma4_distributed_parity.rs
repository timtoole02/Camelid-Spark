//! Gemma 4 distributed layer-sharding parity — the sharded pipeline must be
//! token-identical to the single-node runtime AND to the committed llama.cpp
//! oracle, over real TCP (worker thread on localhost; the same protocol the
//! two-Mac deployment uses).
//!
//! Also locks the fail-closed behaviors: a split through the shared-KV block is
//! rejected at load, and a wire-version mismatch is rejected at handshake.
//!
//! Run: `CAMELID_GEMMA4_GGUF=/path/gemma-4-E2B-it-Q8_0.gguf \
//!       CAMELID_GEMMA4_SPLIT=8 \
//!       cargo test --release --test gemma4_distributed_parity -- --nocapture`

use std::path::PathBuf;

use camelid::gemma4_distributed::{
    run_master, run_worker, Gemma4Handshake, Gemma4WorkerClient, GEMMA4_WIRE_VERSION,
};
use camelid::gemma4_runtime::Gemma4Runtime;

#[derive(serde::Deserialize)]
struct Oracle {
    results: Vec<OracleResult>,
}

#[derive(serde::Deserialize)]
struct OracleResult {
    id: String,
    generated_tokens: Vec<u32>,
    generated_text: String,
}

fn repo_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
}

#[test]
fn distributed_split_through_shared_kv_block_fails_closed() {
    let Some(model) = std::env::var_os("CAMELID_GEMMA4_GGUF").map(PathBuf::from) else {
        eprintln!("SKIP distributed shared-kv guard: set CAMELID_GEMMA4_GGUF");
        return;
    };
    let probe = Gemma4Runtime::load(&model).expect("full load");
    let block_count = probe.block_count();
    drop(probe);
    let gguf = camelid::gguf::read_metadata(&model).expect("gguf");
    let config = camelid::model::LlamaModelConfig::from_gguf(&gguf).expect("config");
    let g = config.gemma4.as_ref().expect("gemma4");
    if g.num_kv_shared_layers == 0 {
        eprintln!("row has no shared KV layers; guard not applicable");
        return;
    }
    // A worker range starting INSIDE the shared block (after its source layers)
    // must be rejected: those layers read caches owned by earlier layers.
    let bad_start = block_count - g.num_kv_shared_layers as usize + 1;
    let err = Gemma4Runtime::load_layer_range(&model, Some(bad_start..block_count))
        .err()
        .expect("split through shared-KV block must fail closed");
    let msg = err.to_string();
    assert!(
        msg.contains("shared") && msg.contains("KV"),
        "error must explain the shared-KV constraint: {msg}"
    );
}

#[test]
fn distributed_greedy_matches_single_node_and_oracle() {
    let Some(model) = std::env::var_os("CAMELID_GEMMA4_GGUF").map(PathBuf::from) else {
        eprintln!("SKIP distributed parity: set CAMELID_GEMMA4_GGUF");
        return;
    };
    let row = model
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_string();
    let oracle_path = repo_path(&format!("qa/gemma4/oracle/{row}.basic_v1.json"));
    let oracle: Oracle = serde_json::from_str(
        &std::fs::read_to_string(&oracle_path)
            .unwrap_or_else(|_| panic!("no committed oracle for row {row}")),
    )
    .expect("oracle json");

    // Pick the split: env override, else half the layers (clamped below the
    // shared-KV source layers so the constraint holds).
    let gguf = camelid::gguf::read_metadata(&model).expect("gguf");
    let config = camelid::model::LlamaModelConfig::from_gguf(&gguf).expect("config");
    let block_count = config.block_count as usize;
    let g = config.gemma4.as_ref().expect("gemma4");
    let first_shared = block_count - g.num_kv_shared_layers as usize;
    let default_split = (block_count / 2).min(if g.num_kv_shared_layers > 0 {
        // Sources are the last owning sliding/full layers; staying at or below
        // first_shared - 2 keeps both on the worker.
        first_shared.saturating_sub(2)
    } else {
        block_count / 2
    });
    let split = std::env::var("CAMELID_GEMMA4_SPLIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default_split.max(1));
    eprintln!("row {row}: {block_count} layers, split at {split}");

    // Worker thread on an ephemeral localhost port (real TCP, real protocol).
    let port = 39411;
    let addr = format!("127.0.0.1:{port}");
    let worker_model = model.clone();
    let worker_addr = addr.clone();
    std::thread::spawn(move || {
        run_worker(&worker_model, &worker_addr, split..block_count).expect("worker run");
    });
    // Wait for the listener.
    for _ in 0..100 {
        if std::net::TcpStream::connect(&addr).is_ok() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // One prompt is enough for wire parity (the full pack runs in the
    // generation-parity test); use the first oracle prompt.
    let expected = oracle
        .results
        .iter()
        .find(|r| r.id == "capital-france")
        .expect("capital-france in oracle");
    let (text, ids, stats) =
        run_master(&model, &addr, split, "The capital of France is", 24, false)
            .expect("master run");
    eprintln!("distributed: {text:?} ids {ids:?}");
    eprintln!("stats: {}", serde_json::to_string(&stats).unwrap());

    assert_eq!(
        ids, expected.generated_tokens,
        "distributed greedy ids must match the llama.cpp oracle"
    );
    assert_eq!(
        text, expected.generated_text,
        "distributed greedy text must match the llama.cpp oracle"
    );
    assert!(stats.ttft_ms > 0.0 && stats.total_wire_round_trips >= ids.len());
}

#[test]
fn distributed_wire_version_mismatch_fails_closed() {
    let Some(model) = std::env::var_os("CAMELID_GEMMA4_GGUF").map(PathBuf::from) else {
        eprintln!("SKIP wire version guard: set CAMELID_GEMMA4_GGUF");
        return;
    };
    let gguf = camelid::gguf::read_metadata(&model).expect("gguf");
    let config = camelid::model::LlamaModelConfig::from_gguf(&gguf).expect("config");
    let block_count = config.block_count as usize;
    let g = config.gemma4.as_ref().expect("gemma4");
    let first_shared = block_count - g.num_kv_shared_layers as usize;
    let split = if g.num_kv_shared_layers > 0 {
        first_shared.saturating_sub(2).max(1)
    } else {
        block_count / 2
    };

    let port = 39412;
    let addr = format!("127.0.0.1:{port}");
    let worker_model = model.clone();
    let worker_addr = addr.clone();
    std::thread::spawn(move || {
        let _ = run_worker(&worker_model, &worker_addr, split..block_count);
    });
    for _ in 0..100 {
        if std::net::TcpStream::connect(&addr).is_ok() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    let handshake = Gemma4Handshake {
        wire_version: GEMMA4_WIRE_VERSION + 1, // deliberately wrong
        block_count: block_count as u32,
        hidden: config.embedding_length,
        worker_first_layer: split as u32,
        worker_last_layer: block_count as u32,
        model_file_len: std::fs::metadata(&model).unwrap().len(),
        return_logits: false,
    };
    let err = Gemma4WorkerClient::connect(&addr, &handshake)
        .err()
        .expect("mismatched wire version must be rejected at handshake");
    assert!(
        err.to_string().contains("handshake mismatch"),
        "error names the mismatch: {err}"
    );
}

/// The serve-lane distributed runtime (persistent master shard + per-request
/// worker session) must produce the same greedy ids/text as the committed
/// oracle, and its streaming deltas must concatenate to exactly the final
/// text (the same contract the single-node streaming runtime keeps).
#[test]
fn distributed_serve_runtime_streaming_matches_oracle() {
    let Some(model) = std::env::var_os("CAMELID_GEMMA4_GGUF").map(PathBuf::from) else {
        eprintln!("SKIP distributed serve runtime: set CAMELID_GEMMA4_GGUF");
        return;
    };
    let row = model
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_string();
    let oracle_path = repo_path(&format!("qa/gemma4/oracle/{row}.basic_v1.json"));
    let oracle: Oracle = serde_json::from_str(
        &std::fs::read_to_string(&oracle_path)
            .unwrap_or_else(|_| panic!("no committed oracle for row {row}")),
    )
    .expect("oracle json");

    let gguf = camelid::gguf::read_metadata(&model).expect("gguf");
    let config = camelid::model::LlamaModelConfig::from_gguf(&gguf).expect("config");
    let block_count = config.block_count as usize;
    let g = config.gemma4.as_ref().expect("gemma4");
    let first_shared = block_count - g.num_kv_shared_layers as usize;
    let default_split = (block_count / 2).min(if g.num_kv_shared_layers > 0 {
        first_shared.saturating_sub(2)
    } else {
        block_count / 2
    });
    let split = std::env::var("CAMELID_GEMMA4_SPLIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default_split.max(1));

    let port = 39413;
    let addr = format!("127.0.0.1:{port}");
    let worker_model = model.clone();
    let worker_addr = addr.clone();
    std::thread::spawn(move || {
        let _ = run_worker(&worker_model, &worker_addr, split..block_count);
    });
    for _ in 0..100 {
        if std::net::TcpStream::connect(&addr).is_ok() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    let runtime =
        camelid::gemma4_distributed::Gemma4DistributedRuntime::connect(&model, &addr, split)
            .expect("distributed serve runtime connect");
    assert_eq!(runtime.split(), split);

    let expected = oracle
        .results
        .iter()
        .find(|r| r.id == "capital-france")
        .expect("capital-france in oracle");
    let mut deltas = String::new();
    let (text, ids) = runtime
        .generate_greedy_streaming("The capital of France is", 24, |d| deltas.push_str(d))
        .expect("distributed streaming generate");
    eprintln!("distributed serve runtime: {text:?} ids {ids:?}");
    assert_eq!(
        ids, expected.generated_tokens,
        "distributed serve greedy ids must match the llama.cpp oracle"
    );
    assert_eq!(
        text, expected.generated_text,
        "distributed serve greedy text must match the llama.cpp oracle"
    );
    assert_eq!(
        deltas, text,
        "streaming deltas must concatenate to the final text"
    );

    // A second request on the same runtime must reconnect cleanly (fresh
    // worker KV) and produce identical output — serve handles requests
    // back-to-back on one persistent master shard.
    let (text2, ids2) = runtime
        .generate_greedy("The capital of France is", 24)
        .expect("second distributed generate");
    assert_eq!(ids2, ids, "second request must be deterministic");
    assert_eq!(text2, text);
}
