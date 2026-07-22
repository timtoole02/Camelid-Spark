//! Pillar Two — the execution-trace rollup survives the receipt emit→verify round trip.
//!
//! `replay_receipt_request` is the exact API generation path that BOTH receipt emission and
//! `verify-receipt` re-run funnel through, so exercising it proves emission and verification
//! cannot desync: a deterministic replay captures a rollup digest, and a second replay
//! re-derives the identical digest (the verification guarantee). Off the deterministic lane the
//! rollup is absent (fail closed).
//!
//! Env-gated on the real GGUF (self-skips without it):
//! ```text
//! CAMELID_TINYLLAMA_Q8_GGUF=/path/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
//!   cargo test --release --test execution_trace_receipt -- --nocapture
//! ```

use std::path::PathBuf;

use camelid::receipt::{host_isa_marker, ExecutionTraceBlock, ReceiptRequest};
use serde_json::json;

const METAL_KEYS: &[&str] = &[
    "CAMELID_METAL_RESIDENT_DECODE",
    "CAMELID_METAL_F32Y",
    "CAMELID_METAL_WIRE",
    "CAMELID_METAL_WIRE_NSG8",
    "CAMELID_METAL_ATTN2",
    "CAMELID_METAL_RESIDENT_PREFILL",
    "CAMELID_METAL_MM",
    "CAMELID_METAL_LINEAR",
    "CAMELID_METAL_Q8",
    "CAMELID_METAL_NOCOPY",
];

fn deterministic_lane_on() {
    std::env::set_var("CAMELID_DETERMINISTIC", "1");
    for key in METAL_KEYS {
        std::env::set_var(key, "0");
    }
    std::env::set_var("CAMELID_NO_GPU_SAMPLE", "1");
}

fn request() -> ReceiptRequest {
    ReceiptRequest {
        endpoint: "/v1/completions".to_string(),
        messages_or_prompt: json!("hello"),
        max_tokens: 6,
        temperature: 0.0,
        top_p: None,
        top_k: None,
        seed: None,
        stop: Vec::new(),
        response_format: None,
    }
}

#[tokio::test]
async fn execution_trace_round_trips_through_replay() {
    let Some(model) = std::env::var_os("CAMELID_TINYLLAMA_Q8_GGUF").map(PathBuf::from) else {
        eprintln!("SKIP execution_trace_receipt: set CAMELID_TINYLLAMA_Q8_GGUF");
        return;
    };

    // Emission lane: a deterministic replay captures a rollup digest.
    deterministic_lane_on();
    let replay1 = camelid::api::replay_receipt_request(&model, None, &request())
        .await
        .expect("deterministic replay");
    let digest = replay1
        .execution_trace_digest
        .clone()
        .expect("rollup digest present on the deterministic lane");
    eprintln!("execution_trace_receipt: digest = {digest}");
    assert_eq!(
        digest.len(),
        64,
        "rollup digest must be 64 lowercase-hex chars"
    );

    // Verification guarantee: an independent re-run re-derives the identical digest.
    let replay2 = camelid::api::replay_receipt_request(&model, None, &request())
        .await
        .expect("deterministic replay (re-derive)");
    assert_eq!(
        Some(&digest),
        replay2.execution_trace_digest.as_ref(),
        "rollup digest must re-derive identically (this is what verify-receipt checks)"
    );

    // The receipt block records the lane + this host's ISA so a verifier knows when it can
    // re-derive the digest.
    let block = ExecutionTraceBlock::rollup_v1(digest.clone(), 1, 6);
    assert_eq!(block.schema, "camelid.execution-trace/v1");
    assert_eq!(block.host_isa, host_isa_marker());
    assert_eq!(block.digest, digest);

    // Fail closed: off the deterministic lane no rollup is captured.
    std::env::remove_var("CAMELID_DETERMINISTIC");
    let replay_plain = camelid::api::replay_receipt_request(&model, None, &request())
        .await
        .expect("non-deterministic replay");
    assert!(
        replay_plain.execution_trace_digest.is_none(),
        "no rollup may be produced off the deterministic lane"
    );
}
