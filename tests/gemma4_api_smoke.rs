//! Gemma 4 API smoke — fail-closed behavior runs with no model file; the live
//! generation smoke is gated on `CAMELID_GEMMA4_GGUF` + `CAMELID_GEMMA4_SERVE`.

use axum::{
    body::{to_bytes, Body},
    http::{header, Request, StatusCode},
};
use serde_json::{json, Value};
use tower::ServiceExt;

async fn post_chat(body: Value) -> (StatusCode, Value) {
    let app = camelid::api::router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
            .unwrap_or(Value::Null);
    (status, body)
}

#[tokio::test]
async fn multimodal_image_part_fails_closed_with_typed_error() {
    let (status, body) = post_chat(json!({
        "model": "gemma4_e2b_it_q8_0",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this picture?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}
            ]
        }]
    }))
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "unsupported_multimodal_content");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("image_url"), "names the part: {message}");
    assert!(
        message.contains("text-token"),
        "states the text-only boundary: {message}"
    );
}

#[tokio::test]
async fn multimodal_audio_and_video_parts_fail_closed() {
    let (status, body) = post_chat(json!({
        "model": "gemma4_e4b_it_q8_0",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": "...", "format": "wav"}},
                {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}}
            ]
        }]
    }))
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "unsupported_multimodal_content");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("input_audio") && message.contains("video_url"));
}

#[tokio::test]
async fn text_only_content_parts_array_is_accepted_as_text() {
    // The OpenAI parts form with only text parts must NOT be rejected — it
    // concatenates and proceeds (here failing later for "no model loaded",
    // which proves it got past the multimodal guard).
    let (status, body) = post_chat(json!({
        "model": "gemma4_e2b_it_q8_0",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"}
            ]
        }]
    }))
    .await;
    assert_ne!(status, StatusCode::BAD_REQUEST, "body: {body}");
}

#[tokio::test]
async fn gemma4_request_without_runtime_fails_clearly() {
    // No model loaded: targeting a gemma4 row must produce a clear failure
    // (404/422/503 class), never a silent fallback that generates garbage.
    let (status, body) = post_chat(json!({
        "model": "gemma4_e2b_it_q8_0",
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .await;
    assert!(
        status.is_client_error() || status.is_server_error(),
        "must fail clearly, got {status}: {body}"
    );
}

// --- live serve smoke (gated) ------------------------------------------------

/// Full live smoke: load the exact row, serve non-streaming + streaming chat,
/// and check capability/health state. Requires the model file and the serve
/// flag, and runs the real runtime (slow).
///
/// Run: `CAMELID_GEMMA4_GGUF=/path/row.gguf CAMELID_GEMMA4_SERVE=1 \
///       cargo test --release --test gemma4_api_smoke -- --nocapture live_`
#[tokio::test]
async fn live_gemma4_chat_serve_smoke() {
    let Some(model_path) = std::env::var_os("CAMELID_GEMMA4_GGUF") else {
        eprintln!("SKIP live gemma4 serve smoke: set CAMELID_GEMMA4_GGUF");
        return;
    };
    if std::env::var("CAMELID_GEMMA4_SERVE").map(|v| v == "1") != Ok(true) {
        eprintln!("SKIP live gemma4 serve smoke: set CAMELID_GEMMA4_SERVE=1");
        return;
    }
    let model_path = std::path::PathBuf::from(model_path);

    // Load the runtime directly and serve it through the router state, exactly
    // as the serve path would.
    let runtime = tokio::task::spawn_blocking({
        let p = model_path.clone();
        move || camelid::gemma4_runtime::Gemma4Runtime::load(&p)
    })
    .await
    .unwrap()
    .expect("gemma4 runtime load");

    let state = camelid::api::AppState::default();
    let row_id = match model_path.file_name().and_then(|n| n.to_str()) {
        Some(name) if name.contains("E2B") => "gemma4_e2b_it_q8_0",
        Some(name) if name.contains("E4B") => "gemma4_e4b_it_q8_0",
        other => panic!("no catalog row for {other:?}"),
    };
    state.insert_gemma4_runtime_for_tests(row_id, runtime).await;
    let app = camelid::api::router_with_state(state);

    // Non-streaming chat.
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": row_id,
                        "messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
                        "max_tokens": 16
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();
    eprintln!("non-streaming content: {content:?}");
    assert!(
        content.contains("Paris"),
        "expected Paris in {content:?} (body {body})"
    );
    // Turn markers and thinking channels must never leak into chat content.
    for marker in [
        "<turn|>",
        "<|turn>",
        "<|channel>",
        "<channel|>",
        "<end_of_turn>",
    ] {
        assert!(
            !content.contains(marker),
            "marker {marker} leaked into chat content: {content:?}"
        );
    }

    // Streaming chat: SSE chunks ending with [DONE].
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": row_id,
                        "messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
                        "max_tokens": 16,
                        "stream": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let raw = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8_lossy(&raw);
    assert!(text.contains("chat.completion.chunk"), "sse shape: {text}");
    assert!(text.trim_end().ends_with("[DONE]"), "sse done: {text}");
    let mut streamed = String::new();
    for line in text.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                continue;
            }
            if let Ok(chunk) = serde_json::from_str::<Value>(data) {
                if let Some(delta) = chunk["choices"][0]["delta"]["content"].as_str() {
                    streamed.push_str(delta);
                }
            }
        }
    }
    eprintln!("streamed content: {streamed:?}");
    assert!(streamed.contains("Paris"), "streamed: {streamed:?}");
    for marker in [
        "<turn|>",
        "<|turn>",
        "<|channel>",
        "<channel|>",
        "<end_of_turn>",
    ] {
        assert!(
            !streamed.contains(marker),
            "marker {marker} leaked into streamed content: {streamed:?}"
        );
    }
}

/// Live distributed serve smoke: the chat route served by the DISTRIBUTED
/// lane (loopback worker over real TCP) must answer exactly like the local
/// lane — same row, same prompt, marker/channel leakage rules unchanged.
///
/// Run: `CAMELID_GEMMA4_GGUF=/path/row.gguf CAMELID_GEMMA4_SERVE=1 \
///       cargo test --release --test gemma4_api_smoke -- --nocapture distributed`
#[tokio::test]
async fn live_gemma4_distributed_chat_serve_smoke() {
    let Some(model_path) = std::env::var_os("CAMELID_GEMMA4_GGUF") else {
        eprintln!("SKIP live gemma4 distributed serve smoke: set CAMELID_GEMMA4_GGUF");
        return;
    };
    if std::env::var("CAMELID_GEMMA4_SERVE").map(|v| v == "1") != Ok(true) {
        eprintln!("SKIP live gemma4 distributed serve smoke: set CAMELID_GEMMA4_SERVE=1");
        return;
    }
    let model_path = std::path::PathBuf::from(model_path);

    // Split below the shared-KV source layers, same rule as the parity tests.
    let gguf = camelid::gguf::read_metadata(&model_path).expect("gguf");
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

    let addr = "127.0.0.1:39414".to_string();
    let worker_model = model_path.clone();
    let worker_addr = addr.clone();
    std::thread::spawn(move || {
        let _ = camelid::gemma4_distributed::run_worker(
            &worker_model,
            &worker_addr,
            split..block_count,
        );
    });
    for _ in 0..100 {
        if std::net::TcpStream::connect(&addr).is_ok() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    let runtime = tokio::task::spawn_blocking({
        let p = model_path.clone();
        let a = addr.clone();
        move || camelid::gemma4_distributed::Gemma4DistributedRuntime::connect(&p, &a, split)
    })
    .await
    .unwrap()
    .expect("gemma4 distributed runtime connect");

    let state = camelid::api::AppState::default();
    let row_id = match model_path.file_name().and_then(|n| n.to_str()) {
        Some(name) if name.contains("E2B") => "gemma4_e2b_it_q8_0",
        Some(name) if name.contains("E4B") => "gemma4_e4b_it_q8_0",
        Some(name) if name.contains("12b") || name.contains("12B") => "gemma4_12b_it_q8_0",
        Some(name) if name.contains("26B") || name.contains("26b") => "gemma4_26b_a4b_it_q4_0",
        other => panic!("no catalog row for {other:?}"),
    };
    state
        .insert_gemma4_distributed_runtime_for_tests(row_id, runtime)
        .await;
    let app = camelid::api::router_with_state(state);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": row_id,
                        "messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
                        "max_tokens": 16
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();
    eprintln!("distributed non-streaming content: {content:?}");
    assert!(
        content.contains("Paris"),
        "expected Paris in {content:?} (body {body})"
    );
    for marker in ["<turn|>", "<|turn>", "<|channel>", "<channel|>"] {
        assert!(
            !content.contains(marker),
            "marker {marker} leaked into distributed chat content: {content:?}"
        );
    }
}
