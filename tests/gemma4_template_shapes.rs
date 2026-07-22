//! Gemma 4 chat-template shape parity against the committed reference pack
//! (`qa/gemma4/template_shapes_v1.json`, captured from llama.cpp 5d56eff
//! `--jinja` `/apply-template` on the GGUF-embedded template).
//!
//! Two layers:
//! 1. STRING parity (always runs): Camelid's marker renderer must reproduce
//!    the reference `rendered_prompt` byte-for-byte for every shape in the
//!    supported envelope (leading system + alternating user/assistant turns,
//!    both enable_thinking modes).
//! 2. TOKEN parity (env-gated on `CAMELID_GEMMA4_GGUF`): encoding the rendered
//!    prompt (BOS + parse_special) must match the reference's pinned token ids.

use std::path::PathBuf;

#[derive(serde::Deserialize)]
struct Pack {
    shapes: Vec<Shape>,
}

#[derive(serde::Deserialize)]
struct Shape {
    id: String,
    thinking: bool,
    messages: Vec<Message>,
    rendered_prompt: String,
    prompt_tokens_with_bos: Vec<u32>,
}

#[derive(serde::Deserialize)]
struct Message {
    role: String,
    content: String,
}

fn pack() -> Pack {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("qa/gemma4/template_shapes_v1.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("template shapes pack"))
        .expect("pack json")
}

#[test]
fn renderer_matches_reference_strings_for_all_shapes() {
    let pack = pack();
    assert!(pack.shapes.len() >= 8, "pack must cover both modes");
    for shape in &pack.shapes {
        let messages: Vec<camelid::api::ChatMessage> = shape
            .messages
            .iter()
            .map(|m| camelid::api::ChatMessage {
                role: m.role.clone(),
                content: m.content.clone(),
                unsupported_content_parts: Vec::new(),
            })
            .collect();
        let rendered = camelid::api::gemma4_chat_prompt_for_tests(&messages, shape.thinking);
        assert_eq!(
            rendered, shape.rendered_prompt,
            "shape {} diverges from the reference rendering",
            shape.id
        );
    }
}

#[test]
fn rendered_prompts_tokenize_to_reference_ids() {
    let Some(model) = std::env::var_os("CAMELID_GEMMA4_GGUF").map(PathBuf::from) else {
        eprintln!("SKIP template token parity: set CAMELID_GEMMA4_GGUF");
        return;
    };
    let gguf = camelid::gguf::read_metadata(&model).expect("gguf");
    let tokenizer = camelid::tokenizer::Tokenizer::from_gguf(&gguf).expect("tokenizer");
    for shape in &pack().shapes {
        let tokens = tokenizer
            .encode(&shape.rendered_prompt, true, true)
            .expect("encode");
        assert_eq!(
            tokens, shape.prompt_tokens_with_bos,
            "shape {} tokenization diverges",
            shape.id
        );
    }
}
