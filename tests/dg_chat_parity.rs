//! DiffusionGemma lane Phase 6 gate: the end-to-end chat wrapper (`DgChat`)
//! that backs the `diffusion-gemma-chat` subcommand. The bidirectional
//! denoise generation itself (`mc_generate`) is proven bit-exact by the
//! Phase 5 gate (`dg_mc_loop_parity`); this gate proves the SURROUNDING chat
//! plumbing matches the pinned reference's `diffusion-cli.cpp` chat path:
//!
//!   1. render + tokenize: `DgChat::render_prompt(M)` (the model's own chat
//!      template via minijinja with add_generation_prompt=true, then encode
//!      with add_special=true / parse_special=true) == the reference's
//!      `apply_template` + `common_tokenize(add_special, parse_special)`.
//!   2. detokenize: `DgChat::decode_response(ids)` == the reference's
//!      `common_detokenize(ids, special=false)`.
//!
//! Composition: M --render_prompt(1)--> prompt ids --mc_generate(Phase 5,
//! bit-exact)--> response ids --decode_response(2)--> reply. With (1) and (2)
//! exact and the middle Phase-5-proven, the full `DgChat::generate(M)` reply
//! equals the reference chat reply by construction.
//!
//! Env-gated (skips unless all are set):
//!   CAMELID_DG_GGUF            the tracked DiffusionGemma GGUF
//!   CAMELID_DG_CHAT_MSG        the user message M
//!   CAMELID_DG_CHAT_PROMPT_IDS reference render+tokenize of M (i32 LE)
//!   CAMELID_DG_CHAT_RESP_IDS   the Phase 5 oracle response ids (i32 LE)
//!   CAMELID_DG_CHAT_RESP_TXT   reference common_detokenize of those ids (utf8)

use std::path::PathBuf;

use camelid::diffusion_gemma::chat::DgChat;

fn read_i32(path: &str) -> Vec<i32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("read {path}: {e}"))
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.is_empty())
}

#[test]
fn dg_chat_wrapper_matches_pinned_llamacpp() {
    let (Some(gguf), Some(msg), Some(prompt_ids_p), Some(resp_ids_p), Some(resp_txt_p)) = (
        env("CAMELID_DG_GGUF"),
        env("CAMELID_DG_CHAT_MSG"),
        env("CAMELID_DG_CHAT_PROMPT_IDS"),
        env("CAMELID_DG_CHAT_RESP_IDS"),
        env("CAMELID_DG_CHAT_RESP_TXT"),
    ) else {
        eprintln!("dg_chat_parity: skipped (set CAMELID_DG_GGUF / _CHAT_MSG / _CHAT_PROMPT_IDS / _CHAT_RESP_IDS / _CHAT_RESP_TXT)");
        return;
    };

    let chat = DgChat::load(&PathBuf::from(&gguf)).expect("DgChat::load");

    // (1) render + tokenize parity.
    let ours: Vec<i32> = chat
        .render_prompt(&msg)
        .expect("render_prompt")
        .into_iter()
        .map(|t| t as i32)
        .collect();
    let reference = read_i32(&prompt_ids_p);
    assert_eq!(
        ours.len(),
        reference.len(),
        "prompt token COUNT differs: ours={} reference={}",
        ours.len(),
        reference.len()
    );
    if ours != reference {
        let first = ours
            .iter()
            .zip(&reference)
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        panic!(
            "render+tokenize parity FAILED at token {first}: ours={} reference={}\nours={:?}\nref ={:?}",
            ours[first], reference[first], ours, reference
        );
    }
    eprintln!(
        "render+tokenize parity: PASS ({} prompt tokens)",
        ours.len()
    );

    // (2) detokenize parity.
    let resp_ids = read_i32(&resp_ids_p);
    let ours_text = chat.decode_response(&resp_ids).expect("decode_response");
    let reference_text =
        std::fs::read_to_string(&resp_txt_p).unwrap_or_else(|e| panic!("read {resp_txt_p}: {e}"));
    // The reference text file may carry a single trailing newline from the
    // dumper's `printf("%s\n", ...)`; compare on the exact detok payload.
    let reference_text = reference_text.strip_suffix('\n').unwrap_or(&reference_text);
    assert_eq!(
        ours_text,
        reference_text,
        "detokenize parity FAILED:\nours len={} reference len={}",
        ours_text.len(),
        reference_text.len()
    );
    eprintln!(
        "detokenize parity: PASS ({} response tokens -> {} chars)",
        resp_ids.len(),
        ours_text.len()
    );

    eprintln!("DG CHAT WRAPPER PARITY: PASS (render+tokenize + detokenize bit-exact vs the pinned reference; generation is Phase-5-proven)");
}
