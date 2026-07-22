//! DiffusionGemma lane Phase 1 gate: tokenizer parity against the pinned
//! llama.cpp reference for the exact tracked GGUF.
//!
//! Env-gated: skips unless `CAMELID_DG_GGUF` (the tracked model) and
//! `CAMELID_DG_TOK_REF` (the reference dump produced by
//! `scripts/dg-tokenize-dump.cpp` at the pin, see scripts/dg-tokenizer-parity.sh)
//! are set. The reference dump carries, per case: the input text (for chat
//! cases, the prompt as rendered by the model's own chat template via
//! llama.cpp/minja), llama.cpp's token ids, per-token pieces, and detokenized
//! string — pieces/strings base64-coded because byte-fallback tokens can split
//! UTF-8 sequences.
//!
//! Gate criteria (all must hold for every case):
//!  - encode: camelid token ids == llama.cpp token ids (100% match)
//!  - decode: camelid decode(ids) bytes == llama.cpp per-token pieces
//!    concatenated == llama.cpp detokenize output
//!
//! Raw cases tokenize with add_special=true / parse_special=false; chat cases
//! with parse_special=true, mirroring the reference diffusion CLI
//! (examples/diffusion/diffusion-cli.cpp run_turn at the pin).

use std::io::Write;
use std::path::Path;

use camelid::gguf::read_metadata;
use camelid::tokenizer::Tokenizer;

fn b64_decode(s: &str) -> Vec<u8> {
    const TBL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let inv = {
        let mut inv = [255u8; 256];
        for (i, &c) in TBL.iter().enumerate() {
            inv[c as usize] = i as u8;
        }
        inv
    };
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let mut acc = 0u32;
    let mut bits = 0u32;
    for &c in s.as_bytes() {
        if c == b'=' {
            break;
        }
        let v = inv[c as usize];
        assert_ne!(v, 255, "invalid base64 byte {c}");
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    out
}

#[derive(Debug)]
struct RefCase {
    id: String,
    mode: String,
    text: Vec<u8>,
    tokens: Vec<u32>,
    pieces: Vec<Vec<u8>>,
    detok: Vec<u8>,
}

/// Field extraction for the harness's machine-generated JSON (string values
/// are base64 — no escapes; numbers are non-negative integers).
fn parse_cases(text: &str) -> Vec<RefCase> {
    let cases_at = text.find("\"cases\"").expect("cases array");
    let mut out = Vec::new();
    let mut rest = &text[cases_at..];
    while let Some(start) = rest.find("\"id\"") {
        rest = &rest[start..];
        let take_str = |rest: &str, key: &str| -> String {
            let pat = format!("\"{key}\": \"");
            let s = rest.find(&pat).expect(key) + pat.len();
            let e = rest[s..].find('"').expect("close quote") + s;
            rest[s..e].to_string()
        };
        let take_u32_array = |rest: &str, key: &str| -> Vec<u32> {
            let pat = format!("\"{key}\": [");
            let s = rest.find(&pat).expect(key) + pat.len();
            let e = rest[s..].find(']').expect("close bracket") + s;
            rest[s..e]
                .split(',')
                .map(str::trim)
                .filter(|t| !t.is_empty())
                .map(|t| t.parse().expect("token id"))
                .collect()
        };
        let take_str_array = |rest: &str, key: &str| -> Vec<String> {
            let pat = format!("\"{key}\": [");
            let s = rest.find(&pat).expect(key) + pat.len();
            let e = rest[s..].find(']').expect("close bracket") + s;
            rest[s..e]
                .split(',')
                .map(|t| t.trim().trim_matches('"').to_string())
                .filter(|t| !t.is_empty())
                .collect()
        };
        out.push(RefCase {
            id: take_str(rest, "id"),
            mode: take_str(rest, "mode"),
            text: b64_decode(&take_str(rest, "text_b64")),
            tokens: take_u32_array(rest, "tokens"),
            pieces: take_str_array(rest, "pieces_b64")
                .iter()
                .map(|p| b64_decode(p))
                .collect(),
            detok: b64_decode(&take_str(rest, "detok_b64")),
        });
        rest = &rest[4..];
    }
    out
}

#[test]
fn dg_tokenizer_matches_pinned_llamacpp() {
    let (Ok(gguf_path), Ok(ref_path)) = (
        std::env::var("CAMELID_DG_GGUF"),
        std::env::var("CAMELID_DG_TOK_REF"),
    ) else {
        eprintln!(
            "skipping: CAMELID_DG_GGUF / CAMELID_DG_TOK_REF not set \
             (run scripts/dg-tokenizer-parity.sh)"
        );
        return;
    };

    let ref_text = std::fs::read_to_string(&ref_path).expect("read reference dump");
    let cases = parse_cases(&ref_text);
    assert!(!cases.is_empty(), "reference dump has no cases");

    let gguf = read_metadata(Path::new(&gguf_path)).expect("read tracked GGUF metadata");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("bind tokenizer from GGUF metadata");

    let mut rows = Vec::new();
    let mut failures = Vec::new();

    for case in &cases {
        let text = std::str::from_utf8(&case.text).expect("case text is valid UTF-8");
        let parse_special = case.mode == "chat";
        let ids = tokenizer
            .encode(text, /*add_special=*/ true, parse_special)
            .unwrap_or_else(|e| panic!("{}: encode failed: {e}", case.id));

        let ids_match = ids == case.tokens;
        let first_diff = ids
            .iter()
            .zip(case.tokens.iter())
            .position(|(a, b)| a != b)
            .or_else(|| {
                if ids.len() != case.tokens.len() {
                    Some(ids.len().min(case.tokens.len()))
                } else {
                    None
                }
            });

        let decoded = tokenizer
            .decode(&case.tokens, /*remove_special=*/ false)
            .unwrap_or_else(|e| panic!("{}: decode failed: {e}", case.id));
        let pieces_concat: Vec<u8> = case.pieces.concat();
        let decode_matches_pieces = decoded.as_bytes() == &pieces_concat[..];
        let decode_matches_detok = decoded.as_bytes() == &case.detok[..];

        if !(ids_match && decode_matches_pieces && decode_matches_detok) {
            let diff_at = first_diff.unwrap_or(usize::MAX);
            failures.push(format!(
                "{} ({}): ids_match={ids_match} first_diff_index={:?} (camelid {:?} vs llama.cpp {:?}) decode_vs_pieces={decode_matches_pieces} decode_vs_detok={decode_matches_detok}",
                case.id,
                case.mode,
                first_diff,
                first_diff.map(|i| ids.get(i)),
                first_diff.map(|i| case.tokens.get(i)),
            ));
            let _ = diff_at;
        }
        rows.push(format!(
            "    {{\"id\": \"{}\", \"mode\": \"{}\", \"n_tokens\": {}, \"token_ids_match\": {}, \"first_diff_index\": {}, \"decode_matches_pieces\": {}, \"decode_matches_detok\": {}}}",
            case.id,
            case.mode,
            case.tokens.len(),
            ids_match,
            first_diff.map_or("-1".to_string(), |i| i.to_string()),
            decode_matches_pieces,
            decode_matches_detok,
        ));
    }

    let report = format!(
        "{{\n  \"comparison\": \"camelid Tokenizer::from_gguf encode/decode vs pinned llama.cpp (scripts/dg-tokenize-dump.cpp: common_tokenize + common_token_to_piece + common_detokenize; chat prompts rendered by the model's chat template via llama.cpp minja)\",\n  \"llamacpp_pinned_commit\": \"{}\",\n  \"gguf\": \"{}\",\n  \"gate\": \"100% token-id match, decode == pieces == detokenize, every case\",\n  \"cases\": [\n{}\n  ],\n  \"pass\": {}\n}}\n",
        std::env::var("CAMELID_DG_PIN_SHA").unwrap_or_else(|_| "UNRECORDED".to_string()),
        Path::new(&gguf_path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        rows.join(",\n"),
        failures.is_empty(),
    );

    if let Ok(out_path) = std::env::var("CAMELID_DG_TOK_OUT") {
        let mut f = std::fs::File::create(&out_path).expect("create gate report");
        f.write_all(report.as_bytes()).expect("write gate report");
        eprintln!("gate report written to {out_path}");
    }
    eprintln!("{report}");

    assert!(
        failures.is_empty(),
        "tokenizer parity failures:\n{}",
        failures.join("\n")
    );
}
