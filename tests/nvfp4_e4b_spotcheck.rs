//! BASALT Phase 2 / Gate G2: cross-engine dequant spot-check on the PRODUCED pilot row.
//!
//! `tests/fixtures/dequant/nvfp4_e4b_spotcheck.json` holds 5,120 blocks sampled from
//! 10 NVFP4 tensors (all 7 matmul families, depths blk.0–blk.41) of
//! `gemma-4-E4B-it-NVFP4-mm.gguf` — the Amendment 2 pilot artifact, wire bytes read
//! verbatim from the produced GGUF, expected values from `dequantize_row_nvfp4` in the
//! PINNED llama.cpp (acd79d603) `ggml-base.dll` (route: linked-libs, same provenance as
//! the Phase 1 fixtures). This proves Camelid's decode agrees bit-for-bit with the pin
//! on the real campaign artifact, not just on synthetic and qwen3-sized inputs.
//! Comparisons are on `f32::to_bits()` — never float equality.

use std::path::Path;

use camelid::inference::nvfp4_wire_block_dequant;
use camelid::tensor::{decode_nvfp4_tensor, NVFP4_VALUES_PER_BLOCK, NVFP4_WIRE_BYTES_PER_BLOCK};
use serde::Deserialize;
use serde_json::Value;

const PIN_SHA: &str = "acd79d603";
/// sha256 of the produced pilot row this fixture was sampled from (Phase 2 quantize
/// receipts + the independent spot-check extraction both recorded this value).
const PILOT_ROW_SHA256: &str = "eb293344972e2b292a043b8e7649b9788dca915b034e5c2721cfc531cf9863d9";

#[derive(Deserialize)]
struct SpotBlock {
    /// index into the fixture's `tensors` array
    t: usize,
    /// block index within that tensor (recorded for reproducibility; not re-derived here)
    #[allow(dead_code)]
    b: u64,
    /// wire bytes, base64 (36 B)
    w: String,
    /// expected dequant, concatenated %08x bit patterns (64 values)
    e: String,
}

fn fixture() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("dequant")
        .join("nvfp4_e4b_spotcheck.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing fixture {}: {e}", path.display()));
    let v: Value = serde_json::from_str(&raw).expect("spotcheck fixture parses");
    assert_eq!(
        v["provenance"]["pin_sha"].as_str(),
        Some(PIN_SHA),
        "fixture provenance pin mismatch"
    );
    assert_eq!(
        v["model"]["sha256"].as_str(),
        Some(PILOT_ROW_SHA256),
        "fixture must be sampled from the receipted pilot row"
    );
    v
}

fn hex_u32(h: &str) -> u32 {
    u32::from_str_radix(h, 16).unwrap_or_else(|e| panic!("bad hex u32 {h:?}: {e}"))
}

fn b64_decode(s: &str) -> Vec<u8> {
    // Minimal RFC 4648 decoder (mirrors tests/nvfp4_format.rs; no new deps).
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let rev = {
        let mut r = [255u8; 256];
        for (i, &c) in TABLE.iter().enumerate() {
            r[c as usize] = i as u8;
        }
        r
    };
    let bytes: Vec<u8> = s.bytes().filter(|&b| b != b'=').collect();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    for chunk in bytes.chunks(4) {
        let mut acc = 0u32;
        for &c in chunk {
            let v = rev[c as usize];
            assert_ne!(v, 255, "invalid base64 byte {c}");
            acc = (acc << 6) | u32::from(v);
        }
        let bits = chunk.len() * 6;
        acc <<= 24 - bits.min(24);
        let full = [(acc >> 16) as u8, (acc >> 8) as u8, acc as u8];
        out.extend_from_slice(&full[..(bits / 8).min(3)]);
    }
    out
}

#[test]
fn produced_pilot_row_blocks_bit_exact_through_both_paths() {
    let fx = fixture();
    let blocks: Vec<SpotBlock> =
        serde_json::from_value(fx["blocks"].clone()).expect("blocks parse");
    assert_eq!(blocks.len(), 5120, "5120 sampled blocks (512 x 10 tensors)");
    let tensor_names: Vec<String> = fx["tensors"]
        .as_array()
        .expect("tensors")
        .iter()
        .map(|t| t["name"].as_str().expect("tensor name").to_string())
        .collect();
    assert_eq!(tensor_names.len(), 10);
    // Every matmul family must be represented — the sampling contract of the receipt.
    for family in [
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "ffn_up",
        "ffn_gate",
        "ffn_down",
    ] {
        assert!(
            tensor_names.iter().any(|n| n.contains(family)),
            "family {family} missing from sampled tensors"
        );
    }

    for (t, name) in tensor_names.iter().enumerate() {
        let group: Vec<&SpotBlock> = blocks.iter().filter(|b| b.t == t).collect();
        assert_eq!(group.len(), 512, "tensor {name}: 512 sampled blocks");

        // Path 1: the pin-bitwise hot-path twin, block by block.
        // Path 2: the D17/T5 fail-closed load path, all sampled blocks as one tensor.
        let mut concat_wire = Vec::with_capacity(group.len() * NVFP4_WIRE_BYTES_PER_BLOCK);
        let mut concat_want: Vec<u32> = Vec::with_capacity(group.len() * NVFP4_VALUES_PER_BLOCK);
        for blk in &group {
            let wire = b64_decode(&blk.w);
            assert_eq!(wire.len(), NVFP4_WIRE_BYTES_PER_BLOCK, "{name}: wire len");
            assert_eq!(
                blk.e.len(),
                NVFP4_VALUES_PER_BLOCK * 8,
                "{name}: expected len"
            );
            let want: Vec<u32> = (0..blk.e.len())
                .step_by(8)
                .map(|i| hex_u32(&blk.e[i..i + 8]))
                .collect();

            let got = nvfp4_wire_block_dequant(&wire);
            for (j, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    *w,
                    "{name}: block-path element {j} got {:#010x} want {w:#010x}",
                    g.to_bits()
                );
            }
            concat_wire.extend_from_slice(&wire);
            concat_want.extend_from_slice(&want);
        }

        let decoded = decode_nvfp4_tensor(name, &concat_wire, concat_want.len())
            .unwrap_or_else(|e| panic!("{name}: tensor path must decode: {e}"));
        assert_eq!(decoded.len(), concat_want.len());
        for (j, (g, w)) in decoded.iter().zip(concat_want.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                *w,
                "{name}: tensor-path element {j} got {:#010x} want {w:#010x}",
                g.to_bits()
            );
        }
    }
}
