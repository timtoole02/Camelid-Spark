//! Stage-by-stage load probe for a real Gemma 4 Q8_0 GGUF.
//!
//! This does NOT exercise the forward pass (which is not implemented for gemma4
//! yet). It only runs the cheap metadata/descriptor stages — architecture
//! recognition, config parsing, tokenizer construction, and tensor binding — to
//! pinpoint the next concrete gap. Skipped unless `CAMELID_GEMMA4_GGUF` points at
//! a gemma4 GGUF (the file is multi-GB and machine-local).
//!
//! Run: `CAMELID_GEMMA4_GGUF=/path/gemma-4-E4B-it-Q8_0.gguf \
//!       cargo test --test gemma4_load -- --nocapture`

use std::path::PathBuf;

use camelid::gguf::read_metadata;
use camelid::model::{Gemma4Binding, LlamaModelConfig, LlamaTensorBinding};
use camelid::tokenizer::Tokenizer;

#[test]
fn gemma4_load_stages() {
    let Some(path) = std::env::var_os("CAMELID_GEMMA4_GGUF").map(PathBuf::from) else {
        eprintln!("SKIP gemma4_load_stages: set CAMELID_GEMMA4_GGUF to a gemma4 Q8_0 GGUF");
        return;
    };

    eprintln!("== reading metadata: {} ==", path.display());
    let gguf = read_metadata(&path).expect("read_metadata should succeed on a valid GGUF");
    eprintln!("general.architecture = {:?}", gguf.architecture());

    eprintln!("== stage 1: LlamaModelConfig::from_gguf ==");
    let config = LlamaModelConfig::from_gguf(&gguf).expect("config should parse for gemma4");
    eprintln!(
        "  blocks={} emb={} heads={} kv_heads={} ffn={} ctx={} rms_eps={}",
        config.block_count,
        config.embedding_length,
        config.attention_head_count,
        config.attention_head_count_kv,
        config.feed_forward_length,
        config.context_length,
        config.rms_norm_epsilon,
    );
    let gemma4 = config
        .gemma4
        .as_ref()
        .expect("gemma4 metadata must be populated for the gemma4 architecture");
    eprintln!("  gemma4 = {gemma4:#?}");
    // Sanity-check the parsed values against the known E4B shape.
    assert!(gemma4.head_dim_sliding > 0 && gemma4.head_dim_global > 0);
    assert_eq!(gemma4.layer_is_sliding.len(), config.block_count as usize);
    assert_eq!(
        gemma4.layer_is_sliding.last(),
        Some(&false),
        "final layer must be full attention"
    );

    eprintln!("== stage 2: Tokenizer::from_gguf + parity vs llama.cpp ==");
    match Tokenizer::from_gguf(&gguf) {
        Ok(tok) => {
            eprintln!("  tokenizer: OK");
            // llama.cpp tokenized "The capital of France is" (add_special) to:
            const ORACLE: &[u32] = &[2, 818, 5279, 529, 7001, 563];
            match tok.encode("The capital of France is", true, false) {
                Ok(ids) => {
                    eprintln!("  encode -> {ids:?}");
                    eprintln!("  oracle -> {ORACLE:?}");
                    eprintln!(
                        "  tokenizer parity: {}",
                        if ids == ORACLE {
                            "MATCH ✅"
                        } else {
                            "MISMATCH ❌"
                        }
                    );
                }
                Err(e) => eprintln!("  encode ERR -> {e}"),
            }
        }
        Err(e) => eprintln!("  tokenizer: ERR -> {e}"),
    }

    eprintln!("== stage 3: LlamaTensorBinding::bind ==");
    match LlamaTensorBinding::bind(&gguf, &config) {
        Ok(binding) => eprintln!(
            "  bind: OK ({} layers, tied_output={})",
            binding.layers.len(),
            binding.output_is_tied_embedding
        ),
        Err(e) => eprintln!("  bind: ERR -> {e}"),
    }
    eprintln!("== stage 4: Gemma4Binding::bind (full gemma4 weight set) ==");
    match Gemma4Binding::bind(&gguf, &config) {
        Ok(binding) => eprintln!(
            "  gemma4 bind: OK ({} layers, tied_output={}, per_layer_embeddings={}, has_post_norm={})",
            binding.layers.len(),
            binding.output_is_tied_embedding,
            binding.has_per_layer_embeddings(),
            binding.layers.first().map(|l| l.post_norm.is_some()).unwrap_or(false),
        ),
        Err(e) => eprintln!("  gemma4 bind: ERR -> {e}"),
    }
    eprintln!("== done ==");
}
