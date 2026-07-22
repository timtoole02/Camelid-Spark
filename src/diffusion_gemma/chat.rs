//! Phase 6: a usable end-to-end chat path over the bit-exact multi-canvas
//! generation loop (`DgEncoderRuntime::mc_generate`). One turn = render the
//! model's own chat template, tokenize, run the block-autoregressive denoise
//! loop, detokenize. This is the first SUPPORTED-integration surface for
//! DiffusionGemma; it is opt-in (CLI subcommand / env-gated serve), and the
//! public posture is unchanged until the integration itself is validated.
//!
//! Correctness note: the generation math is the Phase 2-5 bit-exact CPU path.
//! This module only adds the surrounding chat plumbing (template render, EOG
//! set, tokenize/detokenize), reusing the same EOG construction the Phase 5
//! gate proved against the reference vocab.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use minijinja::{context, Environment};

use crate::gguf::{read_metadata, GgufFile};
use crate::tokenizer::Tokenizer;
use crate::{BackendError, Result};

use super::{DgEbParams, DgEncoderRuntime};

/// The vocab's end-of-generation id set, mirrored from `llama-vocab.cpp` at
/// the pin: the GGUF special eos/eot/eom + fim pad/rep/sep ids, plus the
/// text-matched EOG list, with the gemma4/paddleocr workaround that removes
/// `</s>` when `<|tool_response>` is also present. Identical to the Phase 5
/// gate's construction (verified equal to the reference vocab's set).
pub fn eog_token_set(gguf: &GgufFile, tok: &Tokenizer) -> HashSet<i32> {
    let mut set = HashSet::new();
    for key in [
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.eot_token_id",
        "tokenizer.ggml.eom_token_id",
        "tokenizer.ggml.fim_pad_token_id",
        "tokenizer.ggml.fim_rep_token_id",
        "tokenizer.ggml.fim_sep_token_id",
    ] {
        if let Some(id) = gguf.metadata_u32(key) {
            set.insert(id as i32);
        }
    }
    for text in [
        "<|eot_id|>",
        "<|im_end|>",
        "<|end|>",
        "<|return|>",
        "<|call|>",
        "<|flush|>",
        "<|calls|>",
        "<end_of_turn>",
        "<|endoftext|>",
        "</s>",
        "<|eom_id|>",
        "<EOT>",
        "_<EOT>",
        "[EOT]",
        "[EOS]",
        "<|end_of_text|>",
        "<end_of_utterance>",
        "<eos>",
        "<turn|>",
        "<|tool_response>",
    ] {
        if let Some(id) = tok.token_id(text) {
            set.insert(id as i32);
        }
    }
    if let (Some(tool_resp), Some(s_tok)) = (tok.token_id("<|tool_response>"), tok.token_id("</s>"))
    {
        if set.contains(&(tool_resp as i32)) {
            set.remove(&(s_tok as i32));
        }
    }
    set
}

fn render_chat_prompt(tok: &Tokenizer, template: &str, user: &str) -> Result<String> {
    let eos = tok.token_text(tok.special.eos).unwrap_or("");
    let eot = tok.token_text(tok.special.eot).unwrap_or("");
    let mut env = Environment::new();
    // These chat templates use Python-style dict methods (e.g.
    // `message.get('reasoning')` to read optional keys) that minijinja does not
    // implement natively but llama.cpp's minja — the reference renderer — does.
    // Bridge the methods the templates rely on so render matches the reference.
    env.set_unknown_method_callback(|_state, value, method, args| {
        use minijinja::value::Value;
        use minijinja::{Error, ErrorKind};
        match method {
            // dict.get(key[, default]) -> value, or default/Undefined if absent.
            "get" => {
                let key = args.first().cloned().unwrap_or(Value::UNDEFINED);
                let default = args.get(1).cloned().unwrap_or(Value::UNDEFINED);
                Ok(value
                    .get_item(&key)
                    .ok()
                    .filter(|v| !v.is_undefined())
                    .unwrap_or(default))
            }
            _ => Err(Error::from(ErrorKind::UnknownMethod)),
        }
    });
    env.add_template_owned("chat", template.to_string())
        .map_err(|e| BackendError::InvalidModelMetadata(format!("chat template parse: {e}")))?;
    let compiled = env
        .get_template("chat")
        .map_err(|e| BackendError::InvalidModelMetadata(format!("chat template: {e}")))?;
    let mut msg: BTreeMap<&str, &str> = BTreeMap::new();
    msg.insert("role", "user");
    msg.insert("content", user);
    let messages = vec![msg];
    compiled
        .render(context! {
            messages => messages,
            add_generation_prompt => true,
            // The reference (llama.cpp) renders the template's `{{ bos_token }}`
            // as EMPTY and relies on tokenize(add_special=true) for the single
            // BOS, and applies the chat template with enable_thinking=true by
            // default — which emits the leading <|turn>system\n<|think|> block
            // and suppresses the trailing generation-prompt thought block.
            bos_token => "",
            eos_token => eos,
            eot_token => eot,
            enable_thinking => true,
        })
        .map_err(|e| BackendError::InvalidModelMetadata(format!("chat template render: {e}")))
}

/// A loaded DiffusionGemma chat session: the bit-exact runtime plus the
/// tokenizer, chat template, EOG set, and canvas length needed to turn a
/// user message into a response.
pub struct DgChat {
    runtime: DgEncoderRuntime,
    tokenizer: Tokenizer,
    template: Option<String>,
    eog: HashSet<i32>,
    canvas_length: usize,
}

impl DgChat {
    pub fn load(path: &Path) -> Result<Self> {
        let gguf = read_metadata(path)?;
        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        let template = tokenizer.chat_template.clone();
        let eog = eog_token_set(&gguf, &tokenizer);
        let canvas_length = gguf
            .metadata_u32("diffusion.canvas_length")
            .or_else(|| {
                gguf.metadata_string("diffusion.canvas_length")
                    .and_then(|s| s.parse().ok())
            })
            .ok_or_else(|| {
                BackendError::InvalidModelMetadata("missing diffusion.canvas_length".into())
            })? as usize;
        let runtime = DgEncoderRuntime::load(path)?;
        Ok(Self {
            runtime,
            tokenizer,
            template,
            eog,
            canvas_length,
        })
    }

    /// The denoise canvas width — the per-block answer length.
    pub fn canvas_length(&self) -> usize {
        self.canvas_length
    }

    /// Render the chat template for `user_message` and tokenize it into the
    /// prompt token ids that seed the multi-canvas loop — exactly the reference
    /// chat path (`apply_template(add_generation_prompt=true)` then
    /// `common_tokenize(add_special=true, parse_special=true)`). Exposed so the
    /// Phase 6 gate can check render+tokenize parity in isolation.
    pub fn render_prompt(&self, user_message: &str) -> Result<Vec<u32>> {
        let prompt_text = match &self.template {
            Some(t) => render_chat_prompt(&self.tokenizer, t, user_message)?,
            None => user_message.to_string(),
        };
        let parse_special = self.tokenizer.chat_prompt_parse_special();
        Ok(self
            .tokenizer
            .encode(&prompt_text, true, parse_special)?
            .to_vec())
    }

    /// Detokenize a response (the trimmed multi-canvas tokens) back to text —
    /// the reference's `common_detokenize(response, special=false)`.
    pub fn decode_response(&self, ids: &[i32]) -> Result<String> {
        let resp_ids: Vec<u32> = ids.iter().map(|&t| t as u32).collect();
        self.tokenizer.decode(&resp_ids, false)
    }

    /// One chat turn: render the template, tokenize, run the multi-canvas
    /// loop, detokenize. `on_block` observes each committed block. Returns
    /// the response text, the stop reason, and the response token ids.
    pub fn generate(
        &self,
        user_message: &str,
        params: &DgEbParams,
        n_blocks: i32,
        max_ubatch: i32,
        mut on_block: impl FnMut(usize, usize, usize, usize),
    ) -> Result<(String, String, Vec<i32>)> {
        let prompt = self.render_prompt(user_message)?;
        let (_blocks, response, stop) = self.runtime.mc_generate(
            &prompt,
            params,
            n_blocks,
            max_ubatch,
            &self.eog,
            |b, prefix, records, _canvas, cut| on_block(b, prefix.len(), records.len(), cut),
        )?;
        let text = self.decode_response(&response)?;
        Ok((text, stop, response))
    }

    /// [`generate_with_blocks`](Self::generate_with_blocks) plus a PER-STEP
    /// live preview: `on_preview(block, step_idx, draft_text)` fires after
    /// every denoise step with the decoded argmax canvas — the whole answer
    /// draft, visibly refining in place. Identical generation by construction.
    pub fn generate_live(
        &self,
        user_message: &str,
        params: &DgEbParams,
        n_blocks: i32,
        max_ubatch: i32,
        mut on_preview: impl FnMut(usize, i32, String),
        mut on_committed: impl FnMut(usize, &[i32]),
    ) -> Result<(String, String, Vec<i32>)> {
        let prompt = self.render_prompt(user_message)?;
        let (_blocks, response, stop) = self.runtime.mc_generate_with_steps(
            &prompt,
            params,
            n_blocks,
            max_ubatch,
            &self.eog,
            |b, step, canvas| {
                if let Ok(text) = self.decode_response(canvas) {
                    on_preview(b, step, text);
                }
            },
            |b, _prefix, _records, canvas, cut| on_committed(b, &canvas[..cut]),
        )?;
        let text = self.decode_response(&response)?;
        Ok((text, stop, response))
    }

    /// [`generate`](Self::generate) with a per-block committed-ids callback —
    /// the serve lane's block-level SSE source. `on_committed` receives the
    /// block index and the trimmed canvas ids that block committed (the
    /// response grows by exactly these ids). Identical generation by
    /// construction.
    pub fn generate_with_blocks(
        &self,
        user_message: &str,
        params: &DgEbParams,
        n_blocks: i32,
        max_ubatch: i32,
        mut on_committed: impl FnMut(usize, &[i32]),
    ) -> Result<(String, String, Vec<i32>)> {
        let prompt = self.render_prompt(user_message)?;
        let (_blocks, response, stop) = self.runtime.mc_generate(
            &prompt,
            params,
            n_blocks,
            max_ubatch,
            &self.eog,
            |b, _prefix, _records, canvas, cut| on_committed(b, &canvas[..cut]),
        )?;
        let text = self.decode_response(&response)?;
        Ok((text, stop, response))
    }
}
