//! Gemma 4 generation parity — exact-row, greedy, against the committed
//! llama.cpp oracle pack.
//!
//! For the model named by `CAMELID_GEMMA4_GGUF`, runs every prompt in
//! `qa/gemma4/prompt_packs/basic_v1.json` through [`Gemma4Runtime`] greedy
//! decode and asserts, per prompt:
//!   1. prompt token ids   == the oracle's tokenization,
//!   2. generated token ids == the oracle's greedy continuation,
//!   3. generated text      == the oracle's text.
//!
//! The oracle files live in `qa/gemma4/oracle/<row>.basic_v1.json` and were
//! captured from the pinned llama.cpp build (llama-server 5d56eff, CPU,
//! temperature=0/top_k=1, cache_prompt=false). A model file whose row has no
//! committed oracle FAILS (not skips): no raw evidence, no claim.
//!
//! Set `CAMELID_GEMMA4_GPU=1` to run the same assertions through the
//! GPU-resident runtime instead (macOS only).
//!
//! Run: `CAMELID_GEMMA4_GGUF=/path/gemma-4-E2B-it-Q8_0.gguf \
//!       cargo test --release --test gemma4_generation_parity -- --nocapture`

use std::path::PathBuf;

use camelid::gemma4_runtime::Gemma4Runtime;

#[derive(serde::Deserialize)]
struct Pack {
    pack_id: String,
    prompts: Vec<PackPrompt>,
    /// Context packs carry their target window; used to size the GPU runtime's
    /// KV capacity. Absent on the short packs.
    #[serde(default)]
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    target_context_window: Option<usize>,
}

#[derive(serde::Deserialize)]
struct PackPrompt {
    id: String,
    text: String,
    max_new_tokens: usize,
}

#[derive(serde::Deserialize)]
struct Oracle {
    row: String,
    pack_id: String,
    results: Vec<OracleResult>,
}

#[derive(serde::Deserialize)]
struct OracleResult {
    id: String,
    prompt_tokens: Vec<u32>,
    generated_tokens: Vec<u32>,
    generated_text: String,
    /// Recorded knife-edge frontier for the CPU runtime: full parity is
    /// asserted only for the first `parity_prefix_tokens` generated tokens of
    /// this prompt on CPU (the GPU runtime asserts the full budget). Used when
    /// a near-tie argmax (top-2 logit gap ~0.1% or less, measured and quoted
    /// in `reason`) resolves differently across accumulation orders — the same
    /// frontier class as the recorded TinyLlama deep-generation divergence.
    cpu_known_frontier: Option<CpuKnownFrontier>,
}

#[derive(serde::Deserialize)]
struct CpuKnownFrontier {
    parity_prefix_tokens: usize,
    reason: String,
    /// When camelid CPU and GPU agree with each other and BOTH sit on the other
    /// side of the reference's knife-edge, the frontier bounds the GPU
    /// assertion too. Default false (GPU asserts the full budget).
    #[serde(default)]
    applies_to_gpu: bool,
    /// GPU-specific verified prefix when it differs from the CPU one (the two
    /// runtimes can flip at different knife-edge positions).
    #[serde(default)]
    gpu_parity_prefix_tokens: Option<usize>,
}

fn repo_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// The row id is the model file stem — oracle packs are keyed by it, so the
/// wrong oracle can never be applied to the wrong file.
fn row_id(model_path: &std::path::Path) -> String {
    model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_string()
}

#[test]
fn gemma4_greedy_generation_matches_llama_cpp_oracle() {
    let Some(model) = std::env::var_os("CAMELID_GEMMA4_GGUF").map(PathBuf::from) else {
        eprintln!("SKIP gemma4 generation parity: set CAMELID_GEMMA4_GGUF");
        return;
    };
    let row = row_id(&model);
    // Pack selection: CAMELID_GEMMA4_PACK names the pack file stem under
    // qa/gemma4/prompt_packs/ (default basic_v1). The oracle is keyed by BOTH
    // the row and the pack stem, so mismatched artifacts can never pair up.
    let pack_stem = std::env::var("CAMELID_GEMMA4_PACK").unwrap_or_else(|_| "basic_v1".to_string());
    let pack: Pack = serde_json::from_str(
        &std::fs::read_to_string(repo_path(&format!(
            "qa/gemma4/prompt_packs/{pack_stem}.json"
        )))
        .expect("prompt pack"),
    )
    .expect("prompt pack json");
    let oracle_path = repo_path(&format!("qa/gemma4/oracle/{row}.{pack_stem}.json"));
    let oracle: Oracle =
        serde_json::from_str(&std::fs::read_to_string(&oracle_path).unwrap_or_else(|_| {
            panic!(
                "no committed oracle for row {row} at {} — capture the llama.cpp \
                 oracle before asserting anything about this row",
                oracle_path.display()
            )
        }))
        .expect("oracle json");
    assert_eq!(oracle.row, row, "oracle row mismatch");
    assert_eq!(oracle.pack_id, pack.pack_id, "oracle pack mismatch");

    let use_gpu = std::env::var("CAMELID_GEMMA4_GPU").is_ok_and(|v| v == "1");
    // Windows CUDA-resident lane: strict full-budget parity vs the same committed
    // oracle (no frontier relaxation — basic_v1 carries no cpu_known_frontier, so
    // the harness's default is already token-for-token). Feature-gated because
    // `Gemma4CudaResident` only compiles under `--features cuda`.
    let use_cuda = std::env::var("CAMELID_GEMMA4_CUDA").is_ok_and(|v| v == "1");
    eprintln!(
        "row {row}: {} prompts, runtime = {}",
        pack.prompts.len(),
        if use_cuda {
            "cuda-resident"
        } else if use_gpu {
            "gpu-resident"
        } else {
            "cpu"
        }
    );

    let cpu_runtime = if use_gpu || use_cuda {
        None
    } else {
        Some(Gemma4Runtime::load(&model).expect("load gemma4 runtime"))
    };
    #[cfg(target_os = "macos")]
    let gpu_runtime = if use_gpu {
        // KV capacity: the context window when the pack declares one, else
        // enough for the longest prompt + budget.
        let max_positions = pack.target_context_window.unwrap_or_else(|| {
            pack.prompts
                .iter()
                .map(|p| p.text.len() / 2 + p.max_new_tokens)
                .max()
                .unwrap_or(192)
        }) + 64;
        Some(
            camelid::gemma4_runtime::Gemma4GpuRuntime::load(&model, max_positions)
                .expect("load gemma4 gpu runtime"),
        )
    } else {
        None
    };
    #[cfg(not(target_os = "macos"))]
    if use_gpu {
        panic!("CAMELID_GEMMA4_GPU=1 requires macOS/Metal");
    }

    // CUDA-resident runtime (Windows/NVIDIA). KV capacity: the pack's declared
    // window, else the longest prompt + budget with headroom — same sizing as the
    // Metal branch.
    #[cfg(feature = "cuda")]
    let mut cuda_runtime = if use_cuda {
        let max_positions = pack.target_context_window.unwrap_or_else(|| {
            pack.prompts
                .iter()
                .map(|p| p.text.len() / 2 + p.max_new_tokens)
                .max()
                .unwrap_or(192)
        }) + 64;
        Some(
            camelid::gemma4_runtime::Gemma4CudaResident::load(&model, max_positions)
                .expect("load gemma4 cuda runtime"),
        )
    } else {
        None
    };
    #[cfg(not(feature = "cuda"))]
    if use_cuda {
        panic!("CAMELID_GEMMA4_CUDA=1 requires a --features cuda build");
    }

    let mut failures = Vec::new();
    for prompt in &pack.prompts {
        let expected = oracle
            .results
            .iter()
            .find(|r| r.id == prompt.id)
            .unwrap_or_else(|| panic!("oracle missing prompt {}", prompt.id));

        // Prompt-token parity (BOS + plain text, no special parsing of body).
        // The tokenizer borrow is scoped here so it releases before the CUDA
        // runtime's `&mut self` generate below.
        let prompt_tokens = {
            let runtime_tokenizer = if let Some(rt) = cpu_runtime.as_ref() {
                rt.tokenizer()
            } else if use_cuda {
                #[cfg(feature = "cuda")]
                {
                    cuda_runtime.as_ref().unwrap().tokenizer()
                }
                #[cfg(not(feature = "cuda"))]
                unreachable!()
            } else {
                #[cfg(target_os = "macos")]
                {
                    gpu_runtime.as_ref().unwrap().tokenizer()
                }
                #[cfg(not(target_os = "macos"))]
                unreachable!()
            };
            runtime_tokenizer
                .encode(&prompt.text, true, true)
                .expect("encode prompt")
        };
        if prompt_tokens != expected.prompt_tokens {
            failures.push(format!(
                "{}: prompt tokens diverge\n  camelid: {prompt_tokens:?}\n  oracle:  {:?}",
                prompt.id, expected.prompt_tokens
            ));
            continue;
        }

        let (text, generated) = if let Some(rt) = cpu_runtime.as_ref() {
            rt.generate_greedy(&prompt.text, prompt.max_new_tokens)
                .expect("generate")
        } else if use_cuda {
            #[cfg(feature = "cuda")]
            {
                cuda_runtime
                    .as_mut()
                    .unwrap()
                    .generate_greedy(&prompt.text, prompt.max_new_tokens)
                    .expect("generate (cuda)")
            }
            #[cfg(not(feature = "cuda"))]
            unreachable!()
        } else {
            #[cfg(target_os = "macos")]
            {
                gpu_runtime
                    .as_ref()
                    .unwrap()
                    .generate_greedy(&prompt.text, prompt.max_new_tokens)
                    .expect("generate (gpu)")
            }
            #[cfg(not(target_os = "macos"))]
            unreachable!()
        };

        // A recorded knife-edge frontier bounds the CPU assertion to its measured
        // prefix. The GPU lanes (Metal and CUDA) only honor a frontier the oracle
        // marks `applies_to_gpu` (or that carries a GPU-specific prefix); a CPU-only
        // frontier does NOT relax the GPU/CUDA assertion, which then stays strict.
        let gpu_lane = use_gpu || use_cuda;
        let frontier_active = expected
            .cpu_known_frontier
            .as_ref()
            .filter(|f| !gpu_lane || f.applies_to_gpu || f.gpu_parity_prefix_tokens.is_some());
        if let Some(frontier) = frontier_active {
            let n = if gpu_lane {
                frontier
                    .gpu_parity_prefix_tokens
                    .unwrap_or(frontier.parity_prefix_tokens)
            } else {
                frontier.parity_prefix_tokens
            };
            if generated.len() < n || generated[..n] != expected.generated_tokens[..n] {
                failures.push(format!(
                    "{}: generated tokens diverge INSIDE the recorded {n}-token frontier\n  \
                     camelid: {generated:?}\n  oracle:  {:?}",
                    prompt.id, expected.generated_tokens
                ));
                continue;
            }
            eprintln!(
                "  {} OK to recorded CPU frontier ({n} of {} oracle tokens): {}",
                prompt.id,
                expected.generated_tokens.len(),
                frontier.reason
            );
            continue;
        }
        if generated != expected.generated_tokens {
            failures.push(format!(
                "{}: generated tokens diverge\n  camelid: {generated:?}\n  oracle:  {:?}",
                prompt.id, expected.generated_tokens
            ));
            continue;
        }
        if text != expected.generated_text {
            failures.push(format!(
                "{}: generated text diverges\n  camelid: {text:?}\n  oracle:  {:?}",
                prompt.id, expected.generated_text
            ));
            continue;
        }
        eprintln!(
            "  {} OK ({} tokens): {:?}",
            prompt.id,
            generated.len(),
            text
        );
    }

    assert!(
        failures.is_empty(),
        "row {row}: {} of {} prompts diverged from the llama.cpp oracle:\n{}",
        failures.len(),
        pack.prompts.len(),
        failures.join("\n")
    );
}
