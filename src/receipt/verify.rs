//! Standalone receipt verifier behind `camelid verify-receipt`.
//!
//! Anyone with the receipt, the exact GGUF, and (optionally) a llama.cpp
//! `llama-server` binary can independently prove a receipt honest. The
//! verifier asserts exactly what a receipt claims — one request, one lane —
//! and nothing more: a verified receipt is not a support promotion, and a
//! non-reproducible receipt is never reported as verified.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::{sha256_file_hex, ParityReceipt, ReceiptResult, NO_DIVERGENCE, RECEIPT_SCHEMA_V1};

/// First index where the two token sequences differ (length mismatch counts
/// as divergence at the shorter length), or [`NO_DIVERGENCE`] when identical.
/// Mirrors `firstDifference()` in `scripts/chat-parity-tinyllama.mjs` so the
/// Rust verifier and the parity harness agree on what "match" means.
pub fn first_divergent_index(left: &[u32], right: &[u32]) -> i64 {
    let max = left.len().max(right.len());
    for index in 0..max {
        if left.get(index) != right.get(index) {
            return index as i64;
        }
    }
    NO_DIVERGENCE
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyMode {
    /// All steps: self-digest, reproducibility, lane identity, Camelid
    /// re-run, reference re-run.
    Full,
    /// Skip the llama.cpp reference re-run (e.g. no llama.cpp installed).
    SelfOnly,
    /// Skip the in-process Camelid re-run.
    ReferenceOnly,
}

pub struct VerifyOptions {
    pub receipt_path: PathBuf,
    pub gguf: PathBuf,
    pub llama_server: String,
    pub mode: VerifyMode,
    pub llama_ctx: u32,
    pub llama_port: u16,
    pub threads: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyOutcome {
    /// Steps 1, 3, 4, 5 all passed for a reproducible receipt.
    Verified,
    /// Every attempted step passed, but a half was skipped via
    /// `--self-only` / `--reference-only`; full parity is NOT asserted.
    PartiallyVerified,
    /// The receipt is stamped `reproducible: false`; parity cannot be
    /// asserted. Only the self-digest and lane identity were checked.
    NotReproducible,
    /// The receipt's own parity block records a divergence at emit time; it
    /// is a divergence record, not a parity claim, and is never reported as
    /// `RECEIPT VERIFIED`.
    DivergenceRecord,
    /// The receipt carries a `lossy` quality tier. A lossy format does not
    /// produce the headline token stream, so it is reported as a quality
    /// ATTESTATION (its measured same-format agreement and ppl delta), never as
    /// a `RECEIPT VERIFIED` parity contract. Distinct from `NotVerified`: the
    /// receipt is honest, it just makes a weaker (non-parity) claim.
    LossyAttestation,
    /// A verification step failed.
    NotVerified,
}

impl VerifyOutcome {
    pub fn exit_code(self) -> i32 {
        match self {
            Self::Verified | Self::PartiallyVerified => 0,
            Self::NotVerified => 1,
            Self::NotReproducible => 2,
            Self::DivergenceRecord => 3,
            // Honest non-parity attestation: distinct from a hard failure (1)
            // and from a divergence record (3).
            Self::LossyAttestation => 4,
        }
    }
}

/// The format-honesty invariant on a receipt's quality tier, factored out pure
/// so it is unit-testable without a model or a server.
///
/// Returns:
/// - `Ok(None)` — no tier, or a headline-Q8 tier: defer to the normal parity flow.
/// - `Ok(Some(()))` — a HONEST lossy tier: the caller reports a [`VerifyOutcome::LossyAttestation`]
///   (never `RECEIPT VERIFIED`). A lossy tier must compare against the SAME format.
/// - `Err(reason)` — a DISHONEST tier (e.g. a lossy run claiming a cross-format match);
///   the caller reports [`VerifyOutcome::NotVerified`].
pub fn check_format_honesty(receipt: &ParityReceipt) -> Result<Option<()>, String> {
    use crate::receipt::QualityTierKind;
    let Some(tier) = &receipt.quality_tier else {
        return Ok(None);
    };
    if tier.schema != crate::receipt::QUALITY_TIER_SCHEMA_V1 {
        return Err(format!(
            "unknown quality-tier schema {:?} (this verifier understands {:?})",
            tier.schema,
            crate::receipt::QUALITY_TIER_SCHEMA_V1
        ));
    }
    match tier.tier {
        QualityTierKind::HeadlineQ8 => Ok(None),
        QualityTierKind::Lossy => {
            if !tier.is_format_honest() {
                Err(format!(
                    "lossy quality tier compares format {:?} against a DIFFERENT baseline \
                     format {:?}; a lossy tier may only be measured against its own format \
                     (never a cross-format token match)",
                    tier.format, tier.baseline_format
                ))
            } else {
                Ok(Some(()))
            }
        }
    }
}

/// Run the verification steps in order, printing one PASS/FAIL line each,
/// then a single final verdict line.
pub async fn run(options: VerifyOptions) -> VerifyOutcome {
    // Step 0 (implicit): the receipt must parse and carry the known schema.
    let raw = match std::fs::read_to_string(&options.receipt_path) {
        Ok(raw) => raw,
        Err(err) => {
            println!(
                "FAIL self-digest: could not read {}: {err}",
                options.receipt_path.display()
            );
            return not_verified("self-digest");
        }
    };
    let receipt: ParityReceipt = match serde_json::from_str(&raw) {
        Ok(receipt) => receipt,
        Err(err) => {
            println!("FAIL self-digest: receipt does not parse: {err}");
            return not_verified("self-digest");
        }
    };
    if receipt.schema != RECEIPT_SCHEMA_V1 {
        println!(
            "FAIL self-digest: unknown schema {:?} (this verifier understands {RECEIPT_SCHEMA_V1:?})",
            receipt.schema
        );
        return not_verified("self-digest");
    }

    // Format-honesty gate (pure): a lossy quality tier must compare against the SAME
    // format and is reported as a quality ATTESTATION, never as RECEIPT VERIFIED. A
    // cross-format lossy tier is dishonest and fails outright. Headline-Q8 / no tier
    // defers to the normal parity flow. Checked before any model load so the verdict is
    // cheap and unambiguous.
    match check_format_honesty(&receipt) {
        Ok(None) => {}
        Ok(Some(())) => {
            let tier = receipt
                .quality_tier
                .as_ref()
                .expect("lossy tier present (check_format_honesty returned Some)");
            // Still confirm the receipt is not tampered.
            if let Err(err) = receipt.verify_self_digest() {
                println!("FAIL self-digest: {err}; the receipt is tampered or malformed");
                return not_verified("self-digest");
            }
            println!(
                "PASS self-digest: receipt_id matches the canonical body ({})",
                receipt.receipt_id
            );
            println!(
                "NOTE quality-tier: this is a LOSSY {} receipt (baseline {} {} @ {}); it does \
                 NOT produce the headline token stream. Reported as a quality attestation: \
                 greedy_agreement={:.3}%, ppl_delta={:.4} (measured against the SAME format).",
                tier.format,
                tier.baseline_engine,
                tier.baseline_format,
                tier.baseline_commit,
                tier.greedy_agreement_pct,
                tier.ppl_delta
            );
            println!(
                "RECEIPT IS A LOSSY QUALITY ATTESTATION (honest non-parity claim; not verified \
                 as parity)"
            );
            return VerifyOutcome::LossyAttestation;
        }
        Err(reason) => {
            println!("FAIL quality-tier: {reason}");
            return not_verified("quality-tier");
        }
    }

    // Full verification runs as two isolated subprocess passes — a reference
    // pass and a self pass — each loading exactly ONE model and fully exiting
    // (so the OS reclaims its entire footprint) before the next starts. Two
    // co-resident 7.7 GB models OOM-kill one of them on a 16 GB host; isolated
    // passes keep at most one model resident, so a 7B receipt verifies on
    // consumer hardware. The passes invoke this same binary in --reference-only
    // and --self-only mode, whose single-pass flow is the in-process code below.
    if options.mode == VerifyMode::Full {
        // Size the reference llama-server's context to what THIS receipt actually
        // needs (its prompt + generated tokens, plus margin) instead of a fixed
        // large default. A receipt records one bounded generation, so this keeps
        // the reference's KV-cache working set small, which matters when it loads
        // right after the Camelid pass left the model's file pages in the cache.
        let needed_ctx =
            (receipt.result.prompt_token_ids.len() + receipt.result.generated_token_ids.len() + 16)
                .next_multiple_of(64)
                .clamp(64, options.llama_ctx as usize) as u32;
        return run_full_via_isolated_passes(&options, needed_ctx);
    }

    // Step 1: self-digest (cheap tamper check).
    match receipt.verify_self_digest() {
        Ok(()) => println!(
            "PASS self-digest: receipt_id matches the canonical body ({})",
            receipt.receipt_id
        ),
        Err(err) => {
            println!("FAIL self-digest: {err}; the receipt is tampered or malformed");
            return not_verified("self-digest");
        }
    }

    // Step 3 runs for every receipt, including non-reproducible ones.
    let lane_identity_ok = check_lane_identity(&receipt, &options);

    // Step 2: reproducibility gate.
    if !receipt.reproducible {
        println!(
            "NOTE reproducibility: this receipt is stamped reproducible:false (sampled run); \
             parity cannot be asserted for a non-deterministic generation. Only the \
             self-digest and lane identity were checked."
        );
        println!("RECEIPT NOT VERIFIABLE (non-reproducible receipt)");
        return VerifyOutcome::NotReproducible;
    }
    println!("PASS reproducibility: receipt records a deterministic (greedy) run");

    if !lane_identity_ok {
        return not_verified("lane-identity");
    }

    // Divergence-record gate: a receipt whose own parity block records a
    // mismatch at emit time is a divergence record, not a parity claim.
    // Verifying it must never print RECEIPT VERIFIED — that would assert
    // more than the receipt itself claims.
    let records_divergence = [
        receipt.parity.prompt_tokens_match,
        receipt.parity.generated_tokens_match,
        receipt.parity.generated_text_match,
    ]
    .contains(&Some(false));
    if records_divergence {
        println!(
            "NOTE divergence-record: this receipt's parity block records a mismatch against \
             the reference at emit time (it documents non-parity). The digest and lane \
             identity above are checked; parity is not asserted."
        );
        println!("RECEIPT IS A DIVERGENCE RECORD (not a parity claim; not verified as parity)");
        return VerifyOutcome::DivergenceRecord;
    }

    // Reached only in --self-only / --reference-only mode (Full delegates to two
    // isolated passes above), so at most one of the two re-runs below executes
    // and they are never co-resident.

    // Step 4: Camelid re-run (proves Camelid is internally deterministic for
    // this lane and that the receipt's recorded output is real). Its AppState
    // and weights are owned within `replay_receipt_request` and dropped on
    // return.
    if options.mode != VerifyMode::ReferenceOnly {
        // An execution-trace block pins the deterministic CPU lane and a host ISA. We can only
        // re-derive its rollup on the same ISA (the Q8_0 dot rounds differently across ISAs),
        // so check the trace only when this host matches; otherwise re-run on the normal lane
        // and skip the digest comparison with an honest note.
        let trace_check = receipt
            .execution_trace
            .as_ref()
            .map(|trace| trace.host_isa == crate::receipt::host_isa_marker())
            .unwrap_or(false);
        if trace_check {
            force_deterministic_lane();
        }
        match crate::api::replay_receipt_request(&options.gguf, options.threads, &receipt.request)
            .await
        {
            Ok(replay) => {
                if let Err(detail) = compare_results(&receipt.result, &replay.result) {
                    println!("FAIL camelid-rerun: {detail}");
                    return not_verified("camelid-rerun");
                }
                println!(
                    "PASS camelid-rerun: prompt tokens ({}), generated tokens ({}), and text \
                     match the receipt",
                    replay.result.prompt_token_ids.len(),
                    replay.result.generated_token_ids.len()
                );
                if let Some(trace) = &receipt.execution_trace {
                    if !trace_check {
                        println!(
                            "SKIP execution-trace: receipt rollup is for ISA {} but this host is {}; \
                             the digest is ISA-specific and cannot be re-derived here",
                            trace.host_isa,
                            crate::receipt::host_isa_marker()
                        );
                    } else {
                        match &replay.execution_trace_digest {
                            Some(rederived) if *rederived == trace.digest => {
                                println!(
                                    "PASS execution-trace: re-derived the {} rollup digest \
                                     identically (lane {}, ISA {}, {} checkpoints)",
                                    trace.algorithm, trace.lane, trace.host_isa, trace.fold_count
                                );
                            }
                            Some(rederived) => {
                                println!(
                                    "FAIL execution-trace: re-derived rollup {rederived} != receipt \
                                     {}",
                                    trace.digest
                                );
                                return not_verified("execution-trace");
                            }
                            None => {
                                println!(
                                    "FAIL execution-trace: the re-run produced no rollup (the \
                                     deterministic lane did not arm)"
                                );
                                return not_verified("execution-trace");
                            }
                        }
                    }
                }
            }
            Err(err) => {
                println!("FAIL camelid-rerun: {err}");
                return not_verified("camelid-rerun");
            }
        }
    } else {
        println!("SKIP camelid-rerun: --reference-only");
    }

    // Step 5: reference re-run against llama.cpp, started now that the Camelid
    // replay has returned and its model memory is released.
    if options.mode != VerifyMode::SelfOnly {
        let outcome = start_reference_server(&options, receipt.reference.version.as_deref())
            .and_then(|server| reference_rerun(&receipt, &options, &server));
        match outcome {
            Ok(()) => {}
            Err(detail) => {
                println!("FAIL reference-rerun: {detail}");
                return not_verified("reference-rerun");
            }
        }
    } else {
        println!("SKIP reference-rerun: --self-only");
    }

    // Step 6: verdict.
    match options.mode {
        VerifyMode::Full => {
            println!("RECEIPT VERIFIED");
            VerifyOutcome::Verified
        }
        VerifyMode::SelfOnly => {
            println!(
                "RECEIPT PARTIALLY VERIFIED (self checks only: digest, lane identity, and \
                 Camelid's own determinism; the llama.cpp reference re-run was skipped, so \
                 full parity is NOT asserted)"
            );
            VerifyOutcome::PartiallyVerified
        }
        VerifyMode::ReferenceOnly => {
            println!(
                "RECEIPT PARTIALLY VERIFIED (reference checks only: digest, lane identity, and \
                 the llama.cpp re-run; the Camelid re-run was skipped)"
            );
            VerifyOutcome::PartiallyVerified
        }
    }
}

fn not_verified(step: &str) -> VerifyOutcome {
    println!("RECEIPT NOT VERIFIED (failed step: {step})");
    VerifyOutcome::NotVerified
}

/// Pin this verifier process to the deterministic CPU lane so a receipt's execution-trace
/// rollup re-derives identically: enable deterministic mode and force the whole Metal/GPU
/// stack off. Mirrors the CLI `--deterministic` arm; verify-receipt is a one-shot command, so
/// mutating process env here is safe.
fn force_deterministic_lane() {
    std::env::set_var("CAMELID_DETERMINISTIC", "1");
    for key in [
        "CAMELID_METAL_RESIDENT_DECODE",
        "CAMELID_METAL_F32Y",
        "CAMELID_METAL_WIRE",
        "CAMELID_METAL_WIRE_NSG8",
        "CAMELID_METAL_ATTN2",
        "CAMELID_METAL_RESIDENT_PREFILL",
        "CAMELID_METAL_MM",
        "CAMELID_METAL_LINEAR",
        "CAMELID_METAL_Q8",
        "CAMELID_METAL_Q8_RETAINED",
        "CAMELID_HYBRID_Q8_RETAINED",
        "CAMELID_METAL_NOCOPY",
    ] {
        std::env::set_var(key, "0");
    }
    std::env::set_var("CAMELID_NO_GPU_SAMPLE", "1");
}

/// Full verification as two isolated passes. Each pass invokes this same binary
/// in a single mode (`--reference-only`, then `--self-only`), so it loads
/// exactly one model and exits — the OS reclaims its full footprint before the
/// next pass starts. This keeps at most one model resident at a time, which is
/// what lets a large receipt (a 7B Q8 is ~7.7 GB) verify on a 16 GB host where
/// two co-resident loads would OOM-kill one of them.
fn run_full_via_isolated_passes(options: &VerifyOptions, reference_ctx: u32) -> VerifyOutcome {
    let exe = match std::env::current_exe() {
        Ok(path) => path,
        Err(err) => {
            println!(
                "RECEIPT NOT VERIFIED (could not locate the camelid executable to spawn \
                 isolated verification passes: {err})"
            );
            return VerifyOutcome::NotVerified;
        }
    };
    println!(
        "Verifying in two isolated passes — each loads one model and is reclaimed \
         before the next, so a large receipt verifies within one model's memory footprint."
    );

    // Camelid pass FIRST, while the box is cleanest. It allocates the model as
    // anonymous heap, which must fit or OOM and is the hard constraint; the
    // reference pass's llama-server maps the file instead, which can reuse or
    // reclaim page cache. Running the reference first leaves ~7.7 GB of its file
    // pages cached and the second heap load then contends with them and OOMs.
    println!("\n=== camelid pass (in-process replay) ===");
    match run_isolated_verify_pass(&exe, options, IsolatedPass::SelfReplay) {
        0 => {}
        code => {
            println!("\nRECEIPT NOT VERIFIED (camelid pass failed)");
            return outcome_from_exit_code(code);
        }
    }

    // Reference pass: loads only the llama.cpp reference model, after the
    // Camelid pass subprocess has exited and its heap is reclaimed.
    println!("\n=== reference pass (llama.cpp) ===");
    match run_isolated_verify_pass(&exe, options, IsolatedPass::Reference { reference_ctx }) {
        0 => {}
        code => {
            println!("\nRECEIPT NOT VERIFIED (reference pass failed)");
            return outcome_from_exit_code(code);
        }
    }

    println!(
        "\nRECEIPT VERIFIED (self-digest, lane identity, Camelid replay, and llama.cpp \
         reference re-run all passed across isolated passes)"
    );
    VerifyOutcome::Verified
}

#[derive(Clone, Copy)]
enum IsolatedPass {
    Reference { reference_ctx: u32 },
    SelfReplay,
}

/// Spawn one single-mode verification pass as a child of this binary, streaming
/// its output through, and return its exit code (0 = that half verified).
fn run_isolated_verify_pass(
    exe: &std::path::Path,
    options: &VerifyOptions,
    pass: IsolatedPass,
) -> i32 {
    let mut cmd = Command::new(exe);
    cmd.arg("verify-receipt")
        .arg(&options.receipt_path)
        .arg("--gguf")
        .arg(&options.gguf);
    match pass {
        IsolatedPass::SelfReplay => {
            cmd.arg("--self-only");
            // Load the replay's weights as page-aligned mmap wire pages instead
            // of ~7 GB of anonymous heap blocks: file-backed pages are
            // reclaimable page cache, so a 7B replay fits on a host with only a
            // few GB free (the heap path OOMs there). Same GPU kernels, so the
            // replayed tokens are identical. Ignored on non-macOS / without the
            // wire stack, where it falls back to the block path.
            cmd.env("CAMELID_METAL_NOCOPY", "1");
        }
        IsolatedPass::Reference { reference_ctx } => {
            cmd.arg("--reference-only")
                .arg("--llama-server")
                .arg(&options.llama_server)
                .arg("--llama-ctx")
                .arg(reference_ctx.to_string())
                .arg("--llama-port")
                .arg(options.llama_port.to_string());
        }
    }
    if let Some(threads) = options.threads {
        cmd.arg("--threads").arg(threads.to_string());
    }
    // Inherit stdio so each pass's own PASS/FAIL lines stream straight through.
    match cmd.status() {
        Ok(status) => status.code().unwrap_or(1),
        Err(err) => {
            println!("FAIL: could not spawn the isolated verification pass: {err}");
            1
        }
    }
}

/// Map a child pass's exit code back to the orchestrator's outcome. Mirrors
/// `VerifyOutcome::exit_code` (0 is handled by the caller as success).
fn outcome_from_exit_code(code: i32) -> VerifyOutcome {
    match code {
        2 => VerifyOutcome::NotReproducible,
        3 => VerifyOutcome::DivergenceRecord,
        4 => VerifyOutcome::LossyAttestation,
        _ => VerifyOutcome::NotVerified,
    }
}

/// Step 3: the supplied GGUF must be the exact file the receipt names.
fn check_lane_identity(receipt: &ParityReceipt, options: &VerifyOptions) -> bool {
    match sha256_file_hex(&options.gguf) {
        Ok(actual) if actual == receipt.lane.gguf_sha256 => {
            println!("PASS lane-identity: --gguf sha256 matches lane.gguf_sha256 ({actual})");
            true
        }
        Ok(actual) => {
            println!(
                "FAIL lane-identity: --gguf sha256 is {actual} but the receipt names \
                 {}; this receipt is not about this file",
                receipt.lane.gguf_sha256
            );
            false
        }
        Err(err) => {
            println!("FAIL lane-identity: could not hash --gguf: {err}");
            false
        }
    }
}

fn compare_results(expected: &ReceiptResult, actual: &ReceiptResult) -> Result<(), String> {
    let prompt_divergence =
        first_divergent_index(&expected.prompt_token_ids, &actual.prompt_token_ids);
    if prompt_divergence != NO_DIVERGENCE {
        return Err(format!(
            "prompt token ids diverge at index {prompt_divergence} (receipt has {} tokens, \
             re-run produced {})",
            expected.prompt_token_ids.len(),
            actual.prompt_token_ids.len()
        ));
    }
    let generated_divergence =
        first_divergent_index(&expected.generated_token_ids, &actual.generated_token_ids);
    if generated_divergence != NO_DIVERGENCE {
        return Err(format!(
            "generated token ids diverge at index {generated_divergence} (receipt has {} \
             tokens, re-run produced {})",
            expected.generated_token_ids.len(),
            actual.generated_token_ids.len()
        ));
    }
    if expected.generated_text != actual.generated_text {
        return Err(format!(
            "generated text differs: receipt {:?} vs re-run {:?}",
            expected.generated_text, actual.generated_text
        ));
    }
    Ok(())
}

/// A running reference `llama-server` instance for step 5.
struct ReferenceServer {
    host: &'static str,
    port: u16,
    _child: ChildGuard,
}

/// Spawn `llama-server` on the receipt's GGUF and wait until it is healthy.
/// Called before the in-process Camelid replay loads any weights.
fn start_reference_server(
    options: &VerifyOptions,
    recorded_version: Option<&str>,
) -> Result<ReferenceServer, String> {
    if let Some(version) = llama_server_version(&options.llama_server) {
        match recorded_version {
            Some(recorded) if recorded != version => println!(
                "INFO reference-rerun: local llama-server reports {version:?}; the receipt \
                 was made against {recorded:?} (informational; parity is still re-checked \
                 byte-for-byte below)"
            ),
            _ => println!("INFO reference-rerun: llama-server version {version:?}"),
        }
    }

    let host = "127.0.0.1";
    if TcpStream::connect((host, options.llama_port)).is_ok() {
        return Err(format!(
            "port {} is already in use; pass a free --llama-port",
            options.llama_port
        ));
    }
    let child = Command::new(&options.llama_server)
        .args([
            "--host",
            host,
            "--port",
            &options.llama_port.to_string(),
            "-m",
            &options.gguf.display().to_string(),
            "-ngl",
            "0",
            "-c",
            &options.llama_ctx.to_string(),
            "--no-warmup",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|err| {
            format!(
                "could not start llama-server binary {:?}: {err}; pass --llama-server or use \
                 --self-only",
                options.llama_server
            )
        })?;
    let mut child = ChildGuard(child);

    wait_for_health(host, options.llama_port, &mut child)?;
    Ok(ReferenceServer {
        host,
        port: options.llama_port,
        _child: child,
    })
}

/// Step 5: feed the receipt's exact prompt token ids to the reference
/// `llama-server` (`/completion` accepts token arrays) and compare the
/// continuation. Feeding token ids pins the comparison to the exact prompt
/// the receipt claims — cross-engine chat-template rendering legitimately
/// differs (system preambles, dates), and template/tokenizer equivalence is
/// attested at emit time by the parity harness, not re-derived here.
fn reference_rerun(
    receipt: &ParityReceipt,
    _options: &VerifyOptions,
    server: &ReferenceServer,
) -> Result<(), String> {
    let request = &receipt.request;
    let mut payload = json!({
        "prompt": receipt.result.prompt_token_ids,
        "n_predict": request.max_tokens,
        "stream": false,
        "temperature": request.temperature,
        "cache_prompt": false,
        "return_tokens": true,
    });
    if let Some(top_p) = request.top_p {
        payload["top_p"] = json!(top_p);
    }
    if let Some(top_k) = request.top_k {
        payload["top_k"] = json!(top_k);
    }
    if let Some(seed) = request.seed {
        payload["seed"] = json!(seed);
    }
    if !request.stop.is_empty() {
        payload["stop"] = json!(request.stop);
    }
    println!(
        "INFO reference-rerun: prompt fed as the receipt's exact {} token ids",
        receipt.result.prompt_token_ids.len()
    );
    let (status, response) = http_json(
        server.host,
        server.port,
        "POST",
        "/completion",
        Some(&payload),
        Duration::from_secs(900),
    )?;
    if status != 200 {
        return Err(format!(
            "llama-server /completion failed ({status}): {response}"
        ));
    }
    let llama_text = response["content"].as_str().unwrap_or_default().to_string();
    // Newer builds return the generated token ids directly; older builds get
    // the text re-tokenized as a fallback.
    let llama_generated: Vec<u32> = match response["tokens"].as_array() {
        Some(tokens) if !tokens.is_empty() => tokens
            .iter()
            .filter_map(|token| token.as_u64().map(|id| id as u32))
            .collect(),
        _ if llama_text.is_empty() => Vec::new(),
        _ => tokenize(server.host, server.port, &llama_text, false)?,
    };

    let receipt_ids = &receipt.result.generated_token_ids;
    let divergence = first_divergent_index(receipt_ids, &llama_generated);
    // llama.cpp reports the stop token inconsistently across builds: accept a
    // single missing trailing stop token when the run finished on a stop and
    // every preceding token matches — and say so explicitly.
    let trailing_stop_only = divergence != NO_DIVERGENCE
        && receipt.result.finish_reason == "stop"
        && receipt_ids.len() == llama_generated.len() + 1
        && divergence == llama_generated.len() as i64;
    if divergence != NO_DIVERGENCE && !trailing_stop_only {
        return Err(format!(
            "generated token ids diverge from llama.cpp at index {divergence} (receipt has \
             {} tokens, llama.cpp produced {})",
            receipt_ids.len(),
            llama_generated.len()
        ));
    }
    if receipt.result.generated_text != llama_text {
        return Err(format!(
            "generated text differs from llama.cpp: receipt {:?} vs llama.cpp {:?}",
            receipt.result.generated_text, llama_text
        ));
    }
    if trailing_stop_only {
        println!(
            "PASS reference-rerun: generated tokens ({}) and text match llama.cpp; the \
             receipt additionally records the trailing stop token {} that llama.cpp does \
             not report (first_divergent_token_index={NO_DIVERGENCE})",
            llama_generated.len(),
            receipt_ids.last().copied().unwrap_or_default()
        );
    } else {
        println!(
            "PASS reference-rerun: generated tokens ({}) and text match llama.cpp \
             (first_divergent_token_index={NO_DIVERGENCE})",
            llama_generated.len()
        );
    }
    Ok(())
}

fn tokenize(host: &str, port: u16, content: &str, add_special: bool) -> Result<Vec<u32>, String> {
    let (status, response) = http_json(
        host,
        port,
        "POST",
        "/tokenize",
        Some(&json!({ "content": content, "add_special": add_special })),
        Duration::from_secs(60),
    )?;
    if status != 200 {
        return Err(format!(
            "llama-server /tokenize failed ({status}): {response}"
        ));
    }
    Ok(response["tokens"]
        .as_array()
        .map(|tokens| {
            tokens
                .iter()
                .filter_map(|token| token.as_u64().map(|id| id as u32))
                .collect()
        })
        .unwrap_or_default())
}

fn wait_for_health(host: &str, port: u16, child: &mut ChildGuard) -> Result<(), String> {
    // Large models cold-loading from slow external storage (a 7B Q8 is ~7.7 GB)
    // can spend minutes in llama.cpp's "Loading model" state before /health
    // returns 200 — and the receipt's own Camelid replay runs first with
    // F_NOCACHE, so it does not warm the page cache for the reference. The cap
    // is generous so a real load is never mistaken for a hang; a genuinely dead
    // server is caught immediately by the child-exit check below, not by this
    // deadline.
    const HEALTH_TIMEOUT_SECS: u64 = 900;
    let started = Instant::now();
    let deadline = started + Duration::from_secs(HEALTH_TIMEOUT_SECS);
    let mut last_error = String::from("no response");
    while Instant::now() < deadline {
        if let Ok(Some(status)) = child.0.try_wait() {
            return Err(format!("llama-server exited during startup ({status})"));
        }
        match http_json(host, port, "GET", "/health", None, Duration::from_secs(5)) {
            Ok((200, _)) => return Ok(()),
            Ok((status, body)) => last_error = format!("{status}: {body}"),
            Err(err) => last_error = err,
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    Err(format!(
        "llama-server did not become healthy on {host}:{port} within {}s (last: {last_error})",
        started.elapsed().as_secs()
    ))
}

fn llama_server_version(binary: &str) -> Option<String> {
    let output = Command::new(binary)
        .arg("--version")
        .stdin(Stdio::null())
        .output()
        .ok()?;
    let text = if output.stdout.is_empty() {
        String::from_utf8_lossy(&output.stderr).to_string()
    } else {
        String::from_utf8_lossy(&output.stdout).to_string()
    };
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(ToOwned::to_owned)
}

struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Minimal blocking HTTP/1.1 JSON client for talking to a local
/// `llama-server` (the repo deliberately carries no HTTP client dependency;
/// wire code is hand-rolled here as in `distributed.rs`). Sends
/// `Connection: close` and handles both Content-Length and chunked bodies.
fn http_json(
    host: &str,
    port: u16,
    method: &str,
    path: &str,
    body: Option<&Value>,
    timeout: Duration,
) -> Result<(u16, Value), String> {
    let address = format!("{host}:{port}");
    let stream = TcpStream::connect(&address)
        .map_err(|err| format!("could not connect to {address}: {err}"))?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|err| err.to_string())?;
    stream
        .set_write_timeout(Some(Duration::from_secs(30)))
        .map_err(|err| err.to_string())?;
    let mut stream = stream;

    let body_bytes = body
        .map(|value| serde_json::to_vec(value).map_err(|err| err.to_string()))
        .transpose()?
        .unwrap_or_default();
    let mut request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {address}\r\nAccept: application/json\r\nConnection: close\r\n"
    );
    if !body_bytes.is_empty() {
        request.push_str("Content-Type: application/json\r\n");
        request.push_str(&format!("Content-Length: {}\r\n", body_bytes.len()));
    }
    request.push_str("\r\n");
    stream
        .write_all(request.as_bytes())
        .and_then(|()| stream.write_all(&body_bytes))
        .map_err(|err| format!("request write to {address} failed: {err}"))?;

    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .map_err(|err| format!("response read from {address} failed: {err}"))?;
    parse_http_response(&raw)
}

/// Parse a full HTTP/1.1 response (status, headers, body) read to EOF.
fn parse_http_response(raw: &[u8]) -> Result<(u16, Value), String> {
    let header_end = find_subsequence(raw, b"\r\n\r\n")
        .ok_or_else(|| "malformed HTTP response: missing header terminator".to_string())?;
    let head = String::from_utf8_lossy(&raw[..header_end]);
    let mut lines = head.split("\r\n");
    let status_line = lines.next().unwrap_or_default();
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .ok_or_else(|| format!("malformed HTTP status line: {status_line:?}"))?;

    let mut chunked = false;
    let mut content_length: Option<usize> = None;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let name = name.trim().to_ascii_lowercase();
        let value = value.trim();
        if name == "transfer-encoding" && value.to_ascii_lowercase().contains("chunked") {
            chunked = true;
        } else if name == "content-length" {
            content_length = value.parse().ok();
        }
    }

    let body_raw = &raw[header_end + 4..];
    let body = if chunked {
        decode_chunked(body_raw)?
    } else if let Some(length) = content_length {
        body_raw.get(..length).unwrap_or(body_raw).to_vec()
    } else {
        body_raw.to_vec()
    };
    if body.is_empty() {
        return Ok((status, Value::Null));
    }
    let value = serde_json::from_slice(&body)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&body).trim().to_string()));
    Ok((status, value))
}

fn decode_chunked(mut raw: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    loop {
        let line_end = find_subsequence(raw, b"\r\n")
            .ok_or_else(|| "malformed chunked body: missing size line".to_string())?;
        let size_line = String::from_utf8_lossy(&raw[..line_end]);
        let size_text = size_line.split(';').next().unwrap_or_default().trim();
        let size = usize::from_str_radix(size_text, 16)
            .map_err(|_| format!("malformed chunk size: {size_text:?}"))?;
        raw = &raw[line_end + 2..];
        if size == 0 {
            return Ok(out);
        }
        let chunk = raw
            .get(..size)
            .ok_or_else(|| "malformed chunked body: truncated chunk".to_string())?;
        out.extend_from_slice(chunk);
        raw = raw.get(size + 2..).unwrap_or(&[]);
    }
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_divergent_index_matches_harness_semantics() {
        assert_eq!(first_divergent_index(&[1, 2, 3], &[1, 2, 3]), NO_DIVERGENCE);
        assert_eq!(first_divergent_index(&[], &[]), NO_DIVERGENCE);
        assert_eq!(first_divergent_index(&[1, 2, 3], &[1, 9, 3]), 1);
        // Length mismatch diverges at the shorter length, like firstDifference().
        assert_eq!(first_divergent_index(&[1, 2], &[1, 2, 3]), 2);
        assert_eq!(first_divergent_index(&[1, 2, 3], &[1, 2]), 2);
    }

    #[test]
    fn parses_content_length_response() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 13\r\n\r\n{\"ok\":true}\r\n";
        let (status, value) = parse_http_response(raw).unwrap();
        assert_eq!(status, 200);
        assert_eq!(value["ok"], true);
    }

    #[test]
    fn parses_chunked_response() {
        let raw =
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\n{\"a\":\r\n5\r\ntrue}\r\n0\r\n\r\n";
        let (status, value) = parse_http_response(raw).unwrap();
        assert_eq!(status, 200);
        assert_eq!(value["a"], true);
    }

    #[test]
    fn parses_error_status_with_plain_body() {
        let raw = b"HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\n\r\nnot found";
        let (status, value) = parse_http_response(raw).unwrap();
        assert_eq!(status, 404);
        assert_eq!(value, Value::String("not found".to_string()));
    }

    #[test]
    fn rejects_malformed_chunked_body() {
        assert!(decode_chunked(b"zz\r\n").is_err());
        assert!(decode_chunked(b"5\r\nab").is_err());
    }

    use crate::receipt::{
        LaneIdentity, ParityBlock, QualityTier, ReceiptRequest, ReceiptResult, ReferenceIdentity,
        NO_DIVERGENCE, RECEIPT_SCHEMA_V1,
    };
    use serde_json::json;

    fn receipt_with_tier(tier: Option<QualityTier>) -> ParityReceipt {
        let mut r = ParityReceipt {
            schema: RECEIPT_SCHEMA_V1.to_string(),
            receipt_id: String::new(),
            created_utc: "2026-06-21T00:00:00Z".to_string(),
            lane: LaneIdentity {
                model_id: "m".to_string(),
                gguf_sha256: "ab".repeat(32),
                gguf_filename: "m.gguf".to_string(),
                quantization: "Q4_K".to_string(),
                architecture: "qwen3".to_string(),
                tokenizer_kind: "bpe".to_string(),
                tokenizer_sha256: None,
                camelid_version: "0.1.0".to_string(),
                camelid_commit: "deadbee".to_string(),
            },
            reference: ReferenceIdentity {
                tool: "llama.cpp".to_string(),
                binary: "llama-server".to_string(),
                version: Some("b4567".to_string()),
                commit: None,
            },
            request: ReceiptRequest {
                endpoint: "/v1/chat/completions".to_string(),
                messages_or_prompt: json!([{"role":"user","content":"hi"}]),
                max_tokens: 8,
                temperature: 0.0,
                top_p: None,
                top_k: None,
                seed: None,
                stop: vec![],
                response_format: None,
            },
            reproducible: true,
            result: ReceiptResult {
                prompt_token_ids: vec![1, 2],
                generated_token_ids: vec![3, 4],
                generated_text: "x".to_string(),
                completion_tokens: 2,
                finish_reason: "length".to_string(),
            },
            parity: ParityBlock {
                compared_against_reference: true,
                prompt_tokens_match: Some(true),
                generated_tokens_match: Some(true),
                generated_text_match: Some(true),
                first_divergent_token_index: Some(NO_DIVERGENCE),
            },
            execution_lane: None,
            execution_trace: None,
            quality_tier: tier,
            signature: None,
        };
        r.seal().expect("seal");
        r
    }

    #[test]
    fn format_honesty_passes_no_tier_and_headline() {
        // No tier ⇒ defer to the normal parity flow.
        let r = receipt_with_tier(None);
        assert_eq!(check_format_honesty(&r), Ok(None));
        // Headline-Q8 ⇒ also defers.
        let r = receipt_with_tier(Some(QualityTier::headline_q8(
            "llama.cpp".to_string(),
            "b4567".to_string(),
            100.0,
            0.0,
        )));
        assert_eq!(check_format_honesty(&r), Ok(None));
    }

    #[test]
    fn format_honesty_accepts_same_format_lossy() {
        // A lossy tier measured against its own format is honest ⇒ attestation.
        let r = receipt_with_tier(Some(QualityTier::lossy_same_format(
            "Q4_K".to_string(),
            "llama.cpp".to_string(),
            "b4567".to_string(),
            96.0,
            0.2,
        )));
        assert_eq!(check_format_honesty(&r), Ok(Some(())));
    }

    #[test]
    fn format_honesty_rejects_cross_format_lossy() {
        // A lossy tier that claims a DIFFERENT baseline format is dishonest.
        let mut tier = QualityTier::lossy_same_format(
            "Q4_K".to_string(),
            "llama.cpp".to_string(),
            "b4567".to_string(),
            99.0,
            0.05,
        );
        tier.baseline_format = "Q8_0".to_string(); // cross-format token match
        let r = receipt_with_tier(Some(tier));
        assert!(
            check_format_honesty(&r).is_err(),
            "cross-format lossy tier must be rejected"
        );
    }

    #[test]
    fn format_honesty_rejects_unknown_tier_schema() {
        let mut tier = QualityTier::lossy_same_format(
            "Q4_K".to_string(),
            "llama.cpp".to_string(),
            "b4567".to_string(),
            99.0,
            0.05,
        );
        tier.schema = "camelid.quality-tier/v999".to_string();
        let r = receipt_with_tier(Some(tier));
        assert!(check_format_honesty(&r).is_err());
    }

    #[test]
    fn lossy_attestation_exit_code_is_distinct() {
        // Distinct from VERIFIED(0), NOT VERIFIED(1), and divergence(3).
        assert_eq!(VerifyOutcome::LossyAttestation.exit_code(), 4);
        assert_eq!(VerifyOutcome::Verified.exit_code(), 0);
        assert_eq!(VerifyOutcome::NotVerified.exit_code(), 1);
        assert_eq!(VerifyOutcome::DivergenceRecord.exit_code(), 3);
    }
}
