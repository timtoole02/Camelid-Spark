//! The parity-gated calibration tournament (Lane B).
//!
//! On first encounter of a `gait_key`, calibration discovers the fastest
//! execution configuration for *this* model on *this* machine, then persists it.
//! The engine here owns the hard part — extracting the model's real stage
//! shapes, running candidates through a **parity gate** (a faster candidate that
//! diverges is disqualified, full stop), picking a winner only if it beats
//! baseline by a margin, computing the roofline %, and failing closed to the
//! proven baseline otherwise.
//!
//! The act of *timing real decode* is an injected seam (`trial`): production
//! passes a closure that runs the engine under a candidate's configuration and
//! returns its tok/s plus a parity token (the greedy-output digest, matching the
//! `qa/speed` methodology); tests pass deterministic stubs. This keeps the
//! selection logic fully testable without ever fabricating a throughput number.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::{gait_dir, store_in, GaitReceipt, MachineSig, MemoryMeasurement, ModelSig};
use crate::execution_plan::ExecutionProfile;
use crate::gguf::GgufFile;

/// A representative tensor shape for one inference stage class — the geometry a
/// candidate kernel is timed against. Calibration tunes on the model's *actual*
/// shapes, not synthetic ones.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageShape {
    pub stage: String,
    pub dims: Vec<u64>,
    pub quant: String,
}

/// One representative shape per matmul stage class present in the model.
pub fn stage_shapes(gguf: &GgufFile) -> Vec<StageShape> {
    let mut by_class: BTreeMap<&'static str, StageShape> = BTreeMap::new();
    for tensor in &gguf.tensors {
        if tensor.dimensions.len() != 2 {
            continue; // GEMV/GEMM stages are rank-2 weights.
        }
        let class = super::tensor_class(&tensor.name);
        if !is_matmul_stage(class) {
            continue;
        }
        by_class.entry(class).or_insert_with(|| StageShape {
            stage: class.to_string(),
            dims: tensor.dimensions.clone(),
            quant: format!("{:?}", tensor.tensor_type),
        });
    }
    by_class.into_values().collect()
}

/// Total quantized weight bytes read per decoded token — the numerator of the
/// roofline ratio (achieved bytes/s ÷ measured DRAM bytes/s). Sums the matmul
/// weight tensors a decode step streams once.
pub fn decode_weight_bytes(gguf: &GgufFile) -> u64 {
    gguf.tensors
        .iter()
        .filter(|t| is_matmul_stage(super::tensor_class(&t.name)))
        .map(|t| t.n_bytes)
        .sum()
}

fn is_matmul_stage(class: &str) -> bool {
    matches!(
        class,
        "attn_q"
            | "attn_k"
            | "attn_v"
            | "attn_qkv"
            | "attn_output"
            | "ffn_gate"
            | "ffn_up"
            | "ffn_down"
            | "output"
    )
}

/// §5: the three managed x86 Q8 `groups_per_chunk` tiling knobs
/// (`MANAGED_ENV_KEYS`). Tiling only — the arithmetic is unchanged, so varying
/// these is parity-neutral *by construction* (and the tournament's parity gate
/// enforces it regardless). The defaults match the kernel readers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupsPerChunk {
    pub attn_qkv_decode: usize,
    pub ffn_gate_up_decode: usize,
    pub packed_rows4_matmul: usize,
}

/// The kernel defaults — the center of the bounded search.
pub const GPC_CENTER: GroupsPerChunk = GroupsPerChunk {
    attn_qkv_decode: 16,
    ffn_gate_up_decode: 16,
    packed_rows4_matmul: 8,
};

/// A bounded coordinate-neighbor set around [`GPC_CENTER`] — a *short local
/// search*, not a full grid, so the (expensive, child-process) trial count stays
/// small. Each point keeps all knobs `> 0` as the kernel readers require.
pub fn groups_per_chunk_neighbors() -> Vec<GroupsPerChunk> {
    vec![
        GPC_CENTER,
        GroupsPerChunk {
            attn_qkv_decode: 8,
            ffn_gate_up_decode: 8,
            ..GPC_CENTER
        },
        GroupsPerChunk {
            attn_qkv_decode: 32,
            ffn_gate_up_decode: 32,
            ..GPC_CENTER
        },
        GroupsPerChunk {
            packed_rows4_matmul: 16,
            ..GPC_CENTER
        },
    ]
}

/// A configuration to evaluate. `profile` is what gets recorded and later
/// applied; `label` is for evidence/logs.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub label: String,
    pub profile: ExecutionProfile,
    /// Whether this candidate disables EcoQoS execution-speed throttling (a
    /// Windows scheduling-substrate dimension). The engine only records it; the
    /// trial closure applies it before timing.
    pub eco_qos_opt_out: bool,
    /// §5: per-stage `groups_per_chunk` tiling overrides. `None` uses the
    /// profile's kernel defaults; `Some` is applied (and recorded) by the trial.
    pub groups_per_chunk: Option<GroupsPerChunk>,
}

/// The result of timing one candidate: its throughput and a parity token. Two
/// candidates are parity-equal iff their tokens are equal — the gate that
/// disqualifies a fast-but-divergent candidate. Serializable so a supervised
/// child-process trial (§1.4 crash isolation) can hand it back over stdout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialResult {
    pub tokens_per_s: f64,
    pub parity_token: String,
}

/// Tournament knobs.
#[derive(Debug, Clone, Copy)]
pub struct TournamentConfig {
    /// A candidate must beat baseline by at least this factor to win; otherwise
    /// the tournament fails closed to baseline. Slower-but-correct always wins.
    /// The default (1.10) sits above the per-trial variance of the §1.4
    /// child-process trials, so a sub-10% median gap — indistinguishable from
    /// measurement noise on this execution model — never persists a gait.
    pub min_speedup: f64,
    /// Measured rounds per variant. Each round times every variant once, in the
    /// same order, so any two share a thermal/clock neighborhood (matched-clock
    /// A/B by interleaving). The per-variant statistic is the median across
    /// rounds, which rejects the run-to-run outliers a single trial cannot.
    pub rounds: usize,
    /// Leading rounds discarded as warmup (cache/clock settling).
    pub warmup_rounds: usize,
}

impl Default for TournamentConfig {
    fn default() -> Self {
        Self {
            min_speedup: 1.10,
            rounds: 5,
            warmup_rounds: 1,
        }
    }
}

/// Per-variant measurement spread, recorded as honest evidence so a reader can
/// see the noise behind the selection rather than just a single number.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateSamples {
    pub label: String,
    pub median_tokens_per_s: f64,
    pub min_tokens_per_s: f64,
    pub max_tokens_per_s: f64,
    pub measured_rounds: usize,
    /// True when every round reproduced this variant's parity token AND it
    /// matched the baseline's (eligible); false means it was disqualified.
    pub parity_ok: bool,
}

/// The evidence a calibration produced — recorded into the gait receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationOutcome {
    pub selected_profile: ExecutionProfile,
    pub reason: String,
    pub baseline_tokens_per_s: f64,
    pub selected_tokens_per_s: f64,
    pub speedup: f64,
    pub roofline_pct: f64,
    /// True when no candidate qualified and the proven baseline was kept.
    pub fell_back: bool,
    /// Candidate labels disqualified for diverging from baseline parity.
    pub parity_disqualified: Vec<String>,
    /// Whether the selected configuration disables EcoQoS throttling. Recorded so
    /// the gait can be reproduced. Defaults to false for receipts written before
    /// the substrate dimension existed.
    #[serde(default)]
    pub selected_eco_qos_opt_out: bool,
    /// §5: the `groups_per_chunk` tiling the winner used (`None` = kernel defaults
    /// / baseline). Recorded so the selected gait reproduces the chosen tiling.
    #[serde(default)]
    pub selected_groups_per_chunk: Option<GroupsPerChunk>,
    /// Measured rounds per variant behind the medians.
    #[serde(default)]
    pub measured_rounds: usize,
    /// Per-variant measurement spread (baseline first), for evidence.
    #[serde(default)]
    pub samples: Vec<CandidateSamples>,
}

fn roofline_pct(tokens_per_s: f64, weight_bytes_per_token: u64, stream_gbs: f64) -> f64 {
    if stream_gbs <= 0.0 || weight_bytes_per_token == 0 {
        return 0.0;
    }
    (tokens_per_s * weight_bytes_per_token as f64) / (stream_gbs * 1e9)
}

fn median(samples: &[f64]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    Some(if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    })
}

/// An outlier-robust *low* estimate: the worst round after discarding a single
/// unlucky outlier (the median's companion for the downside). For `>= 2` samples
/// this is the second-smallest; for one sample it is that sample. Used by the
/// significance gate so a single bad round does not veto a genuine win, while a
/// broadly-noisy candidate (several rounds below baseline) is still caught.
fn robust_low(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if sorted.len() >= 2 {
        sorted[1]
    } else {
        sorted.first().copied().unwrap_or(0.0)
    }
}

/// Per-variant accumulation across interleaved rounds.
struct VariantStats {
    tok_samples: Vec<f64>,
    parity_token: Option<String>,
    parity_consistent: bool,
}

/// Run the tournament with interleaved, matched-clock rounds. Each round times
/// every variant once in a fixed order (baseline first), so any two are measured
/// in the same thermal/clock neighborhood; the per-variant statistic is the
/// median across measured rounds. `trial` times one variant and returns its
/// throughput + parity token (or `None` if it could not be run that round).
///
/// A candidate is eligible only if it reproduced its own parity token across all
/// rounds AND that token equals the baseline's. The fastest eligible candidate
/// whose median clears `min_speedup` wins; otherwise the tournament fails closed
/// to baseline. Slower-but-correct always wins.
pub fn run_tournament(
    baseline: &Candidate,
    candidates: &[Candidate],
    config: &TournamentConfig,
    weight_bytes_per_token: u64,
    memory: &MemoryMeasurement,
    mut trial: impl FnMut(&Candidate) -> Option<TrialResult>,
) -> CalibrationOutcome {
    let gbs = memory.stream_triad_gbs;

    // Index 0 is the baseline; the rest are candidates, measured interleaved.
    let variants: Vec<&Candidate> = std::iter::once(baseline).chain(candidates).collect();
    let mut stats: Vec<VariantStats> = variants
        .iter()
        .map(|_| VariantStats {
            tok_samples: Vec::new(),
            parity_token: None,
            parity_consistent: true,
        })
        .collect();

    let measured = config.rounds.max(1);
    let total_rounds = config.warmup_rounds + measured;
    for round in 0..total_rounds {
        for (i, variant) in variants.iter().enumerate() {
            let Some(result) = trial(variant) else {
                continue;
            };
            match &stats[i].parity_token {
                None => stats[i].parity_token = Some(result.parity_token.clone()),
                Some(token) if *token != result.parity_token => {
                    stats[i].parity_consistent = false;
                }
                _ => {}
            }
            if round >= config.warmup_rounds && result.tokens_per_s > 0.0 {
                stats[i].tok_samples.push(result.tokens_per_s);
            }
        }
    }

    let base_parity = stats[0].parity_token.clone();
    let base_median = median(&stats[0].tok_samples);

    // Evidence: per-variant spread, baseline first.
    let samples: Vec<CandidateSamples> = variants
        .iter()
        .zip(stats.iter())
        .map(|(variant, stat)| {
            let med = median(&stat.tok_samples).unwrap_or(0.0);
            let min = stat
                .tok_samples
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            let max = stat
                .tok_samples
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let parity_ok = stat.parity_consistent && stat.parity_token == base_parity;
            CandidateSamples {
                label: variant.label.clone(),
                median_tokens_per_s: super::round_sig6(med),
                min_tokens_per_s: super::round_sig6(if min.is_finite() { min } else { 0.0 }),
                max_tokens_per_s: super::round_sig6(if max.is_finite() { max } else { 0.0 }),
                measured_rounds: stat.tok_samples.len(),
                parity_ok,
            }
        })
        .collect();

    let fail_closed = |reason: String, base_tok: f64, disq: Vec<String>| CalibrationOutcome {
        selected_profile: baseline.profile.clone(),
        reason,
        baseline_tokens_per_s: super::round_sig6(base_tok),
        selected_tokens_per_s: super::round_sig6(base_tok),
        speedup: 1.0,
        roofline_pct: super::round_sig6(roofline_pct(base_tok, weight_bytes_per_token, gbs)),
        fell_back: true,
        parity_disqualified: disq,
        selected_eco_qos_opt_out: baseline.eco_qos_opt_out,
        selected_groups_per_chunk: baseline.groups_per_chunk,
        measured_rounds: measured,
        samples: samples.clone(),
    };

    let base_tok = match base_median {
        Some(t) if t > 0.0 => t,
        _ => {
            return fail_closed(
                "baseline trial failed; failing closed to baseline".to_string(),
                0.0,
                Vec::new(),
            );
        }
    };

    // Evaluate candidates (indices 1..). A candidate wins only if it clears BOTH
    // gates: (a) MARGIN — its median beats the baseline median by `min_speedup`;
    // and (b) SIGNIFICANCE — even after discarding a single unlucky round, its
    // worst remaining round still beats the baseline median (`robust_low`).
    //
    // The significance gate exists because the §1.4 child-process trials carry
    // wide per-trial variance (each trial is a fresh process: fresh load, fresh
    // page-cache and scheduler placement). A margin-only gate would crown a noise
    // winner on a lucky median and persist it. Requiring the candidate to stay
    // above baseline even on its near-worst round rejects that, while the
    // outlier-robust `robust_low` still admits a real win that had one bad round.
    let mut best: Option<(&Candidate, f64)> = None;
    let mut disqualified = Vec::new();
    let mut margin_but_noisy = false;
    for (i, candidate) in candidates.iter().enumerate() {
        let stat = &stats[i + 1];
        if stat.tok_samples.is_empty() {
            continue; // never ran — skip silently
        }
        let eligible = stat.parity_consistent && stat.parity_token == base_parity;
        if !eligible {
            disqualified.push(candidate.label.clone()); // PARITY GATE
            continue;
        }
        let med = median(&stat.tok_samples).unwrap_or(0.0);
        if med < base_tok * config.min_speedup {
            continue; // MARGIN GATE
        }
        if robust_low(&stat.tok_samples) < base_tok {
            margin_but_noisy = true; // SIGNIFICANCE GATE — the win is within noise
            continue;
        }
        match best {
            Some((_, best_tok)) if med <= best_tok => {}
            _ => best = Some((candidate, med)),
        }
    }

    match best {
        Some((winner, tok)) => {
            let speedup = tok / base_tok;
            CalibrationOutcome {
                selected_profile: winner.profile.clone(),
                reason: format!(
                    "gait: {} won at {speedup:.3}x over baseline (median of {measured} rounds, noise-significant)",
                    winner.label
                ),
                baseline_tokens_per_s: super::round_sig6(base_tok),
                selected_tokens_per_s: super::round_sig6(tok),
                speedup: super::round_sig6(speedup),
                roofline_pct: super::round_sig6(roofline_pct(tok, weight_bytes_per_token, gbs)),
                fell_back: false,
                parity_disqualified: disqualified,
                selected_eco_qos_opt_out: winner.eco_qos_opt_out,
                selected_groups_per_chunk: winner.groups_per_chunk,
                measured_rounds: measured,
                samples,
            }
        }
        None => {
            let reason = if margin_but_noisy {
                "a candidate beat the margin but not the measurement noise (a round dipped below baseline); keeping baseline"
                    .to_string()
            } else {
                "no parity-clean candidate beat baseline by margin; keeping baseline".to_string()
            };
            fail_closed(reason, base_tok, disqualified)
        }
    }
}

/// End-to-end: fingerprint the model + machine, measure memory, run the
/// tournament, and persist a sealed gait receipt under `dir`. Returns the
/// outcome and the receipt path (`None` if the store write failed — fail-closed,
/// never panics). The caller supplies the timing `trial`.
pub fn calibrate_and_store(
    dir: &Path,
    gguf: &GgufFile,
    baseline: &Candidate,
    candidates: &[Candidate],
    config: &TournamentConfig,
    trial: impl FnMut(&Candidate) -> Option<TrialResult>,
) -> (CalibrationOutcome, Option<PathBuf>) {
    let model_sig = ModelSig::from_gguf(gguf);
    let machine_sig = MachineSig::detect();
    let memory = super::measure_memory();
    let weight_bytes = decode_weight_bytes(gguf);

    let outcome = run_tournament(baseline, candidates, config, weight_bytes, &memory, trial);

    // §6F: attest the host-safety posture (the §1.2 cap + §1.1/§1.2 invariants) and
    // record the measured free-RAM headroom, so the receipt audits how this gait
    // runs. Read physical_cores before machine_sig is moved into the receipt.
    let scheduling =
        super::Scheduling::attest(machine_sig.physical_cores, outcome.selected_eco_qos_opt_out);
    let host_safety = super::HostSafety::capture();

    let receipt = GaitReceipt::new(model_sig, machine_sig, outcome.selected_profile.clone())
        .with_memory(memory)
        .with_calibration(outcome.clone())
        .with_scheduling(scheduling)
        .with_host_safety(host_safety);
    let path = store_in(dir, &receipt).ok();
    (outcome, path)
}

/// Convenience: the default store directory for calibration output.
pub fn default_store_dir() -> Option<PathBuf> {
    gait_dir()
}

/// Outcome of supervising a child-process trial under a hard timeout (§1.4).
#[derive(Debug)]
pub enum WatchdogOutcome {
    /// The child exited on its own; carries its captured output.
    Completed(std::process::Output),
    /// The child overran the timeout and was killed — abandoned and disqualified.
    TimedOut,
    /// Could not wait on / collect the child (OS error).
    Errored,
}

/// Supervise a child trial under a HARD `timeout`, polling every `poll` interval.
///
/// This is the §1.4 crash-isolation valve: candidate kernels run in a separate
/// process, so a candidate that **segfaults** cannot take down the calibrating
/// (or serving) process — it surfaces as a non-success exit — and a candidate
/// that **hangs** is KILLED at the deadline rather than waited on forever. Either
/// way the supervisor returns and the bad candidate is disqualified upstream;
/// it is never persisted. The child should be spawned with stdout piped if the
/// caller needs the [`TrialResult`] it prints.
pub fn supervise(
    mut child: std::process::Child,
    timeout: std::time::Duration,
    poll: std::time::Duration,
) -> WatchdogOutcome {
    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_status)) => {
                return match child.wait_with_output() {
                    Ok(output) => WatchdogOutcome::Completed(output),
                    Err(_) => WatchdogOutcome::Errored,
                };
            }
            Ok(None) => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return WatchdogOutcome::TimedOut;
                }
                std::thread::sleep(poll);
            }
            Err(_) => return WatchdogOutcome::Errored,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgufFile, GgufMetadataValue, GgufTensorDescriptor, GgufTensorType};
    use std::collections::BTreeMap as Map;
    use std::path::PathBuf;

    fn descriptor(
        name: &str,
        ty: GgufTensorType,
        dims: Vec<u64>,
        n_bytes: u64,
    ) -> GgufTensorDescriptor {
        GgufTensorDescriptor {
            name: name.to_string(),
            dimensions: dims,
            tensor_type: ty,
            relative_offset: 0,
            absolute_offset: 0,
            n_bytes,
        }
    }

    fn sample_gguf() -> GgufFile {
        let mut metadata = Map::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufMetadataValue::String("llama".to_string()),
        );
        GgufFile {
            path: PathBuf::from("sample.gguf"),
            version: 3,
            tensor_count: 4,
            metadata_count: metadata.len() as i64,
            alignment: 32,
            data_start_offset: 0,
            metadata,
            tensors: vec![
                descriptor(
                    "token_embd.weight",
                    GgufTensorType::Q8_0,
                    vec![16, 128],
                    4096,
                ),
                descriptor(
                    "blk.0.attn_q.weight",
                    GgufTensorType::Q8_0,
                    vec![16, 16],
                    256,
                ),
                descriptor(
                    "blk.0.ffn_down.weight",
                    GgufTensorType::Q4K,
                    vec![32, 16],
                    288,
                ),
                descriptor("blk.0.attn_norm.weight", GgufTensorType::F32, vec![16], 64),
            ],
        }
    }

    fn cand(label: &str, profile: ExecutionProfile) -> Candidate {
        Candidate {
            label: label.to_string(),
            profile,
            eco_qos_opt_out: false,
            groups_per_chunk: None,
        }
    }

    #[test]
    fn gpc_neighbors_are_bounded_distinct_and_positive() {
        let neighbors = groups_per_chunk_neighbors();
        // A short local search, never a runaway grid.
        assert!(
            (2..=8).contains(&neighbors.len()),
            "search must stay bounded, got {}",
            neighbors.len()
        );
        assert!(neighbors.contains(&GPC_CENTER), "the center must be probed");
        let mut seen = std::collections::BTreeSet::new();
        for g in &neighbors {
            // Every knob must stay > 0 (the kernel readers reject 0).
            assert!(g.attn_qkv_decode > 0 && g.ffn_gate_up_decode > 0 && g.packed_rows4_matmul > 0);
            assert!(
                seen.insert((
                    g.attn_qkv_decode,
                    g.ffn_gate_up_decode,
                    g.packed_rows4_matmul
                )),
                "search points must be distinct"
            );
        }
    }

    fn mem() -> MemoryMeasurement {
        MemoryMeasurement {
            stream_triad_gbs: 30.0,
            dram_latency_ns: 80.0,
        }
    }

    #[test]
    fn stage_shapes_are_matmul_only_and_deduped() {
        let shapes = stage_shapes(&sample_gguf());
        let stages: Vec<&str> = shapes.iter().map(|s| s.stage.as_str()).collect();
        // attn_q and ffn_down are matmul stages; token_embd and norm are not.
        assert!(stages.contains(&"attn_q"));
        assert!(stages.contains(&"ffn_down"));
        assert!(!stages.contains(&"token_embd"));
        assert!(!stages.contains(&"norm"));
    }

    #[test]
    fn decode_weight_bytes_sums_matmul_weights_only() {
        // attn_q (256) + ffn_down (288); token_embd and norm excluded.
        assert_eq!(decode_weight_bytes(&sample_gguf()), 256 + 288);
    }

    #[test]
    fn parity_gate_beats_raw_speed() {
        let baseline = cand("auto", ExecutionProfile::Auto);
        let candidates = vec![
            cand("fast_clean", ExecutionProfile::Experimental),
            cand("faster_diverges", ExecutionProfile::Debug),
            cand("slow_clean", ExecutionProfile::Safe),
        ];
        let outcome = run_tournament(
            &baseline,
            &candidates,
            &TournamentConfig::default(),
            1_000_000,
            &mem(),
            |c| {
                Some(match c.label.as_str() {
                    "auto" => TrialResult {
                        tokens_per_s: 10.0,
                        parity_token: "A".into(),
                    },
                    "fast_clean" => TrialResult {
                        tokens_per_s: 13.0,
                        parity_token: "A".into(),
                    },
                    // Faster, but DIVERGES — must be disqualified despite winning on speed.
                    "faster_diverges" => TrialResult {
                        tokens_per_s: 20.0,
                        parity_token: "B".into(),
                    },
                    "slow_clean" => TrialResult {
                        tokens_per_s: 9.0,
                        parity_token: "A".into(),
                    },
                    _ => unreachable!(),
                })
            },
        );
        assert!(!outcome.fell_back);
        assert_eq!(outcome.selected_profile, ExecutionProfile::Experimental);
        assert!((outcome.speedup - 1.3).abs() < 1e-9);
        assert!(outcome
            .parity_disqualified
            .contains(&"faster_diverges".to_string()));
        assert!(outcome.roofline_pct > 0.0);
    }

    #[test]
    fn median_rejects_single_round_outlier() {
        // The candidate is genuinely faster (median 13) but one round spikes low.
        // A single back-to-back trial that happened to hit the 4 would miss it;
        // the interleaved median does not.
        let baseline = cand("auto", ExecutionProfile::Auto);
        let candidates = vec![cand("fast", ExecutionProfile::Experimental)];
        let cfg = TournamentConfig {
            min_speedup: 1.05,
            rounds: 3,
            warmup_rounds: 0,
        };
        let mut fast_calls = 0;
        let outcome = run_tournament(&baseline, &candidates, &cfg, 1_000_000, &mem(), |c| match c
            .label
            .as_str()
        {
            "auto" => Some(TrialResult {
                tokens_per_s: 10.0,
                parity_token: "A".into(),
            }),
            _ => {
                fast_calls += 1;
                let tps = if fast_calls == 2 { 4.0 } else { 13.0 };
                Some(TrialResult {
                    tokens_per_s: tps,
                    parity_token: "A".into(),
                })
            }
        });
        assert!(!outcome.fell_back);
        assert_eq!(outcome.selected_profile, ExecutionProfile::Experimental);
        assert_eq!(outcome.measured_rounds, 3);
        assert_eq!(outcome.samples.len(), 2); // baseline + candidate
        let fast = outcome.samples.iter().find(|s| s.label == "fast").unwrap();
        assert_eq!(fast.median_tokens_per_s, 13.0);
        assert_eq!(fast.min_tokens_per_s, 4.0);
    }

    #[test]
    fn noisy_candidate_rejected_despite_high_median() {
        // High median (14 > baseline 10, clears even a 1.05 margin) but BROADLY
        // noisy: multiple rounds dip below the baseline median, so the median is a
        // lucky-rounds artifact, not a real win. This is the §5 child-process
        // noise case — the significance gate (not the margin) must fail-close.
        let baseline = cand("auto", ExecutionProfile::Auto);
        let candidates = vec![cand("noisy", ExecutionProfile::Experimental)];
        let cfg = TournamentConfig {
            min_speedup: 1.05,
            rounds: 5,
            warmup_rounds: 0,
        };
        // sorted [8,9,14,14,14] -> median 14, robust_low (2nd-smallest) 9 < 10.
        let noisy = [14.0, 9.0, 14.0, 8.0, 14.0];
        let mut n = 0;
        let outcome = run_tournament(&baseline, &candidates, &cfg, 1_000_000, &mem(), |c| match c
            .label
            .as_str()
        {
            "auto" => Some(TrialResult {
                tokens_per_s: 10.0,
                parity_token: "A".into(),
            }),
            _ => {
                let v = noisy[n % noisy.len()];
                n += 1;
                Some(TrialResult {
                    tokens_per_s: v,
                    parity_token: "A".into(),
                })
            }
        });
        assert!(
            outcome.fell_back,
            "a broadly-noisy candidate must fail-close"
        );
        assert_eq!(outcome.selected_profile, ExecutionProfile::Auto);
        assert!(
            outcome.reason.contains("measurement noise"),
            "reason should name the noise gate, got: {}",
            outcome.reason
        );
    }

    #[test]
    fn clean_win_survives_significance() {
        // A stable, genuinely-faster candidate: every round clears the baseline
        // median, even discounting the slowest. Both gates pass — it wins.
        let baseline = cand("auto", ExecutionProfile::Auto);
        let candidates = vec![cand("fast", ExecutionProfile::Experimental)];
        let cfg = TournamentConfig {
            min_speedup: 1.10,
            rounds: 5,
            warmup_rounds: 0,
        };
        let fast = [12.0, 12.0, 13.0, 12.0, 12.0]; // median 12 = 1.2x, robust_low 12
        let mut n = 0;
        let outcome = run_tournament(&baseline, &candidates, &cfg, 1_000_000, &mem(), |c| match c
            .label
            .as_str()
        {
            "auto" => Some(TrialResult {
                tokens_per_s: 10.0,
                parity_token: "A".into(),
            }),
            _ => {
                let v = fast[n % fast.len()];
                n += 1;
                Some(TrialResult {
                    tokens_per_s: v,
                    parity_token: "A".into(),
                })
            }
        });
        assert!(!outcome.fell_back);
        assert_eq!(outcome.selected_profile, ExecutionProfile::Experimental);
    }

    #[test]
    fn default_margin_rejects_small_stable_win() {
        // A clean, stable but SMALL win (1.06x) — below the noise-robust default
        // margin (1.10). It passes significance but not the margin, so it must
        // fail-close: a sub-10% gait is not worth persisting under child-process
        // measurement variance.
        let baseline = cand("auto", ExecutionProfile::Auto);
        let candidates = vec![cand("slightly_faster", ExecutionProfile::Experimental)];
        let outcome = run_tournament(
            &baseline,
            &candidates,
            &TournamentConfig::default(),
            1_000_000,
            &mem(),
            |c| {
                Some(match c.label.as_str() {
                    "auto" => TrialResult {
                        tokens_per_s: 10.0,
                        parity_token: "A".into(),
                    },
                    _ => TrialResult {
                        tokens_per_s: 10.6,
                        parity_token: "A".into(),
                    },
                })
            },
        );
        assert!(
            outcome.fell_back,
            "a sub-margin win must fail-close under the default 1.10 margin"
        );
        assert_eq!(outcome.selected_profile, ExecutionProfile::Auto);
    }

    #[test]
    fn inconsistent_parity_across_rounds_disqualifies() {
        // A fast candidate whose output flips between rounds is non-deterministic
        // and must be disqualified, not selected.
        let baseline = cand("auto", ExecutionProfile::Auto);
        let candidates = vec![cand("flaky", ExecutionProfile::Experimental)];
        let cfg = TournamentConfig {
            min_speedup: 1.05,
            rounds: 2,
            warmup_rounds: 0,
        };
        let mut n = 0;
        let outcome = run_tournament(&baseline, &candidates, &cfg, 0, &mem(), |c| {
            match c.label.as_str() {
                "auto" => Some(TrialResult {
                    tokens_per_s: 10.0,
                    parity_token: "A".into(),
                }),
                _ => {
                    n += 1;
                    let token = if n == 1 { "A" } else { "B" };
                    Some(TrialResult {
                        tokens_per_s: 20.0,
                        parity_token: token.into(),
                    })
                }
            }
        });
        assert!(outcome.fell_back);
        assert!(outcome.parity_disqualified.contains(&"flaky".to_string()));
    }

    #[test]
    fn records_selected_substrate_dimension() {
        let baseline = cand("auto", ExecutionProfile::Auto);
        let mut winner = cand("auto+ecoqos", ExecutionProfile::Auto);
        winner.eco_qos_opt_out = true;
        let outcome = run_tournament(
            &baseline,
            &[winner],
            &TournamentConfig::default(),
            1_000_000,
            &mem(),
            |c| {
                Some(match c.label.as_str() {
                    "auto" => TrialResult {
                        tokens_per_s: 10.0,
                        parity_token: "A".into(),
                    },
                    _ => TrialResult {
                        tokens_per_s: 12.0,
                        parity_token: "A".into(),
                    },
                })
            },
        );
        assert!(!outcome.fell_back);
        assert!(outcome.selected_eco_qos_opt_out);
    }

    #[test]
    fn fails_closed_when_nothing_beats_margin() {
        let baseline = cand("auto", ExecutionProfile::Auto);
        let candidates = vec![cand("marginal", ExecutionProfile::Experimental)];
        let outcome = run_tournament(
            &baseline,
            &candidates,
            &TournamentConfig::default(),
            1_000_000,
            &mem(),
            |c| {
                Some(match c.label.as_str() {
                    "auto" => TrialResult {
                        tokens_per_s: 10.0,
                        parity_token: "A".into(),
                    },
                    // Only +2%, below the 5% margin.
                    _ => TrialResult {
                        tokens_per_s: 10.2,
                        parity_token: "A".into(),
                    },
                })
            },
        );
        assert!(outcome.fell_back);
        assert_eq!(outcome.selected_profile, ExecutionProfile::Auto);
        assert_eq!(outcome.speedup, 1.0);
    }

    #[test]
    fn fails_closed_when_baseline_trial_fails() {
        let baseline = cand("auto", ExecutionProfile::Auto);
        let outcome = run_tournament(
            &baseline,
            &[cand("x", ExecutionProfile::Experimental)],
            &TournamentConfig::default(),
            0,
            &mem(),
            |_| None,
        );
        assert!(outcome.fell_back);
        assert_eq!(outcome.selected_profile, ExecutionProfile::Auto);
    }

    #[test]
    fn calibrate_and_store_persists_a_readable_receipt() {
        let dir = std::env::temp_dir().join(format!("camelid_gait_calib_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let baseline = cand("auto", ExecutionProfile::Auto);
        let candidates = vec![cand("fast_clean", ExecutionProfile::Experimental)];
        let (outcome, path) = calibrate_and_store(
            &dir,
            &sample_gguf(),
            &baseline,
            &candidates,
            &TournamentConfig::default(),
            |c| {
                Some(match c.label.as_str() {
                    "auto" => TrialResult {
                        tokens_per_s: 10.0,
                        parity_token: "A".into(),
                    },
                    _ => TrialResult {
                        tokens_per_s: 14.0,
                        parity_token: "A".into(),
                    },
                })
            },
        );
        assert_eq!(outcome.selected_profile, ExecutionProfile::Experimental);
        let path = path.expect("receipt stored");
        assert!(path.exists());

        // The stored receipt round-trips and carries the calibration evidence.
        let text = std::fs::read_to_string(&path).unwrap();
        let receipt: GaitReceipt = serde_json::from_str(&text).unwrap();
        assert!(receipt.verify_self_digest());
        assert_eq!(receipt.recorded_profile, ExecutionProfile::Experimental);
        assert_eq!(
            receipt.calibration.unwrap().selected_profile,
            ExecutionProfile::Experimental
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
