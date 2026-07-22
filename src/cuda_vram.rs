//! VRAM headroom policy + contention measurement (Task 4).
//!
//! Two concerns, deliberately split so the *policy* is testable without a GPU:
//!
//! 1. **Headroom policy** ([`evaluate`]) — a pure function over byte counts. At
//!    load time the caller queries free VRAM and the projected allocation, then
//!    asks this function whether the allocation is safe. It refuses (returns
//!    [`VramShortfall`]) when the allocation would not fit (mid-load OOM) *or*
//!    would leave less than the configured minimum post-load headroom — naming
//!    the shortfall in MiB. No CUDA needed; fully unit-tested on any host.
//!
//! 2. **Contention measurement** ([`measure_contention`], `cfg(feature="cuda")`)
//!    — occupies one allocation the size of a resident model, then attempts a
//!    second allocation on the same device, recording whether the second fails
//!    *cleanly* (the driver returns `CUDA_ERROR_OUT_OF_MEMORY`) or the process
//!    OOMs. Runs N times and reports the median + variance, per the project's
//!    measurement discipline. Built-and-gated; must be run on a CUDA host.
//!
//! The Windows dev box can run all of (1) and type-check (2); the contention
//! numbers must be captured on the RTX-class CUDA host (see the findings doc
//! `qa/cuda/CONTENTION_FINDINGS.md`).

use std::fmt;

const MIB: u64 = 1024 * 1024;

/// Default minimum post-load free VRAM to keep, in MiB. Sized so loading a 3B+
/// model does not claim the last slice of memory and trigger a later OOM in the
/// KV cache / scratch / a co-resident engine. Override with
/// `CAMELID_MIN_VRAM_HEADROOM_MIB`.
pub const DEFAULT_MIN_HEADROOM_MIB: u64 = 512;

/// The configured minimum headroom (env override, else [`DEFAULT_MIN_HEADROOM_MIB`]).
pub fn min_headroom_mib() -> u64 {
    std::env::var("CAMELID_MIN_VRAM_HEADROOM_MIB")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_MIN_HEADROOM_MIB)
}

/// An allocation the headroom policy approved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VramPlan {
    pub free_mib: u64,
    pub alloc_mib: u64,
    pub projected_free_mib: u64,
    pub min_headroom_mib: u64,
}

/// The headroom policy refused an allocation, with the numbers that explain why.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VramShortfall {
    pub free_mib: u64,
    pub alloc_mib: u64,
    /// Free VRAM that would remain after the allocation (0 if it would not fit).
    pub projected_free_mib: u64,
    pub min_headroom_mib: u64,
    /// How far below the policy the allocation lands, in MiB.
    pub short_by_mib: u64,
    /// True when the allocation alone exceeds free VRAM (a mid-load OOM).
    pub would_oom: bool,
}

impl fmt::Display for VramShortfall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.would_oom {
            write!(
                f,
                "allocation {} MiB exceeds free VRAM {} MiB (would OOM mid-load); short by {} MiB \
                 including the {} MiB min headroom",
                self.alloc_mib, self.free_mib, self.short_by_mib, self.min_headroom_mib
            )
        } else {
            write!(
                f,
                "allocation {} MiB would leave {} MiB free, below the {} MiB min headroom (short by \
                 {} MiB) of {} MiB free",
                self.alloc_mib,
                self.projected_free_mib,
                self.min_headroom_mib,
                self.short_by_mib,
                self.free_mib
            )
        }
    }
}

/// Decide whether allocating `alloc_bytes` is safe given `free_bytes` free VRAM
/// and a `min_headroom_mib` floor on post-load free VRAM. Pure: the caller does
/// the CUDA query; this only does the arithmetic, so the decision is testable and
/// deterministic. Refuse rather than risk a mid-load OOM.
pub fn evaluate(
    free_bytes: u64,
    alloc_bytes: u64,
    min_headroom_mib: u64,
) -> Result<VramPlan, VramShortfall> {
    let free_mib = free_bytes / MIB;
    let alloc_mib = alloc_bytes / MIB;
    if alloc_bytes > free_bytes {
        // Allocation alone overruns free VRAM: it would OOM partway through.
        let short_by_mib = alloc_mib.saturating_sub(free_mib) + min_headroom_mib;
        return Err(VramShortfall {
            free_mib,
            alloc_mib,
            projected_free_mib: 0,
            min_headroom_mib,
            short_by_mib,
            would_oom: true,
        });
    }
    let projected_free_mib = (free_bytes - alloc_bytes) / MIB;
    if projected_free_mib < min_headroom_mib {
        return Err(VramShortfall {
            free_mib,
            alloc_mib,
            projected_free_mib,
            min_headroom_mib,
            short_by_mib: min_headroom_mib - projected_free_mib,
            would_oom: false,
        });
    }
    Ok(VramPlan {
        free_mib,
        alloc_mib,
        projected_free_mib,
        min_headroom_mib,
    })
}

/// Free VRAM in bytes, or `None` when CUDA is unavailable (feature off / no
/// device). Thin wrapper over the existing device probe so callers stay
/// cfg-free.
pub fn free_vram_bytes() -> Option<u64> {
    crate::cuda::probe_capability().map(|c| c.vram_free_bytes)
}

// --- median / variance (pure; used by the contention report) ----------------

/// Median of a non-empty u64 sample (average of the two middles for even N).
pub fn median_u64(samples: &[u64]) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let mut s = samples.to_vec();
    s.sort_unstable();
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2
    }
}

/// Population variance of a u64 sample, as f64.
pub fn variance_u64(samples: &[u64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let n = samples.len() as f64;
    let mean = samples.iter().map(|&x| x as f64).sum::<f64>() / n;
    samples
        .iter()
        .map(|&x| {
            let d = x as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n
}

// --- contention measurement (CUDA host only) --------------------------------

/// Outcome of one contention run.
#[derive(Debug, Clone)]
pub struct ContentionRun {
    pub free_before_mib: u64,
    pub free_after_primary_mib: u64,
    /// Did the second allocation fail cleanly (driver returned an error)?
    pub second_failed_cleanly: bool,
    /// Did the second allocation unexpectedly succeed (i.e. it fit)?
    pub second_succeeded: bool,
    pub error: Option<String>,
}

/// Aggregated contention findings over N runs.
#[derive(Debug, Clone)]
pub struct ContentionReport {
    pub primary_mib: u64,
    pub second_mib: u64,
    pub runs: usize,
    pub clean_fail_count: usize,
    pub success_count: usize,
    pub median_free_after_primary_mib: u64,
    pub variance_free_after_primary: f64,
    /// `"clean-fail"` (second alloc refused cleanly every run — the safe case),
    /// `"fits"` (it actually fit), or `"mixed"`.
    pub verdict: &'static str,
}

impl ContentionReport {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "schema": "camelid.cuda_contention/v1",
            "primary_alloc_mib": self.primary_mib,
            "second_alloc_mib": self.second_mib,
            "runs": self.runs,
            "second_clean_fail_count": self.clean_fail_count,
            "second_success_count": self.success_count,
            "median_free_after_primary_mib": self.median_free_after_primary_mib,
            "variance_free_after_primary_mib2": self.variance_free_after_primary,
            "verdict": self.verdict,
        })
    }
}

/// Occupy `primary_bytes` of VRAM, then attempt to allocate `second_bytes` on the
/// same device; repeat `runs` times. Records, per run, whether the second
/// allocation failed cleanly (driver `CUDA_ERROR_OUT_OF_MEMORY`) or succeeded,
/// plus free VRAM before and after the primary allocation. The process surviving
/// to return the report is itself evidence that contention surfaces as a clean
/// driver error rather than an abort.
#[cfg(feature = "cuda")]
pub fn measure_contention(
    primary_bytes: usize,
    second_bytes: usize,
    runs: usize,
) -> Result<ContentionReport, String> {
    use cudarc::driver::{result, CudaContext};

    let ordinal = crate::cuda::selected_device_ordinal();
    let mut run_results: Vec<ContentionRun> = Vec::with_capacity(runs);

    for _ in 0..runs {
        let ctx = CudaContext::new(ordinal).map_err(|e| format!("cuda ctx: {e}"))?;
        let stream = ctx.default_stream();
        let (free_before, _total) = result::mem_get_info().unwrap_or((0, 0));

        // Primary allocation: stands in for a resident model's weight footprint.
        let primary = stream
            .alloc_zeros::<u8>(primary_bytes)
            .map_err(|e| format!("primary alloc of {primary_bytes} bytes failed: {e}"))?;
        let (free_after_primary, _t) = result::mem_get_info().unwrap_or((0, 0));

        // Second allocation under contention: the question is clean-fail vs OOM.
        let (second_failed_cleanly, second_succeeded, error) =
            match stream.alloc_zeros::<u8>(second_bytes) {
                Ok(buf) => {
                    drop(buf);
                    (false, true, None)
                }
                Err(e) => (true, false, Some(e.to_string())),
            };

        drop(primary);
        // Each iteration drops its context, releasing all device memory before the
        // next run so the measurements are independent.
        run_results.push(ContentionRun {
            free_before_mib: (free_before as u64) / MIB,
            free_after_primary_mib: (free_after_primary as u64) / MIB,
            second_failed_cleanly,
            second_succeeded,
            error,
        });
    }

    Ok(summarize_contention(
        primary_bytes,
        second_bytes,
        &run_results,
    ))
}

/// Pure aggregation of contention runs (separated so it is testable without CUDA).
pub fn summarize_contention(
    primary_bytes: usize,
    second_bytes: usize,
    runs: &[ContentionRun],
) -> ContentionReport {
    let clean_fail_count = runs.iter().filter(|r| r.second_failed_cleanly).count();
    let success_count = runs.iter().filter(|r| r.second_succeeded).count();
    let frees: Vec<u64> = runs.iter().map(|r| r.free_after_primary_mib).collect();
    let verdict = if !runs.is_empty() && clean_fail_count == runs.len() {
        "clean-fail"
    } else if !runs.is_empty() && success_count == runs.len() {
        "fits"
    } else {
        "mixed"
    };
    ContentionReport {
        primary_mib: (primary_bytes as u64) / MIB,
        second_mib: (second_bytes as u64) / MIB,
        runs: runs.len(),
        clean_fail_count,
        success_count,
        median_free_after_primary_mib: median_u64(&frees),
        variance_free_after_primary: variance_u64(&frees),
        verdict,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headroom_ok_when_enough_remains() {
        // 6 GiB free, 3 GiB model, 512 MiB floor → ~3 GiB left, ok.
        let plan = evaluate(6 * 1024 * MIB, 3 * 1024 * MIB, 512).unwrap();
        assert_eq!(plan.projected_free_mib, 3 * 1024);
        assert_eq!(plan.min_headroom_mib, 512);
    }

    #[test]
    fn refuses_when_headroom_violated_naming_shortfall() {
        // 6 GiB free, 5.75 GiB model, 512 MiB floor → ~256 MiB left < 512 → refuse.
        let model = (6 * 1024 - 256) * MIB; // leaves 256 MiB
        let short = evaluate(6 * 1024 * MIB, model, 512).unwrap_err();
        assert!(!short.would_oom);
        assert_eq!(short.projected_free_mib, 256);
        assert_eq!(short.short_by_mib, 512 - 256);
        assert!(short.to_string().contains("min headroom"));
    }

    #[test]
    fn refuses_and_flags_oom_when_model_exceeds_free() {
        // 4 GiB free, 7 GiB model → would OOM mid-load.
        let short = evaluate(4 * 1024 * MIB, 7 * 1024 * MIB, 512).unwrap_err();
        assert!(short.would_oom);
        assert_eq!(short.projected_free_mib, 0);
        assert!(short.to_string().contains("OOM"));
    }

    #[test]
    fn min_headroom_default_and_override() {
        assert_eq!(DEFAULT_MIN_HEADROOM_MIB, 512);
        // (env override is read at call time; not asserting env here to avoid
        // cross-test interference.)
    }

    #[test]
    fn median_and_variance_are_correct() {
        assert_eq!(median_u64(&[3, 1, 2]), 2);
        assert_eq!(median_u64(&[4, 1, 2, 3]), 2); // (2+3)/2 = 2 (integer)
        assert_eq!(median_u64(&[]), 0);
        assert!((variance_u64(&[2, 2, 2]) - 0.0).abs() < 1e-9);
        assert!((variance_u64(&[1, 3]) - 1.0).abs() < 1e-9); // mean 2, var 1
    }

    #[test]
    fn summarize_classifies_clean_fail() {
        let runs = vec![
            ContentionRun {
                free_before_mib: 6000,
                free_after_primary_mib: 2600,
                second_failed_cleanly: true,
                second_succeeded: false,
                error: Some("out of memory".into()),
            },
            ContentionRun {
                free_before_mib: 6000,
                free_after_primary_mib: 2580,
                second_failed_cleanly: true,
                second_succeeded: false,
                error: Some("out of memory".into()),
            },
        ];
        let rep = summarize_contention(3400 * 1024 * 1024, 3400 * 1024 * 1024, &runs);
        assert_eq!(rep.verdict, "clean-fail");
        assert_eq!(rep.clean_fail_count, 2);
        assert_eq!(rep.median_free_after_primary_mib, 2590);
    }
}
