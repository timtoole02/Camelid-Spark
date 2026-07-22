//! VRAM contention harness (Task 4). Built only with `--features cuda`; the test
//! is `#[ignore]` because it needs a real CUDA device. Run it on the CUDA host:
//!
//! ```text
//! cargo test --features cuda --test cuda_contention -- --ignored --nocapture
//! ```
//!
//! It occupies a model-sized allocation (default ~3.4 GiB, a 3B Q8 proxy), then
//! attempts a second allocation of the same size on the same device, repeating 5
//! times, and records whether the second allocation fails *cleanly* (the driver
//! returns `CUDA_ERROR_OUT_OF_MEMORY`) or the process OOMs. Median + variance of
//! free-VRAM-after-primary are reported. Results are written to
//! `qa/cuda/contention-latest.json` and summarized in
//! `qa/cuda/CONTENTION_FINDINGS.md`.
//!
//! Size overrides (MiB): `CAMELID_CONTENTION_PRIMARY_MIB`,
//! `CAMELID_CONTENTION_SECOND_MIB`, `CAMELID_CONTENTION_RUNS`.

#![cfg(feature = "cuda")]

use camelid::cuda_vram;

fn mib_env(key: &str, default_mib: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(default_mib)
}

#[test]
#[ignore = "requires a CUDA device; run with --ignored on a CUDA host"]
fn contention_primary_then_second_alloc() {
    const MIB: usize = 1024 * 1024;
    // 3B Q8_0 weights ≈ 3.4 GiB; use that as both the primary occupant and the
    // contending second request by default.
    let primary = mib_env("CAMELID_CONTENTION_PRIMARY_MIB", 3481) * MIB;
    let second = mib_env("CAMELID_CONTENTION_SECOND_MIB", 3481) * MIB;
    let runs = std::env::var("CAMELID_CONTENTION_RUNS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(5);

    let report = cuda_vram::measure_contention(primary, second, runs)
        .expect("contention measurement failed");

    let mut json = serde_json::to_string_pretty(&report.to_json()).unwrap();
    json.push('\n');
    eprintln!("{json}");
    let _ = std::fs::create_dir_all("qa/cuda");
    std::fs::write("qa/cuda/contention-latest.json", &json).expect("write findings artifact");

    // Surviving to here is itself evidence the OOM surfaced as a clean driver
    // error rather than aborting the process. The outcome must be consistent
    // across runs (not flaky).
    assert_eq!(report.runs, runs);
    assert_ne!(
        report.verdict, "mixed",
        "contention outcome was inconsistent across runs: {}",
        json
    );
}
