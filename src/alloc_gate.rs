//! Decode zero-alloc gate (Lane B step 6).
//!
//! A counting global allocator plus a driver that decodes real tokens through
//! a loaded model and reports steady-state heap-allocation counts per token.
//! This is the evidence that the scratch-pool strip actually removed the
//! per-token heap churn (main allocated ~3 heap objects per op output, layer-
//! proportional; the stripped loop's steady state is a small constant set:
//! the per-token tensors that escape to the caller — embedding, final norm,
//! logits — plus the timings vec).
//!
//! Why a feature-gated binary and not a `#[cfg(test)]` unit test: test builds
//! deliberately leave the env-flag accessors and `ResolvedRuntimePlan::from_env`
//! UNCACHED (tests mutate env vars), so a unit test would count hundreds of
//! env-parsing allocations per token that do not exist in a real binary. The
//! gate must measure the shipped configuration, so it runs in a normal build
//! with `--features alloc-gate` (see the hidden `bench-alloc-gate` subcommand).
//!
//! Run CPU-pinned like every decode receipt: `CUDA_VISIBLE_DEVICES=-1`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// System allocator wrapper that counts allocation events and requested bytes.
/// Deallocations are not counted (the gate is about allocation churn).
pub struct CountingAllocator;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

/// Attribution mode: when armed, allocations of at least
/// `TRACE_MIN_BYTES` print a backtrace (bounded count, reentrancy-guarded —
/// capturing a backtrace allocates).
static TRACE_BIG: AtomicBool = AtomicBool::new(false);
static TRACES_REMAINING: AtomicU64 = AtomicU64::new(0);
/// Matching allocations to skip before tracing (`CAMELID_ALLOC_GATE_TRACE_SKIP`),
/// so the sample can land mid-token instead of on token-start sites.
static TRACES_TO_SKIP: AtomicU64 = AtomicU64::new(0);
/// Trace threshold; defaults to 1 MiB, overridable at gate start via
/// `CAMELID_ALLOC_GATE_TRACE_MIN` (bytes) to attribute small-alloc churn.
static TRACE_MIN_BYTES: AtomicU64 = AtomicU64::new(1 << 20);

thread_local! {
    static IN_TRACE: Cell<bool> = const { Cell::new(false) };
}

fn maybe_trace(size: usize) {
    if !TRACE_BIG.load(Ordering::Relaxed) || (size as u64) < TRACE_MIN_BYTES.load(Ordering::Relaxed)
    {
        return;
    }
    IN_TRACE.with(|guard| {
        if guard.get() {
            return;
        }
        if TRACES_TO_SKIP
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
            .is_ok()
        {
            return;
        }
        if TRACES_REMAINING
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
            .is_err()
        {
            return;
        }
        guard.set(true);
        let backtrace = std::backtrace::Backtrace::force_capture();
        eprintln!("=== alloc {size} bytes ===\n{backtrace}");
        guard.set(false);
    });
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        maybe_trace(layout.size());
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        maybe_trace(layout.size());
        System.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        maybe_trace(new_size);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn allocation_snapshot() -> (u64, u64) {
    (
        ALLOCATIONS.load(Ordering::Relaxed),
        ALLOCATED_BYTES.load(Ordering::Relaxed),
    )
}

/// Load `model`, decode `warmup` tokens to warm the scratch pools, decode
/// binding cells, and KV growth chunk, then decode `tokens` more and report
/// the allocation delta of that steady-state window.
pub fn run_decode_alloc_gate(
    model: &std::path::Path,
    warmup: usize,
    tokens: usize,
    compute_logits: bool,
    trace_big: bool,
) -> crate::Result<serde_json::Value> {
    let gguf = crate::gguf::read_metadata(model)?;
    let config = crate::model::LlamaModelConfig::from_gguf(&gguf)?;
    let binding = crate::model::LlamaTensorBinding::bind(&gguf, &config)?;
    let store = crate::tensor::TensorStore::open(model, &gguf);
    let weights = crate::inference::LlamaLoadedWeights::load(&store, &binding, None)?;
    let mut session = crate::inference::LlamaInferenceSession::new(config, weights)?;

    // Token identity does not change the decode allocation path; a fixed
    // in-vocab id keeps the run deterministic.
    let token_id = 100u32;
    for _ in 0..warmup {
        session.forward_single_token_alloc_probe(token_id, compute_logits)?;
    }

    if trace_big {
        if let Ok(min) = std::env::var("CAMELID_ALLOC_GATE_TRACE_MIN") {
            if let Ok(min) = min.parse::<u64>() {
                TRACE_MIN_BYTES.store(min, Ordering::Relaxed);
            }
        }
        if let Ok(skip) = std::env::var("CAMELID_ALLOC_GATE_TRACE_SKIP") {
            if let Ok(skip) = skip.parse::<u64>() {
                TRACES_TO_SKIP.store(skip, Ordering::Relaxed);
            }
        }
        TRACES_REMAINING.store(12, Ordering::Relaxed);
        TRACE_BIG.store(true, Ordering::Relaxed);
    }
    let (allocs_before, bytes_before) = allocation_snapshot();
    for _ in 0..tokens {
        session.forward_single_token_alloc_probe(token_id, compute_logits)?;
    }
    let (allocs_after, bytes_after) = allocation_snapshot();
    TRACE_BIG.store(false, Ordering::Relaxed);

    let allocs = allocs_after - allocs_before;
    let bytes = bytes_after - bytes_before;
    Ok(serde_json::json!({
        "schema": "camelid.bench-alloc-gate/v1",
        "model": model.display().to_string(),
        "warmup_tokens": warmup,
        "measured_tokens": tokens,
        "compute_logits": compute_logits,
        "allocations": allocs,
        "allocated_bytes": bytes,
        "allocations_per_token": allocs as f64 / tokens as f64,
        "allocated_bytes_per_token": bytes as f64 / tokens as f64,
    }))
}
