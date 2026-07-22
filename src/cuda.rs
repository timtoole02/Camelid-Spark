//! Optional CUDA GPU backend (additive, gated behind `--features cuda`).
//!
//! The CPU path remains the default build and the correctness reference. This
//! backend must reproduce the exact CPU/llama.cpp parity evidence — a CUDA lane
//! that is fast but diverges from the parity audit is a regression, not a
//! feature. To that end the Q8_0 dot kernel mirrors the CPU reference
//! (`dot_q8_0_encoded_row_with_scales`) operation-for-operation: one thread per
//! output row, an exact integer block dot, then a sequential f32 accumulation
//! of `(int_sum as f32) * weight_scale * input_scale` in block order, compiled
//! with `--fmad=false` so the GPU does not fuse the multiply/add the CPU keeps
//! separate. That yields bit-identical f32 logits and therefore identical
//! greedy argmax / token IDs.
//!
//! Mirrors `src/metal.rs`'s shape: the module is always present; the real
//! implementation is `#[cfg(feature = "cuda")]` and a stub returns "unavailable"
//! otherwise, so callers never need their own cfg gates.

// CUDA is part of the DEFAULT build on Windows (the primary CUDA dev host): `build.rs`
// injects the `cuda` cfg there, and `cudarc` is a non-optional Windows dependency.
// This fails the build loudly if that wiring ever regresses, so a Windows `cargo
// build` can never silently drop to the CPU-only path. (Linux/macOS keep CUDA opt-in.)
#[cfg(all(windows, not(feature = "cuda")))]
compile_error!(
    "CUDA must be enabled by default on Windows: build.rs should emit \
     `cargo:rustc-cfg=feature=\"cuda\"` for windows targets and Cargo.toml should \
     declare cudarc as a non-optional Windows dependency."
);

/// Result of probing for a usable CUDA device at startup.
#[derive(Debug, Clone, Default)]
pub struct CudaDeviceInfo {
    pub available: bool,
    pub device_name: Option<String>,
    /// Why CUDA is unavailable (feature off, no device, init error), for logs.
    pub reason: Option<String>,
}

/// Lightweight device capability snapshot (no kernel compilation), used by the
/// startup hardware-profile probe so VRAM-driven tunables can size to the device.
#[derive(Debug, Clone, Default)]
pub struct CudaCapability {
    pub device_count: usize,
    pub device_name: String,
    /// (major, minor) compute capability; tensor cores require major >= 7.
    pub compute_capability: (u32, u32),
    pub vram_total_bytes: u64,
    pub vram_free_bytes: u64,
}

// Runtime GPU-enable switch, so the UI can toggle the CUDA decode path on/off
// without restarting. Seeded from `CAMELID_CUDA_Q8` on first read, then owned by
// `set_runtime_enabled`. 0 = uninitialised, 1 = disabled, 2 = enabled. This flag
// only *gates* the path; if no CUDA device is present the dispatch still falls
// back to the CPU reference, so enabling it on an unsupported host is harmless.
static RUNTIME_STATE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

fn seed_runtime_from_env() -> bool {
    std::env::var("CAMELID_CUDA_Q8")
        .map(|value| {
            let value = value.trim();
            value.eq_ignore_ascii_case("1")
                || value.eq_ignore_ascii_case("true")
                || value.eq_ignore_ascii_case("on")
                || value.eq_ignore_ascii_case("enabled")
                || value.eq_ignore_ascii_case("yes")
        })
        .unwrap_or(false)
}

/// Whether the CUDA Q8 decode path is currently enabled (UI/env switch). This is
/// the gate the inference dispatch reads; it is independent of whether a device
/// is actually present (see [`is_available`]).
pub fn runtime_enabled() -> bool {
    use std::sync::atomic::Ordering;
    match RUNTIME_STATE.load(Ordering::Relaxed) {
        0 => {
            let enabled = seed_runtime_from_env();
            RUNTIME_STATE.store(if enabled { 2 } else { 1 }, Ordering::Relaxed);
            enabled
        }
        2 => true,
        _ => false,
    }
}

/// Turn the CUDA Q8 decode path on or off at runtime (the UI toggle calls this).
pub fn set_runtime_enabled(enabled: bool) {
    RUNTIME_STATE.store(
        if enabled { 2 } else { 1 },
        std::sync::atomic::Ordering::Relaxed,
    );
}

// Master "GPU acceleration" switch as the user sees it in the UI. This gates the
// GPU-RESIDENT decode engine — the primary, fast GPU path (the legacy `RUNTIME_STATE`
// above only gates the opt-in hybrid Q8 *matmul* used on the CPU-forward fallback, so
// reporting that as "GPU acceleration" read as OFF even while the resident engine ran
// the whole model on the GPU). Defaults ON whenever a CUDA device is present so the
// app uses the GPU out of the box; the UI toggle flips it. 0 = uninitialised,
// 1 = disabled, 2 = enabled.
static GPU_ACCEL_STATE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Whether GPU acceleration (the resident decode engine) is enabled. On by default
/// when a CUDA device is present; flipped by the UI toggle. Independent of the hybrid
/// `runtime_enabled()` switch. Deterministic mode and `CAMELID_CUDA_RESIDENT_DECODE=0`
/// still force it off at their own call sites.
pub fn gpu_accel_enabled() -> bool {
    use std::sync::atomic::Ordering;
    match GPU_ACCEL_STATE.load(Ordering::Relaxed) {
        0 => {
            let on = is_available();
            GPU_ACCEL_STATE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
        2 => true,
        _ => false,
    }
}

/// Turn GPU acceleration (the resident decode engine) on or off at runtime — the UI
/// "GPU acceleration" toggle calls this. A no-op effect on a host without a CUDA
/// device, since inference falls back to the CPU reference either way.
pub fn set_gpu_accel_enabled(enabled: bool) {
    GPU_ACCEL_STATE.store(
        if enabled { 2 } else { 1 },
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// Whether a usable CUDA device is actually present (feature built + device + a
/// kernel that compiled). The UI uses this to decide whether to show the toggle.
pub fn is_available() -> bool {
    detect_cuda_device().available
}

/// The CUDA device name, if a device is present (for the UI label).
pub fn device_name() -> Option<String> {
    detect_cuda_device().device_name
}

/// Which CUDA device every GPU path binds to. Defaults to device 0 — on this
/// laptop the only CUDA device is the discrete NVIDIA RTX 3060 (the Intel iGPU
/// is not CUDA-capable and is never enumerated here). Override with
/// `CAMELID_CUDA_DEVICE=<index>` when a host genuinely has multiple NVIDIA GPUs
/// and the discrete one is not index 0; the chosen index is logged at startup.
pub fn selected_device_ordinal() -> usize {
    std::env::var("CAMELID_CUDA_DEVICE")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(0)
}

#[cfg(feature = "cuda")]
pub use backend::{
    detect_cuda_device, probe_capability, release_async_pool, try_q8_0_block_linear_row,
    try_q8_0_encoded_linear_row, try_q8_0_encoded_linear_rows,
};

#[cfg(not(feature = "cuda"))]
pub use stub::{
    detect_cuda_device, probe_capability, release_async_pool, try_q8_0_block_linear_row,
    try_q8_0_encoded_linear_row, try_q8_0_encoded_linear_rows,
};

#[cfg(not(feature = "cuda"))]
mod stub {
    use super::{CudaCapability, CudaDeviceInfo};

    pub fn probe_capability() -> Option<CudaCapability> {
        None
    }

    /// No-op without CUDA: there is no async memory pool to trim.
    pub fn release_async_pool() {}

    pub fn detect_cuda_device() -> CudaDeviceInfo {
        CudaDeviceInfo {
            available: false,
            device_name: None,
            reason: Some("built without the `cuda` feature".to_string()),
        }
    }

    /// Decode-shaped Q8_0 linear (one input row × `rows` weight rows). Returns
    /// `false` so the caller falls back to the CPU reference path.
    #[allow(clippy::too_many_arguments)]
    pub fn try_q8_0_encoded_linear_row(
        _input_scales: &[f32],
        _input_quants: &[i8],
        _weight_bytes: &[u8],
        _weight_scales: &[f32],
        _rows: usize,
        _blocks_per_row: usize,
        _output: &mut [f32],
    ) -> bool {
        false
    }

    /// Prefill-shaped Q8_0 linear (`input_rows` × `weight_rows`). Returns `false`
    /// so the caller falls back to the CPU reference path.
    #[allow(clippy::too_many_arguments)]
    pub fn try_q8_0_encoded_linear_rows(
        _input_scales: &[f32],
        _input_quants: &[i8],
        _weight_bytes: &[u8],
        _weight_scales: &[f32],
        _input_rows: usize,
        _weight_rows: usize,
        _blocks_per_row: usize,
        _output: &mut [f32],
    ) -> bool {
        false
    }

    /// Decode-shaped Q8_0 linear over the in-memory `Q8_0Block` byte layout
    /// (36 bytes/block: f32 scale + 32 i8 quants), matching the engine's
    /// retained block-dot path. Returns `false` (CPU fallback).
    pub fn try_q8_0_block_linear_row(
        _input_scales: &[f32],
        _input_quants: &[i8],
        _weight_block_bytes: &[u8],
        _rows: usize,
        _blocks_per_row: usize,
        _output: &mut [f32],
    ) -> bool {
        false
    }
}

/// Number of Q8_0 matmuls the CUDA backend has completed on the GPU this
/// process. Zero means the GPU path never ran (e.g. CPU-only fallback). Used by
/// tests/diagnostics to prove the GPU lane was actually exercised.
#[cfg(feature = "cuda")]
pub fn cuda_q8_run_count() -> u64 {
    backend::run_count()
}

#[cfg(not(feature = "cuda"))]
pub fn cuda_q8_run_count() -> u64 {
    0
}

#[cfg(feature = "cuda")]
mod backend {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Mutex, OnceLock};

    use cudarc::driver::{
        result, sys, CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
    };
    use cudarc::nvrtc::{CompileOptions, Ptx};
    use std::sync::Arc;

    use super::CudaDeviceInfo;

    static RUN_COUNT: AtomicU64 = AtomicU64::new(0);
    static LOGGED: AtomicBool = AtomicBool::new(false);
    static ENTRY_LOGGED: AtomicBool = AtomicBool::new(false);

    pub(super) fn run_count() -> u64 {
        RUN_COUNT.load(Ordering::Relaxed)
    }

    /// CUDA C source for the Q8_0 encoded linear kernel. One thread computes one
    /// output row. The integer block dot is exact (i32); the f32 accumulation is
    /// sequential in block order and matches the CPU reference exactly. Built
    /// with `--fmad=false` (see [`compile_options`]) so the `a*b*c + sum` is not
    /// fused into an FMA — the CPU keeps those operations separate.
    const Q8_KERNEL_SRC: &str = r#"
extern "C" __global__ void q8_0_encoded_linear_rows(
    const float* __restrict__ input_scales,   // [input_rows * blocks_per_row]
    const signed char* __restrict__ input_quants, // [input_rows * blocks_per_row * 32]
    const unsigned char* __restrict__ weight_bytes, // [weight_rows * blocks_per_row * 34]
    const float* __restrict__ weight_scales,   // [weight_rows * blocks_per_row]
    const int input_rows,
    const int weight_rows,
    const int blocks_per_row,
    float* __restrict__ output                 // [input_rows * weight_rows]
) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)input_rows * weight_rows;
    if (idx >= total) return;
    int in_row = (int)(idx / weight_rows);
    int w_row = (int)(idx % weight_rows);

    const float* in_scales = input_scales + (long)in_row * blocks_per_row;
    const signed char* in_quants = input_quants + (long)in_row * blocks_per_row * 32;
    const unsigned char* w_bytes = weight_bytes + (long)w_row * blocks_per_row * 34;
    const float* w_scales = weight_scales + (long)w_row * blocks_per_row;

    float sum = 0.0f;
    for (int b = 0; b < blocks_per_row; b++) {
        const unsigned char* wblk = w_bytes + (long)b * 34 + 2; // skip 2-byte f16 scale
        const signed char* iblk = in_quants + (long)b * 32;
        int int_sum = 0;
        for (int j = 0; j < 32; j++) {
            int_sum += (int)((signed char)wblk[j]) * (int)iblk[j];
        }
        sum += (float)int_sum * w_scales[b] * in_scales[b];
    }
    output[(long)in_row * weight_rows + w_row] = sum;
}

// Fast decode matvec over the in-memory Q8_0Block byte layout (36 bytes/block:
// f32 scale at offset 0, then 32 i8 quants). One *warp* per output row: the 32
// lanes stride over the row's blocks so consecutive lanes read consecutive
// blocks (coalesced global loads), each block's 32 i8*i8 products are summed
// exactly with `__dp4a` (4-wide integer dot), the per-block f32 terms are
// accumulated per lane, then a warp-shuffle reduction sums the lanes. The
// per-block integer dot is exact; the cross-block f32 reduction is reassociated
// vs the CPU's sequential sum, so this is token-identical (not bit-identical) —
// the same standard as the Metal GPU path. Verified by the parity audit.
extern "C" __global__ void q8_0_block_linear_row(
    const float* __restrict__ input_scales,   // [blocks_per_row]
    const signed char* __restrict__ input_quants, // [blocks_per_row * 32]
    const unsigned char* __restrict__ weight_bytes, // [rows * blocks_per_row * 36]
    const int rows,
    const int blocks_per_row,
    float* __restrict__ output                 // [rows]
) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = gtid >> 5;          // one warp per output row
    int lane = gtid & 31;
    if (row >= rows) return;
    const unsigned char* wrow = weight_bytes + (long)row * blocks_per_row * 36;
    float partial = 0.0f;
    for (int b = lane; b < blocks_per_row; b += 32) {
        const unsigned char* blk = wrow + (long)b * 36;
        float w_scale = __ldg(reinterpret_cast<const float*>(blk));
        // 32 i8 quants = 8 ints; blk+4 and input are 4-byte aligned.
        const int* wq = reinterpret_cast<const int*>(blk + 4);
        const int* iq = reinterpret_cast<const int*>(input_quants + (long)b * 32);
        int int_sum = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int_sum = __dp4a(wq[k], iq[k], int_sum);
        }
        partial += (float)int_sum * w_scale * input_scales[b];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        partial += __shfl_down_sync(0xffffffffu, partial, off);
    }
    if (lane == 0) output[row] = partial;
}
"#;

    struct CudaBackend {
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        kernel: CudaFunction,
        kernel_block: CudaFunction,
        device_name: String,
        /// GPU-resident weight cache: each Q8_0 weight is uploaded to the GPU
        /// once (keyed by its stable host pointer + length) and reused across
        /// every token, instead of being re-uploaded each step. This is what
        /// makes decode compute-bound (fast) rather than PCIe-bound (slow). The
        /// model's `q8_0_blocks` live for the model's lifetime, so the pointer
        /// is a stable identity; distinct models map at distinct addresses.
        weight_cache: HashMap<(usize, usize), CudaSlice<u8>>,
    }

    // SAFETY: cudarc's context/stream/function handles are Send + Sync; we
    // additionally serialize all access behind a Mutex.
    fn backend() -> Option<&'static Mutex<CudaBackend>> {
        static BACKEND: OnceLock<Option<Mutex<CudaBackend>>> = OnceLock::new();
        BACKEND
            .get_or_init(|| match init_backend() {
                Ok(b) => Some(Mutex::new(b)),
                Err(_) => None,
            })
            .as_ref()
    }

    fn compile_options() -> CompileOptions {
        CompileOptions {
            // Match the CPU reference's separate multiply/add: do not let the
            // compiler contract `a*b*c + sum` into a fused multiply-add, which
            // would round differently and could flip a near-tie token.
            fmad: Some(false),
            // Target a virtual arch that supports the `__dp4a` 8-bit dot
            // intrinsic (compute_61, Pascal+). The PTX is forward-compatible, so
            // the driver JITs it for whatever newer GPU is present (e.g. sm_86).
            arch: Some("compute_61"),
            ..Default::default()
        }
    }

    fn init_backend() -> Result<CudaBackend, String> {
        let ordinal = super::selected_device_ordinal();
        // cudarc panics (rather than returning Err) when the CUDA driver
        // library cannot be loaded — e.g. on a CI runner or any host with no
        // NVIDIA driver. Catch that so `--all-features` builds fall back to the
        // CPU path instead of aborting the process.
        let ctx = std::panic::catch_unwind(|| CudaContext::new(ordinal))
            .map_err(|_| "CUDA driver library not available".to_string())?
            .map_err(|e| format!("CudaContext::new({ordinal}) failed: {e}"))?;
        // After CudaContext::new (which runs cuInit) the driver can report the
        // device count.
        let device_count = result::device::get_count().unwrap_or(0);
        let stream = ctx.default_stream();
        let device_name = ctx
            .name()
            .unwrap_or_else(|_| "unknown CUDA device".to_string());
        // Log the exact device the GPU work binds to, so it is unambiguous which
        // physical GPU runs inference (the Intel iGPU is not a CUDA device and can
        // never appear here). Prints once, at first GPU init.
        let cc_major = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .unwrap_or(0);
        let cc_minor = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .unwrap_or(0);
        let (vram_free, vram_total) = result::mem_get_info().unwrap_or((0, 0));
        eprintln!(
            "[cuda] selected device {ordinal} of {device_count}: \"{device_name}\" \
             (compute capability {cc_major}.{cc_minor}) | VRAM {} MiB free / {} MiB total",
            vram_free / (1024 * 1024),
            vram_total / (1024 * 1024),
        );
        let ptx: Ptx = cudarc::nvrtc::compile_ptx_with_opts(Q8_KERNEL_SRC, compile_options())
            .map_err(|e| format!("nvrtc compile failed: {e}"))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| format!("load_module failed: {e}"))?;
        let kernel = module
            .load_function("q8_0_encoded_linear_rows")
            .map_err(|e| format!("load_function failed: {e}"))?;
        let kernel_block = module
            .load_function("q8_0_block_linear_row")
            .map_err(|e| format!("load_function (block) failed: {e}"))?;
        Ok(CudaBackend {
            ctx,
            stream,
            kernel,
            kernel_block,
            device_name,
            weight_cache: HashMap::new(),
        })
    }

    /// Light device probe for the startup hardware profile: opens the CUDA context
    /// and reads device count / name / compute capability / VRAM, WITHOUT compiling
    /// kernels (so it is cheap and side-effect-free relative to full init). Returns
    /// `None` on any machine without a usable CUDA device.
    pub fn probe_capability() -> Option<super::CudaCapability> {
        let ordinal = super::selected_device_ordinal();
        // See init_backend: a missing CUDA driver library makes cudarc panic
        // rather than return Err, so guard the first call against it.
        let ctx = std::panic::catch_unwind(|| CudaContext::new(ordinal))
            .ok()?
            .ok()?;
        let device_count = result::device::get_count().unwrap_or(0).max(0) as usize;
        let device_name = ctx
            .name()
            .unwrap_or_else(|_| "unknown CUDA device".to_string());
        let cc_major = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .unwrap_or(0)
            .max(0) as u32;
        let cc_minor = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .unwrap_or(0)
            .max(0) as u32;
        let (vram_free, vram_total) = result::mem_get_info().unwrap_or((0, 0));
        Some(super::CudaCapability {
            device_count,
            device_name,
            compute_capability: (cc_major, cc_minor),
            vram_total_bytes: vram_total as u64,
            vram_free_bytes: vram_free as u64,
        })
    }

    /// Return memory cached in the device's default stream-ordered (async) memory
    /// pool to the driver. cudarc allocates device buffers via `cuMemAllocAsync`, so
    /// dropping a `CudaSlice` calls `cuMemFreeAsync`, which returns the bytes to this
    /// pool rather than to the OS — leaving `cuMemGetInfo` (the free-VRAM probe in
    /// `probe_capability`) still counting them as used. After dropping a resident
    /// decode engine we trim the pool to 0 so the freed VRAM becomes visible to the
    /// probe again; otherwise switching to a larger model under-counts free VRAM and
    /// wrongly falls back to the CPU decode path. Best-effort: any error (or a host
    /// without CUDA) is ignored — the caller only loses the reclaim, never correctness.
    pub fn release_async_pool() {
        let ordinal = super::selected_device_ordinal();
        // Retain the primary context so the driver is initialized and the device
        // handle is valid; held until after the trim. Guard the first call against a
        // missing driver library (cudarc panics rather than returning Err there).
        let _ctx = match std::panic::catch_unwind(|| CudaContext::new(ordinal)) {
            Ok(Ok(ctx)) => ctx,
            _ => return,
        };
        let trace = std::env::var_os("CAMELID_RESIDENT_TRACE").is_some();
        let free_before = result::mem_get_info().map(|(f, _)| f).unwrap_or(0);
        // The just-dropped engine released its weight/KV buffers with `cuMemFreeAsync`
        // (cudarc allocates via `cuMemAllocAsync`), which is STREAM-ORDERED: the pool
        // cannot hand that memory back — to the driver via trim OR to the next
        // allocation — until the device has actually retired the frees. Synchronize the
        // context FIRST, then trim. Without the sync the trim runs before the frees
        // retire and reclaims nothing (measured: free stays pinned at the old model's
        // footprint, so the next model's fit probe under-counts and falls back to CPU —
        // the exact bug this function exists to fix).
        let _ = result::ctx::synchronize();
        // SAFETY: `ordinal` indexes a device the driver just reported via the retained
        // context; the default pool handle is valid for the device's lifetime; and
        // `trim_to` only releases pool reservations that no live allocation is using,
        // so it cannot invalidate any outstanding `CudaSlice`.
        unsafe {
            let dev = match result::device::get(ordinal as core::ffi::c_int) {
                Ok(dev) => dev,
                Err(_) => return,
            };
            let pool = match result::device::get_default_mem_pool(dev) {
                Ok(pool) => pool,
                Err(_) => return,
            };
            let _ = result::mem_pool::trim_to(pool, 0);
        }
        if trace {
            let free_after = result::mem_get_info().map(|(f, _)| f).unwrap_or(0);
            eprintln!(
                "[resident-cuda] release_async_pool: free {} MiB -> {} MiB (reclaimed {} MiB)",
                free_before / (1024 * 1024),
                free_after / (1024 * 1024),
                free_after.saturating_sub(free_before) / (1024 * 1024),
            );
        }
    }

    pub fn detect_cuda_device() -> CudaDeviceInfo {
        match backend() {
            Some(b) => {
                let guard = b.lock().expect("cuda backend mutex poisoned");
                CudaDeviceInfo {
                    available: true,
                    device_name: Some(guard.device_name.clone()),
                    reason: None,
                }
            }
            None => CudaDeviceInfo {
                available: false,
                device_name: None,
                reason: Some("no usable CUDA device or initialization failed".to_string()),
            },
        }
    }

    /// Run the Q8_0 encoded linear on the GPU. `input_rows` input rows (each
    /// `blocks_per_row` blocks of 32 i8 quants + per-block f32 scale) are dotted
    /// against `weight_rows` weight rows (each `blocks_per_row` 34-byte blocks +
    /// per-block decoded f32 scale). Output is `input_rows * weight_rows` f32,
    /// laid out row-major by input row. Returns `false` (caller falls back to
    /// CPU) on any error or shape mismatch.
    #[allow(clippy::too_many_arguments)]
    fn run(
        input_scales: &[f32],
        input_quants: &[i8],
        weight_bytes: &[u8],
        weight_scales: &[f32],
        input_rows: usize,
        weight_rows: usize,
        blocks_per_row: usize,
        output: &mut [f32],
    ) -> bool {
        if std::env::var("CAMELID_CUDA_TRACE").as_deref() == Ok("1")
            && !ENTRY_LOGGED.swap(true, Ordering::Relaxed)
        {
            eprintln!(
                "[cuda-trace] run() first call: input_rows={input_rows} weight_rows={weight_rows} blocks_per_row={blocks_per_row} in_scales={} in_quants={} w_bytes={} w_scales={} out={} backend={}",
                input_scales.len(),
                input_quants.len(),
                weight_bytes.len(),
                weight_scales.len(),
                output.len(),
                backend().is_some(),
            );
        }
        if input_rows == 0 || weight_rows == 0 || blocks_per_row == 0 {
            return false;
        }
        // Shape guards: bail to CPU rather than risk an out-of-bounds GPU read.
        if input_scales.len() != input_rows * blocks_per_row
            || input_quants.len() != input_rows * blocks_per_row * 32
            || weight_bytes.len() != weight_rows * blocks_per_row * 34
            || weight_scales.len() != weight_rows * blocks_per_row
            || output.len() != input_rows * weight_rows
        {
            return false;
        }
        let Some(b) = backend() else {
            return false;
        };
        let guard = b.lock().expect("cuda backend mutex poisoned");
        match run_inner(
            &guard,
            input_scales,
            input_quants,
            weight_bytes,
            weight_scales,
            input_rows,
            weight_rows,
            blocks_per_row,
            output,
        ) {
            Ok(()) => {
                RUN_COUNT.fetch_add(1, Ordering::Relaxed);
                if !LOGGED.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "[cuda] Q8_0 hybrid decode active on {} — first GPU matmul completed",
                        guard.device_name
                    );
                }
                true
            }
            Err(_) => false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_inner(
        b: &CudaBackend,
        input_scales: &[f32],
        input_quants: &[i8],
        weight_bytes: &[u8],
        weight_scales: &[f32],
        input_rows: usize,
        weight_rows: usize,
        blocks_per_row: usize,
        output: &mut [f32],
    ) -> Result<(), cudarc::driver::DriverError> {
        let stream = &b.stream;
        let d_in_scales = stream.clone_htod(input_scales)?;
        let d_in_quants = stream.clone_htod(input_quants)?;
        let d_w_bytes = stream.clone_htod(weight_bytes)?;
        let d_w_scales = stream.clone_htod(weight_scales)?;
        let mut d_out = stream.alloc_zeros::<f32>(output.len())?;

        let total = (input_rows * weight_rows) as u32;
        let block_dim = 256u32;
        let grid_dim = total.div_ceil(block_dim);
        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let input_rows_i = input_rows as i32;
        let weight_rows_i = weight_rows as i32;
        let blocks_per_row_i = blocks_per_row as i32;
        let mut builder = stream.launch_builder(&b.kernel);
        builder
            .arg(&d_in_scales)
            .arg(&d_in_quants)
            .arg(&d_w_bytes)
            .arg(&d_w_scales)
            .arg(&input_rows_i)
            .arg(&weight_rows_i)
            .arg(&blocks_per_row_i)
            .arg(&mut d_out);
        // SAFETY: the kernel reads the four input buffers and writes d_out, all
        // sized to match the launch dimensions per the shape guards in `run`.
        unsafe { builder.launch(cfg)? };
        stream.memcpy_dtoh(&d_out, output)?;
        b.ctx.synchronize()?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_q8_0_encoded_linear_row(
        input_scales: &[f32],
        input_quants: &[i8],
        weight_bytes: &[u8],
        weight_scales: &[f32],
        rows: usize,
        blocks_per_row: usize,
        output: &mut [f32],
    ) -> bool {
        // Decode: a single input row against `rows` weight rows.
        run(
            input_scales,
            input_quants,
            weight_bytes,
            weight_scales,
            1,
            rows,
            blocks_per_row,
            output,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_q8_0_encoded_linear_rows(
        input_scales: &[f32],
        input_quants: &[i8],
        weight_bytes: &[u8],
        weight_scales: &[f32],
        input_rows: usize,
        weight_rows: usize,
        blocks_per_row: usize,
        output: &mut [f32],
    ) -> bool {
        run(
            input_scales,
            input_quants,
            weight_bytes,
            weight_scales,
            input_rows,
            weight_rows,
            blocks_per_row,
            output,
        )
    }

    /// Decode matvec over the in-memory `Q8_0Block` byte layout. `weight_bytes`
    /// is `rows * blocks_per_row * 36` bytes (f32 scale + 32 i8 quants/block);
    /// the input row is given as separate `blocks_per_row` scales + quants.
    /// Returns `false` (CPU fallback) on any error or shape mismatch.
    fn run_block(
        input_scales: &[f32],
        input_quants: &[i8],
        weight_bytes: &[u8],
        rows: usize,
        blocks_per_row: usize,
        output: &mut [f32],
    ) -> bool {
        if rows == 0 || blocks_per_row == 0 {
            return false;
        }
        if std::env::var("CAMELID_CUDA_TRACE").as_deref() == Ok("1")
            && !ENTRY_LOGGED.swap(true, Ordering::Relaxed)
        {
            eprintln!(
                "[cuda-trace] run_block() first call: rows={rows} blocks_per_row={blocks_per_row} in_scales={} in_quants={} w_bytes={} out={}",
                input_scales.len(),
                input_quants.len(),
                weight_bytes.len(),
                output.len(),
            );
        }
        if input_scales.len() != blocks_per_row
            || input_quants.len() != blocks_per_row * 32
            || weight_bytes.len() != rows * blocks_per_row * 36
            || output.len() != rows
        {
            return false;
        }
        let Some(b) = backend() else {
            return false;
        };
        let mut guard = b.lock().expect("cuda backend mutex poisoned");
        match run_block_inner(
            &mut guard,
            input_scales,
            input_quants,
            weight_bytes,
            rows,
            blocks_per_row,
            output,
        ) {
            Ok(()) => {
                RUN_COUNT.fetch_add(1, Ordering::Relaxed);
                if !LOGGED.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "[cuda] Q8_0 hybrid decode active on {} — first GPU matmul completed",
                        guard.device_name
                    );
                }
                true
            }
            Err(_) => false,
        }
    }

    fn run_block_inner(
        b: &mut CudaBackend,
        input_scales: &[f32],
        input_quants: &[i8],
        weight_bytes: &[u8],
        rows: usize,
        blocks_per_row: usize,
        output: &mut [f32],
    ) -> Result<(), cudarc::driver::DriverError> {
        // Upload this weight to the GPU once and keep it resident; reuse the
        // cached device buffer on every later token. The per-token traffic is
        // then just the small input vector and output vector, so decode becomes
        // GPU-compute-bound instead of PCIe-bound. On a failed upload (e.g. out
        // of VRAM) the `?` propagates and the caller falls back to the CPU dot.
        let key = (weight_bytes.as_ptr() as usize, weight_bytes.len());
        if !b.weight_cache.contains_key(&key) {
            let resident = b.stream.clone_htod(weight_bytes)?;
            b.weight_cache.insert(key, resident);
        }
        let d_w_bytes = b.weight_cache.get(&key).expect("weight just inserted");

        let stream = &b.stream;
        let d_in_scales = stream.clone_htod(input_scales)?;
        let d_in_quants = stream.clone_htod(input_quants)?;
        let mut d_out = stream.alloc_zeros::<f32>(output.len())?;

        // One warp (32 threads) per output row.
        let block_dim = 256u32;
        let grid_dim = ((rows as u32) * 32).div_ceil(block_dim);
        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let rows_i = rows as i32;
        let blocks_per_row_i = blocks_per_row as i32;
        let mut builder = stream.launch_builder(&b.kernel_block);
        builder
            .arg(&d_in_scales)
            .arg(&d_in_quants)
            .arg(d_w_bytes)
            .arg(&rows_i)
            .arg(&blocks_per_row_i)
            .arg(&mut d_out);
        // SAFETY: buffers are sized to the launch dimensions per the shape
        // guards in `run_block`.
        unsafe { builder.launch(cfg)? };
        stream.memcpy_dtoh(&d_out, output)?;
        b.ctx.synchronize()?;
        Ok(())
    }

    /// Decode-shaped Q8_0 linear over the in-memory `Q8_0Block` byte layout.
    pub fn try_q8_0_block_linear_row(
        input_scales: &[f32],
        input_quants: &[i8],
        weight_block_bytes: &[u8],
        rows: usize,
        blocks_per_row: usize,
        output: &mut [f32],
    ) -> bool {
        run_block(
            input_scales,
            input_quants,
            weight_block_bytes,
            rows,
            blocks_per_row,
            output,
        )
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        // Reference dot, identical in operation order to the CPU engine's
        // `dot_q8_0_encoded_row_with_scales`: exact i32 block dot, then a
        // sequential f32 accumulation of `(int_sum as f32) * w_scale * i_scale`
        // in block order, no FMA. The GPU kernel must match this bit-for-bit.
        fn reference_row(
            input_scales: &[f32],
            input_quants: &[i8],
            weight_block_quants: &[&[i8]],
            weight_scales: &[f32],
            blocks_per_row: usize,
        ) -> f32 {
            let mut sum = 0.0f32;
            for b in 0..blocks_per_row {
                let mut int_sum = 0i32;
                for j in 0..32 {
                    int_sum +=
                        i32::from(weight_block_quants[b][j]) * i32::from(input_quants[b * 32 + j]);
                }
                sum += int_sum as f32 * weight_scales[b] * input_scales[b];
            }
            sum
        }

        // Tiny deterministic LCG so the test needs no rand dependency and is
        // reproducible across runs/platforms.
        struct Lcg(u64);
        impl Lcg {
            fn next_u32(&mut self) -> u32 {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (self.0 >> 32) as u32
            }
            fn next_i8(&mut self) -> i8 {
                (self.next_u32() & 0xff) as u8 as i8
            }
            fn next_scale(&mut self) -> f32 {
                // Small positive f16-ish scales, like real Q8_0 block scales.
                ((self.next_u32() % 1000) as f32 + 1.0) / 4096.0
            }
        }

        // Requires a CUDA device; ignored by default so GPU-less CI (which has
        // no NVIDIA driver) compiles but does not run it. Run on a CUDA host
        // with `cargo test --features cuda -- --ignored`.
        #[test]
        #[ignore = "requires a CUDA device"]
        fn cuda_q8_kernel_is_bit_identical_to_cpu_reference() {
            if !detect_cuda_device().available {
                eprintln!("skipping: no CUDA device available");
                return;
            }
            let blocks_per_row = 64usize; // TinyLlama hidden 2048 = 64 blocks of 32
            let weight_rows = 300usize; // not a multiple of the 256 block size
            let mut rng = Lcg(0x1234_5678_9abc_def0);

            // Input row.
            let mut input_quants = vec![0i8; blocks_per_row * 32];
            for q in input_quants.iter_mut() {
                *q = rng.next_i8();
            }
            let input_scales: Vec<f32> = (0..blocks_per_row).map(|_| rng.next_scale()).collect();

            // Weight rows: 34-byte blocks (2-byte scale header + 32 quants) plus
            // a separately decoded f32 scale per block (as the engine passes).
            let mut weight_bytes = vec![0u8; weight_rows * blocks_per_row * 34];
            let mut weight_scales = vec![0f32; weight_rows * blocks_per_row];
            for r in 0..weight_rows {
                for b in 0..blocks_per_row {
                    let blk = (r * blocks_per_row + b) * 34;
                    // header bytes are ignored by the kernel; fill with noise.
                    weight_bytes[blk] = rng.next_i8() as u8;
                    weight_bytes[blk + 1] = rng.next_i8() as u8;
                    for j in 0..32 {
                        weight_bytes[blk + 2 + j] = rng.next_i8() as u8;
                    }
                    weight_scales[r * blocks_per_row + b] = rng.next_scale();
                }
            }

            // CPU reference.
            let mut expected = vec![0f32; weight_rows];
            for (r, slot) in expected.iter_mut().enumerate() {
                let block_quants: Vec<&[i8]> = (0..blocks_per_row)
                    .map(|b| {
                        let start = (r * blocks_per_row + b) * 34 + 2;
                        // reinterpret the u8 quants as i8
                        let bytes = &weight_bytes[start..start + 32];
                        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<i8>(), 32) }
                    })
                    .collect();
                *slot = reference_row(
                    &input_scales,
                    &input_quants,
                    &block_quants,
                    &weight_scales[r * blocks_per_row..(r + 1) * blocks_per_row],
                    blocks_per_row,
                );
            }

            // GPU.
            let mut got = vec![0f32; weight_rows];
            let ok = try_q8_0_encoded_linear_row(
                &input_scales,
                &input_quants,
                &weight_bytes,
                &weight_scales,
                weight_rows,
                blocks_per_row,
                &mut got,
            );
            assert!(ok, "GPU kernel did not run");

            let mut mismatches = 0;
            for (r, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                if g.to_bits() != e.to_bits() {
                    if mismatches < 5 {
                        eprintln!(
                            "row {r}: gpu={g} ({:#010x}) cpu={e} ({:#010x})",
                            g.to_bits(),
                            e.to_bits()
                        );
                    }
                    mismatches += 1;
                }
            }
            assert_eq!(
                mismatches, 0,
                "{mismatches}/{weight_rows} rows differ bit-for-bit from the CPU reference"
            );
        }

        // The fast block kernel sums blocks across a warp (reassociated f32), so
        // it matches the CPU reference very closely but not bit-for-bit. Assert a
        // tight relative tolerance; end-to-end token identity is covered by the
        // TinyLlama parity audit.
        #[test]
        #[ignore = "requires a CUDA device"]
        fn cuda_block_kernel_matches_cpu_reference_within_tolerance() {
            if !detect_cuda_device().available {
                eprintln!("skipping: no CUDA device available");
                return;
            }
            let blocks_per_row = 64usize;
            let weight_rows = 257usize;
            let mut rng = Lcg(0x0fed_cba9_8765_4321);

            // Single input row, as separate scales + quants (what the engine
            // passes via with_q8_0_block_scales_and_quants).
            let mut input_quants = vec![0i8; blocks_per_row * 32];
            for q in input_quants.iter_mut() {
                *q = rng.next_i8();
            }
            let input_scales: Vec<f32> = (0..blocks_per_row).map(|_| rng.next_scale()).collect();

            // Weight rows in the Q8_0Block byte layout: 36 bytes/block =
            // f32 scale (LE) + 32 i8 quants.
            let mut weight_bytes = vec![0u8; weight_rows * blocks_per_row * 36];
            let mut weight_scales = vec![0f32; weight_rows * blocks_per_row];
            for r in 0..weight_rows {
                for b in 0..blocks_per_row {
                    let blk = (r * blocks_per_row + b) * 36;
                    let scale = rng.next_scale();
                    weight_scales[r * blocks_per_row + b] = scale;
                    weight_bytes[blk..blk + 4].copy_from_slice(&scale.to_le_bytes());
                    for j in 0..32 {
                        weight_bytes[blk + 4 + j] = rng.next_i8() as u8;
                    }
                }
            }

            // CPU reference (same op order as q8_0_dot_rows scalar path).
            let mut expected = vec![0f32; weight_rows];
            for (r, slot) in expected.iter_mut().enumerate() {
                let block_quants: Vec<&[i8]> = (0..blocks_per_row)
                    .map(|b| {
                        let start = (r * blocks_per_row + b) * 36 + 4;
                        let bytes = &weight_bytes[start..start + 32];
                        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<i8>(), 32) }
                    })
                    .collect();
                *slot = reference_row(
                    &input_scales,
                    &input_quants,
                    &block_quants,
                    &weight_scales[r * blocks_per_row..(r + 1) * blocks_per_row],
                    blocks_per_row,
                );
            }

            let mut got = vec![0f32; weight_rows];
            let ok = try_q8_0_block_linear_row(
                &input_scales,
                &input_quants,
                &weight_bytes,
                weight_rows,
                blocks_per_row,
                &mut got,
            );
            assert!(ok, "GPU block kernel did not run");

            let mut worst = 0.0f32;
            for (g, e) in got.iter().zip(expected.iter()) {
                let denom = e.abs().max(1.0);
                worst = worst.max((g - e).abs() / denom);
            }
            assert!(
                worst < 1e-4,
                "block-kernel worst relative error {worst} exceeds 1e-4 vs CPU reference"
            );
        }
    }
}
