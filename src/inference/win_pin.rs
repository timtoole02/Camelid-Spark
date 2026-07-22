//! STAMPEDE Phase 1 — optional Windows worker pinning for the dedicated
//! decode/prefill rayon pools.
//!
//! Thread placement only: no arithmetic runs here and no kernel changes, so
//! every mode is bit-identical by construction. Without pinning the Windows
//! scheduler is free to co-schedule two pool workers on the SMT siblings of
//! one physical core and to migrate workers mid-token — both cost achieved
//! memory bandwidth on the decode weight stream (llama.cpp exposes the same
//! lever as its `-C` cpumask; macOS workers get a QoS class, Windows workers
//! previously got nothing).
//!
//! `CAMELID_WIN_PIN` selects the mode, default OFF until the A/B receipts
//! flip it:
//! * `ideal` — `SetThreadIdealProcessor` placement hint (soft; the scheduler
//!   may still migrate under pressure).
//! * `hard`  — `SetThreadAffinityMask` to the worker's physical core (both
//!   SMT siblings stay in the mask, so the core is owned but the scheduler
//!   can still bounce between its siblings).
//!
//! Placement policy (both modes): pool worker `i` owns physical core
//! `i % cores`, using the per-core sibling masks reported by
//! `GetLogicalProcessorInformation` (no adjacent-enumeration assumption).
//! The decode pool is sized to the physical core count, so this is one
//! worker per core; the prefill pool spans the logical count, so sibling
//! workers land on the same core's mask in pairs — preserving the P0.6
//! wider-prefill win instead of fighting it. Detection failure or an
//! out-of-range worker index degrades to no pin (fail-open: unpinned is the
//! shipped behavior).

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
use std::sync::OnceLock;

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WinPinMode {
    Off,
    Ideal,
    Hard,
}

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
pub(super) fn win_pin_mode() -> WinPinMode {
    static MODE: OnceLock<WinPinMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        match std::env::var("CAMELID_WIN_PIN") {
            Ok(value) => {
                let value = value.trim();
                if value.eq_ignore_ascii_case("ideal") {
                    WinPinMode::Ideal
                } else if value.eq_ignore_ascii_case("hard") {
                    WinPinMode::Hard
                } else {
                    // Unknown values (and explicit off/0) stay unpinned: the
                    // pinned lanes are opt-in and fail-open.
                    WinPinMode::Off
                }
            }
            Err(_) => WinPinMode::Off,
        }
    })
}

/// Per-physical-core logical-processor masks from
/// `GetLogicalProcessorInformation` (one `RelationProcessorCore` record per
/// core, its `ProcessorMask` covering that core's SMT siblings). `None` when
/// detection fails; order matches the OS enumeration order of cores.
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
fn windows_core_masks() -> Option<&'static Vec<usize>> {
    static MASKS: OnceLock<Option<Vec<usize>>> = OnceLock::new();
    MASKS
        .get_or_init(|| {
            use windows_sys::Win32::System::SystemInformation::{
                GetLogicalProcessorInformation, SYSTEM_LOGICAL_PROCESSOR_INFORMATION,
            };
            const RELATION_PROCESSOR_CORE: i32 = 0;
            unsafe {
                let mut len: u32 = 0;
                // First call sizes the buffer (fails with ERROR_INSUFFICIENT_BUFFER).
                GetLogicalProcessorInformation(std::ptr::null_mut(), &mut len);
                if len == 0 {
                    return None;
                }
                let count =
                    len as usize / std::mem::size_of::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION>();
                if count == 0 {
                    return None;
                }
                let mut buf: Vec<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> = Vec::with_capacity(count);
                if GetLogicalProcessorInformation(buf.as_mut_ptr(), &mut len) == 0 {
                    return None;
                }
                buf.set_len(count);
                let masks: Vec<usize> = buf
                    .iter()
                    .filter(|info| info.Relationship == RELATION_PROCESSOR_CORE)
                    .map(|info| info.ProcessorMask)
                    .filter(|mask| *mask != 0)
                    .collect();
                (!masks.is_empty()).then_some(masks)
            }
        })
        .as_ref()
}

/// Pin the calling pool worker (index `worker`) per the selected mode.
/// Shared by the decode and prefill pool `start_handler`s.
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
pub(super) fn pin_pool_worker(pool: &'static str, worker: usize) {
    use windows_sys::Win32::System::Threading::{
        GetCurrentThread, SetThreadAffinityMask, SetThreadIdealProcessor,
    };
    let mode = win_pin_mode();
    if mode == WinPinMode::Off {
        return;
    }
    let Some(masks) = windows_core_masks() else {
        return;
    };
    let core = worker % masks.len();
    let mask = masks[core];
    // SAFETY: GetCurrentThread returns a pseudo-handle that needs no
    // CloseHandle; both Set* calls only reconfigure the calling thread's
    // scheduling and cannot alias memory.
    unsafe {
        match mode {
            WinPinMode::Hard => {
                if SetThreadAffinityMask(GetCurrentThread(), mask) == 0 {
                    tracing::debug!(pool, worker, core, mask, "SetThreadAffinityMask failed");
                }
            }
            WinPinMode::Ideal => {
                let ideal = mask.trailing_zeros();
                if SetThreadIdealProcessor(GetCurrentThread(), ideal) == u32::MAX {
                    tracing::debug!(pool, worker, ideal, "SetThreadIdealProcessor failed");
                }
            }
            WinPinMode::Off => unreachable!(),
        }
    }
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        tracing::info!(
            ?mode,
            cores = masks.len(),
            "CAMELID_WIN_PIN: pinning pool workers to physical cores"
        );
    });
}

#[cfg(not(all(target_os = "windows", target_arch = "x86_64")))]
pub(super) fn pin_pool_worker(_pool: &'static str, _worker: usize) {}
