//! Windows scheduling substrate (Lane C).
//!
//! Cross-platform-first engines omit the Windows-native QoS controls that keep
//! inference threads off the efficiency/background track. The highest-leverage,
//! least-known of these is the **EcoQoS power-throttling opt-out**: by default
//! Windows may classify a long-running compute process's threads as background
//! work and park/downclock them, which silently caps sustained decode. Opting
//! out keeps the process at full execution speed.
//!
//! Every call here is best-effort and `cfg(windows)`-gated: on any other target,
//! or if the Win32 call fails, it returns [`EcoQosStatus::Unavailable`] and the
//! engine runs exactly as before. This module changes scheduling only — never
//! the math — so it cannot affect parity.

use serde::{Deserialize, Serialize};

/// Outcome of an EcoQoS control request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EcoQosStatus {
    /// Throttling explicitly disabled: the process runs at full execution speed.
    OptedOut,
    /// Reverted to OS-managed throttling.
    OsManaged,
    /// The control was unavailable (non-Windows host, or the API rejected it).
    Unavailable,
}

/// Process-level EcoQoS control. `opt_out = true` disables execution-speed
/// throttling for the whole process — every thread, including the Rayon decode
/// pool — so Windows cannot park or downclock inference as "background" work.
/// `opt_out = false` returns to OS-managed throttling.
///
/// Best-effort and idempotent: returns [`EcoQosStatus::Unavailable`] rather than
/// erroring, so callers never have to handle a failure path.
#[cfg(windows)]
pub fn set_eco_qos_opt_out(opt_out: bool) -> EcoQosStatus {
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, ProcessPowerThrottling, SetProcessInformation,
        PROCESS_POWER_THROTTLING_CURRENT_VERSION, PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
        PROCESS_POWER_THROTTLING_STATE,
    };

    // opt-out: control EXECUTION_SPEED, state cleared => "do not throttle".
    // OS-managed: control 0 => "let Windows decide".
    let control_mask = if opt_out {
        PROCESS_POWER_THROTTLING_EXECUTION_SPEED
    } else {
        0
    };
    let state = PROCESS_POWER_THROTTLING_STATE {
        Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        ControlMask: control_mask,
        StateMask: 0,
    };

    // SAFETY: `state` outlives the call; size is the struct's own size.
    let ok = unsafe {
        SetProcessInformation(
            GetCurrentProcess(),
            ProcessPowerThrottling,
            &state as *const PROCESS_POWER_THROTTLING_STATE as *const core::ffi::c_void,
            std::mem::size_of::<PROCESS_POWER_THROTTLING_STATE>() as u32,
        )
    };
    if ok != 0 {
        if opt_out {
            EcoQosStatus::OptedOut
        } else {
            EcoQosStatus::OsManaged
        }
    } else {
        EcoQosStatus::Unavailable
    }
}

#[cfg(not(windows))]
pub fn set_eco_qos_opt_out(_opt_out: bool) -> EcoQosStatus {
    EcoQosStatus::Unavailable
}

/// Per-thread EcoQoS control for the **current** thread, via
/// `SetThreadInformation(ThreadPowerThrottling, …)`. This is the §1.2-correct
/// granularity: applied across the compute pool it disables execution-speed
/// throttling for inference workers only, leaving UI/background threads on the
/// eco-friendly OS-managed default (unlike the process-wide
/// [`set_eco_qos_opt_out`], whose blast radius is every thread).
///
/// Best-effort and idempotent: returns [`EcoQosStatus::Unavailable`] rather than
/// erroring.
#[cfg(windows)]
pub fn set_thread_eco_qos_opt_out(opt_out: bool) -> EcoQosStatus {
    use windows_sys::Win32::System::Threading::{
        GetCurrentThread, SetThreadInformation, ThreadPowerThrottling,
        THREAD_POWER_THROTTLING_CURRENT_VERSION, THREAD_POWER_THROTTLING_EXECUTION_SPEED,
        THREAD_POWER_THROTTLING_STATE,
    };

    let control_mask = if opt_out {
        THREAD_POWER_THROTTLING_EXECUTION_SPEED
    } else {
        0
    };
    let state = THREAD_POWER_THROTTLING_STATE {
        Version: THREAD_POWER_THROTTLING_CURRENT_VERSION,
        ControlMask: control_mask,
        StateMask: 0,
    };

    // SAFETY: `state` outlives the call; size is the struct's own size.
    let ok = unsafe {
        SetThreadInformation(
            GetCurrentThread(),
            ThreadPowerThrottling,
            &state as *const THREAD_POWER_THROTTLING_STATE as *const core::ffi::c_void,
            std::mem::size_of::<THREAD_POWER_THROTTLING_STATE>() as u32,
        )
    };
    if ok != 0 {
        if opt_out {
            EcoQosStatus::OptedOut
        } else {
            EcoQosStatus::OsManaged
        }
    } else {
        EcoQosStatus::Unavailable
    }
}

#[cfg(not(windows))]
pub fn set_thread_eco_qos_opt_out(_opt_out: bool) -> EcoQosStatus {
    EcoQosStatus::Unavailable
}

/// Apply the per-thread EcoQoS opt-out across the Rayon **compute pool** (every
/// global-pool worker) plus the calling thread — the §1.2-scoped replacement for
/// the process-wide opt-out. Decode parallelism runs on these threads (and the
/// caller participates via work-stealing), so this covers the inference workers
/// without touching the process's UI/background threads. Returns the calling
/// thread's status as a representative outcome. Inert (`Unavailable`) off Windows.
pub fn set_compute_pool_eco_qos(opt_out: bool) -> EcoQosStatus {
    // `broadcast` runs the closure once on every global-pool worker thread.
    rayon::broadcast(|_| {
        let _ = set_thread_eco_qos_opt_out(opt_out);
    });
    // The calling thread also participates in Rayon work-stealing.
    set_thread_eco_qos_opt_out(opt_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eco_qos_round_trip_does_not_panic() {
        // The call must always return a status, never panic, on any host.
        let on = set_eco_qos_opt_out(true);
        let off = set_eco_qos_opt_out(false);
        #[cfg(windows)]
        {
            // Windows 10/11 supports ProcessPowerThrottling, so the opt-out applies.
            assert_eq!(on, EcoQosStatus::OptedOut);
            assert_eq!(off, EcoQosStatus::OsManaged);
        }
        #[cfg(not(windows))]
        {
            assert_eq!(on, EcoQosStatus::Unavailable);
            assert_eq!(off, EcoQosStatus::Unavailable);
        }
    }

    #[test]
    fn thread_eco_qos_round_trip_does_not_panic() {
        let on = set_thread_eco_qos_opt_out(true);
        let off = set_thread_eco_qos_opt_out(false);
        #[cfg(windows)]
        {
            assert_eq!(on, EcoQosStatus::OptedOut);
            assert_eq!(off, EcoQosStatus::OsManaged);
        }
        #[cfg(not(windows))]
        {
            assert_eq!(on, EcoQosStatus::Unavailable);
            assert_eq!(off, EcoQosStatus::Unavailable);
        }
    }

    #[test]
    fn compute_pool_eco_qos_does_not_panic() {
        // Broadcasts the per-thread opt-out across the Rayon pool; must always
        // return a status, never panic, on any host.
        let _ = set_compute_pool_eco_qos(true);
        let _ = set_compute_pool_eco_qos(false);
    }
}
