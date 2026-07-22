//! Safe-boot sentinel (§4) + kill-switches (§1.3) for GAIT.
//!
//! Persisted, auto-applied execution state is the most dangerous property of any
//! perf system: a configuration that crashes, freezes, or wedges the host will
//! reload and crash again on the next launch — a crash *loop* the customer cannot
//! escape without clearing the cache by hand. This module makes that structurally
//! impossible.
//!
//! The protocol (all state under the gait dir, `%LOCALAPPDATA%\Camelid\gait\`):
//!
//! - **`.applying`** — a marker written *before* a gait/substrate is applied,
//!   recording `{ gait_key, layers, pid, utc }`. It is cleared once the gait has
//!   proven healthy (a real decode completed via [`mark_healthy`]) or on an
//!   orderly exit ([`clean_shutdown`]). If it is still present at the *next*
//!   launch, the previous run applied a gait and never cleared it → an unclean
//!   exit → that gait is the suspect.
//! - **`.quarantine/`** — receipts disabled by the sentinel, kept for diagnosis
//!   and never loaded again.
//! - **`DISABLE`** — an unconditional kill-file: while it exists the selector
//!   serves the baseline path and calibration is suppressed.
//!
//! [`reconcile_on_startup`] runs once at process start: if it finds a stale
//! `.applying`, it strikes (and, past the threshold, quarantines) the offending
//! `gait_key`, deletes the marker, and lets the engine boot the proven baseline.
//! A crash *loop* therefore degrades to a single crash followed by automatic
//! self-heal.
//!
//! Everything here is best-effort and fail-safe: any I/O error biases toward
//! quarantine / baseline (the safe direction), never toward re-applying a suspect
//! gait. The module is inert unless the `CAMELID_GAIT` bring-up gate is on and a
//! gait was actually applied, so the default path is byte-identical to today.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

/// In-progress apply marker.
const APPLYING: &str = ".applying";
/// Subdirectory holding quarantined (permanently disabled) receipts.
const QUARANTINE_DIR: &str = ".quarantine";
/// Unconditional kill-file: while present, baseline serves and calibration is off.
const DISABLE_FILE: &str = "DISABLE";
/// Per-key strike-count sidecar suffix (`<sanitized_key>.strikes`).
const STRIKES_SUFFIX: &str = ".strikes";

/// Quarantine a `gait_key` after this many unclean exits. The spec sets the
/// default to **1** (quarantine on the first unclean exit) and forbids a value
/// above 2 — a higher tolerance would let a genuinely host-wedging gait keep
/// getting re-applied. Enforced at compile time.
const QUARANTINE_STRIKE_THRESHOLD: u32 = 1;
const _: () = assert!(QUARANTINE_STRIKE_THRESHOLD >= 1 && QUARANTINE_STRIKE_THRESHOLD <= 2);

/// The marker written before a gait/substrate is applied. Its presence at the
/// next launch is the unclean-exit signal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApplyingMarker {
    /// The gait being applied (so a suspect can be quarantined by key).
    pub gait_key: String,
    /// Which layers were in flight (`"gait"` and/or `"substrate"`), so the
    /// offending layer is identifiable for diagnosis.
    pub layers: Vec<String>,
    /// PID of the applying process — distinguishes our own marker from a stale
    /// one when clearing.
    pub pid: u32,
    /// Wall-clock seconds since the Unix epoch at apply time (diagnosis only).
    pub utc_epoch_secs: u64,
}

/// The location + PID of this process's in-progress marker, set by
/// [`begin_apply`] and consumed by [`mark_healthy`] / [`clean_shutdown`]. A
/// process applies at most one gait, so a single slot suffices.
static ARMED: Mutex<Option<(PathBuf, u32)>> = Mutex::new(None);

/// Outcome of the startup reconciliation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StartupReconcile {
    /// No in-progress marker — the previous run (if any) exited cleanly.
    Clean,
    /// A stale marker was found: the previous run applied a gait and never
    /// cleared it. `gait_key` is the suspect (None if the marker was
    /// unparseable); `quarantined` is true if its receipt was moved to
    /// `.quarantine/` this launch (false if it merely took a strike).
    UncleanExit {
        gait_key: Option<String>,
        quarantined: bool,
    },
}

/// §1.3 kill-file: true when `DISABLE` exists under `dir`. While true the caller
/// must serve the baseline path unconditionally — no cached profile, no
/// substrate, no calibration.
pub fn disable_present(dir: &Path) -> bool {
    dir.join(DISABLE_FILE).exists()
}

/// §4 startup reconciliation. Detects a stale `.applying` marker left by a prior
/// unclean exit, strikes/quarantines the suspect `gait_key`, removes the marker,
/// and returns what happened. Pure with respect to `dir` (no globals), so it is
/// directly testable. Best-effort: I/O failures bias toward quarantine.
///
/// Call this **once**, early at process start, before any gait is applied — at
/// that point any `.applying` present is necessarily from a previous process.
pub fn reconcile_on_startup(dir: &Path) -> StartupReconcile {
    let marker_path = dir.join(APPLYING);
    let raw = match std::fs::read_to_string(&marker_path) {
        Ok(raw) => raw,
        // No marker (or it cannot be read) → nothing to reconcile.
        Err(_) => return StartupReconcile::Clean,
    };

    // A marker exists at startup → the previous run applied a gait/substrate and
    // never cleared it → treat it as the cause of an unclean exit.
    let gait_key = serde_json::from_str::<ApplyingMarker>(&raw)
        .ok()
        .map(|m| m.gait_key);

    let mut quarantined = false;
    if let Some(ref key) = gait_key {
        let strikes = bump_strikes(dir, key);
        if strikes >= QUARANTINE_STRIKE_THRESHOLD {
            quarantined = quarantine_receipt(dir, key);
            // The suspect is gone; reset its strike count.
            let _ = std::fs::remove_file(strikes_path(dir, key));
        }
    }
    let _ = std::fs::remove_file(&marker_path);

    eprintln!(
        "[gait] safe-boot: unclean exit detected (prior run left {APPLYING}); \
         key={gait_key:?} quarantined={quarantined} -> booting baseline"
    );
    StartupReconcile::UncleanExit {
        gait_key,
        quarantined,
    }
}

/// Write the in-progress marker and arm it for clearing. Call immediately
/// **before** applying a gait's profile/substrate, so a crash during apply or
/// early use is caught on the next launch. Best-effort.
pub fn begin_apply(dir: &Path, gait_key: &str, layers: &[&str]) {
    let _ = std::fs::create_dir_all(dir);
    let marker = ApplyingMarker {
        gait_key: gait_key.to_string(),
        layers: layers.iter().map(|s| s.to_string()).collect(),
        pid: std::process::id(),
        utc_epoch_secs: now_epoch_secs(),
    };
    let path = dir.join(APPLYING);
    if let Ok(json) = serde_json::to_string_pretty(&marker) {
        let _ = std::fs::write(&path, json);
    }
    if let Ok(mut armed) = ARMED.lock() {
        *armed = Some((path, marker.pid));
    }
}

/// Clear the in-progress marker because the applied gait has proven healthy — a
/// real decode completed without wedging the host. Idempotent and cheap after
/// the first call (the armed slot is taken), so it is safe to call on every
/// generation. A no-op if no gait was applied this process.
pub fn mark_healthy() {
    let armed = ARMED.lock().ok().and_then(|mut slot| slot.take());
    if let Some((path, pid)) = armed {
        clear_if_ours(&path, pid);
    }
}

/// Clear the in-progress marker on an orderly process exit. Belt-and-suspenders
/// alongside [`mark_healthy`]: covers short-lived commands that exit before any
/// decode, and defensively removes a residual marker this process owns.
pub fn clean_shutdown(dir: &Path) {
    mark_healthy();
    clear_if_ours(&dir.join(APPLYING), std::process::id());
}

/// Move a `gait_key`'s receipt into `.quarantine/` so it is never loaded again.
/// Returns true if a receipt was disabled. If the quarantine move cannot be
/// completed, the receipt is removed outright — the safety goal is "never
/// re-apply this suspect", and diagnosis is secondary to that.
pub fn quarantine_receipt(dir: &Path, gait_key: &str) -> bool {
    let src = dir.join(super::key_filename(gait_key));
    if !src.exists() {
        return false;
    }
    let qdir = dir.join(QUARANTINE_DIR);
    if std::fs::create_dir_all(&qdir).is_err() {
        let _ = std::fs::remove_file(&src);
        return true;
    }
    let dst = qdir.join(super::key_filename(gait_key));
    match std::fs::rename(&src, &dst) {
        Ok(()) => true,
        // Could not move (e.g. cross-volume) → remove so it cannot be reloaded.
        Err(_) => {
            let _ = std::fs::remove_file(&src);
            true
        }
    }
}

/// Increment and persist the unclean-exit strike count for `gait_key`, returning
/// the new count. An unreadable/garbled sidecar counts as zero prior strikes; a
/// write failure still returns the incremented value (so a cache we cannot write
/// biases toward quarantine on the first strike, the safe direction).
fn bump_strikes(dir: &Path, gait_key: &str) -> u32 {
    let path = strikes_path(dir, gait_key);
    let prev = std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .unwrap_or(0);
    let next = prev.saturating_add(1);
    let _ = std::fs::write(&path, next.to_string());
    next
}

/// Remove `path` only if it is this process's marker (PID matches). A marker we
/// cannot parse is left in place for the next launch's reconciliation rather than
/// risk clearing another process's marker.
fn clear_if_ours(path: &Path, our_pid: u32) {
    if let Ok(raw) = std::fs::read_to_string(path) {
        let ours = serde_json::from_str::<ApplyingMarker>(&raw)
            .map(|m| m.pid == our_pid)
            .unwrap_or(false);
        if ours {
            let _ = std::fs::remove_file(path);
        }
    }
}

fn strikes_path(dir: &Path, gait_key: &str) -> PathBuf {
    dir.join(format!("{}{STRIKES_SUFFIX}", gait_key.replace(':', "_")))
}

fn now_epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("camelid_sentinel_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn reconcile_is_clean_with_no_marker() {
        let dir = temp_dir("clean");
        assert_eq!(reconcile_on_startup(&dir), StartupReconcile::Clean);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn begin_apply_writes_a_parseable_marker() {
        let dir = temp_dir("begin");
        begin_apply(&dir, "modelhash:machinehash", &["gait", "substrate"]);
        let raw = std::fs::read_to_string(dir.join(APPLYING)).unwrap();
        let marker: ApplyingMarker = serde_json::from_str(&raw).unwrap();
        assert_eq!(marker.gait_key, "modelhash:machinehash");
        assert_eq!(marker.layers, vec!["gait", "substrate"]);
        assert_eq!(marker.pid, std::process::id());
        // Our own marker clears on a healthy/clean exit.
        clean_shutdown(&dir);
        assert!(!dir.join(APPLYING).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn strike_threshold_default_quarantines_on_first_unclean_exit() {
        // The compile-time default is 1, so a single stale marker quarantines.
        let dir = temp_dir("strike");
        let key = "k1:k2";
        std::fs::write(dir.join(super::super::key_filename(key)), "{}").unwrap();
        std::fs::write(
            dir.join(APPLYING),
            serde_json::to_string(&ApplyingMarker {
                gait_key: key.to_string(),
                layers: vec!["gait".to_string()],
                pid: 4242,
                utc_epoch_secs: 0,
            })
            .unwrap(),
        )
        .unwrap();

        match reconcile_on_startup(&dir) {
            StartupReconcile::UncleanExit {
                gait_key,
                quarantined,
            } => {
                assert_eq!(gait_key.as_deref(), Some(key));
                assert!(
                    quarantined,
                    "default threshold 1 must quarantine immediately"
                );
            }
            other => panic!("expected UncleanExit, got {other:?}"),
        }
        assert!(!dir.join(APPLYING).exists(), "marker must be cleared");
        assert!(
            dir.join(QUARANTINE_DIR)
                .join(super::super::key_filename(key))
                .exists(),
            "receipt must be moved into .quarantine/"
        );
        assert!(
            !dir.join(super::super::key_filename(key)).exists(),
            "receipt must no longer be loadable from the live store"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn disable_present_tracks_the_kill_file() {
        let dir = temp_dir("disable");
        assert!(!disable_present(&dir));
        std::fs::write(dir.join(DISABLE_FILE), "").unwrap();
        assert!(disable_present(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
