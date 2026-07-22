//! Windows console init for the line-mode renderers (inline + agent).
//!
//! The full-screen TUI gets ANSI/VT processing for free via crossterm, but the
//! inline and agent line renderers write raw ANSI + UTF-8 straight to stdout. A
//! classic conhost window starts with its OEM output code page (US default 437,
//! NOT UTF-8) and — depending on the host — without ANSI escape processing. So
//! without this, the sandy splash is fine (pure ASCII) but colors can show as
//! literal `\x1b[…m` and glyphs like `› └ · ● ▸ ✓` become mojibake (a multi-byte
//! UTF-8 sequence decoded as several CP437 chars). macOS/Linux terminals handle
//! both natively, which is why this whole step is Windows-only.
//!
//! Best-effort and idempotent: each step is skipped when the handle is not a
//! console (output redirected to a pipe/file), so the non-TTY fallback — which
//! emits no ANSI anyway — is never disturbed.

use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
use windows_sys::Win32::System::Console::{
    GetConsoleMode, GetStdHandle, SetConsoleCP, SetConsoleMode, SetConsoleOutputCP,
    ENABLE_VIRTUAL_TERMINAL_PROCESSING, STD_ERROR_HANDLE, STD_OUTPUT_HANDLE,
};

/// UTF-8 code page identifier.
const CP_UTF8: u32 = 65001;

/// Enable ANSI escape processing + UTF-8 I/O on the current console so the inline
/// and agent line renderers look the way they do on macOS/Linux. Safe to call
/// unconditionally — a redirected (non-console) handle is left untouched.
pub fn init() {
    // SAFETY: these calls take only a code-page id / std-handle id and a local
    // out-param; they have no preconditions and signal failure via their return
    // value, which we intentionally ignore (best-effort).
    unsafe {
        // UTF-8 output so multi-byte glyphs render instead of OEM mojibake, and
        // UTF-8 input so pasted/typed non-ASCII reaches the model intact.
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
    }
    enable_vt(STD_OUTPUT_HANDLE);
    enable_vt(STD_ERROR_HANDLE);
}

/// Add `ENABLE_VIRTUAL_TERMINAL_PROCESSING` to one std handle's console mode. A
/// no-op when the handle is invalid or not a console (e.g. redirected to a file).
fn enable_vt(std_handle: u32) {
    // SAFETY: GetStdHandle returns a process-owned handle (or an invalid sentinel
    // we check for); GetConsoleMode/SetConsoleMode are called only on a verified
    // console handle with a valid local mode out-param.
    unsafe {
        let handle = GetStdHandle(std_handle);
        if handle.is_null() || handle == INVALID_HANDLE_VALUE {
            return;
        }
        let mut mode = 0u32;
        if GetConsoleMode(handle, &mut mode) == 0 {
            // Not a console (redirected): the line renderers emit no ANSI here.
            return;
        }
        SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    }
}
