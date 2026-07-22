//! Camel mascot + inline-mode splash/glyph for `camelid chat`, with
//! `NO_COLOR` / non-TTY fallback. The full-screen TUI reuses [`CAMEL_LINES`] for
//! its empty-state art.
//!
//! Color is opt-in: emitted only when stdout is a TTY and `NO_COLOR` is unset.
//! The identical art renders with zero escape codes otherwise, so a piped or
//! redirected stream never sees a stray `\x1b[…m` sequence.

use std::io::IsTerminal;

/// The dromedary mascot, kept as a single raw const (the color path styles it by
/// substituting the three placeholders, never by scattering lines through code).
/// Backslashes are escaped (`\\`); the leading `\` is a line-continuation so the
/// banner has no blank first line.
const CAMEL_BANNER: &str = "\
           __
        .-~  ~-.___          {title}
       /  .--.     `\\        local GGUF inference · {addr}
      |  ( oo )      \\       ----------------------------
      |   `--'        |      {model}
       \\            __/
        \\   /\\   /\\    `\\
         | |  | |  |     |
         |_|  |_|  |_____|";

/// The mascot alone (no placeholders), for the TUI empty state. Same dromedary.
pub const CAMEL_LINES: &str = "\
           __
        .-~  ~-.___
       /  .--.     `\\
      |  ( oo )      \\
      |   `--'        |
       \\            __/
        \\   /\\   /\\    `\\
         | |  | |  |     |
         |_|  |_|  |_____|";

/// One-line marker printed before each assistant turn in inline mode — a little
/// camel face in the same sandy tan as the splash.
const TURN_GLYPH: &str = "(oo)>";

// Warm sandy tan (ANSI 256), a bold/bright sand for the title, and dim for the
// secondary lines. `\x1b[0m` fully resets; intensity resets stay inside the tan.
const TAN: &str = "\x1b[38;5;179m";
const TITLE: &str = "\x1b[1;38;5;223m";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

/// True when colored output is both possible (stdout is a TTY) and wanted
/// (`NO_COLOR` unset). Used by every styled inline surface so the fallback is
/// consistent.
pub fn color_enabled() -> bool {
    std::env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal()
}

/// Render the inline splash. `version` is the binary version, `addr` the resolved
/// server address, `model_line` the active-model status line.
pub fn splash(version: &str, addr: &str, model_line: &str) -> String {
    if color_enabled() {
        // Each styled segment restores `TAN` afterward so the rest of its line
        // stays camel-colored.
        let title = format!("{TITLE}CAMELID  {version}{RESET}{TAN}");
        let addr = format!("{DIM}{addr}{RESET}{TAN}");
        let model = format!("{DIM}{model_line}{RESET}{TAN}");
        let body = CAMEL_BANNER
            .replace("{title}", &title)
            .replace("{addr}", &addr)
            .replace("{model}", &model);
        format!("{TAN}{body}{RESET}")
    } else {
        CAMEL_BANNER
            .replace("{title}", &format!("CAMELID  {version}"))
            .replace("{addr}", addr)
            .replace("{model}", model_line)
    }
}

/// The assistant-turn prefix (glyph + trailing space), colored when enabled.
pub fn turn_prefix() -> String {
    if color_enabled() {
        format!("{TAN}{TURN_GLYPH}{RESET} ")
    } else {
        format!("{TURN_GLYPH} ")
    }
}

/// Dim a short notice line (e.g. "history reset"), or pass it through verbatim
/// when color is off.
pub fn dim(text: &str) -> String {
    if color_enabled() {
        format!("{DIM}{text}{RESET}")
    } else {
        text.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholders_are_substituted_and_camel_survives() {
        let out = splash("v9.9.9", "127.0.0.1:8181", "no model loaded");
        assert!(out.contains("v9.9.9"));
        assert!(out.contains("127.0.0.1:8181"));
        assert!(out.contains("no model loaded"));
        // The dromedary's eyes are intact.
        assert!(out.contains("( oo )"));
        // No unsubstituted placeholders leak.
        assert!(!out.contains("{title}"));
        assert!(!out.contains("{addr}"));
        assert!(!out.contains("{model}"));
    }

    #[test]
    fn camel_lines_has_no_placeholders() {
        assert!(!CAMEL_LINES.contains('{'));
        assert!(CAMEL_LINES.contains("( oo )"));
    }
}
