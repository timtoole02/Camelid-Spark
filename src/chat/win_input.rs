//! Windows GUI input for the computer-control agent (Phase 1).
//!
//! Synthesizes real keyboard and mouse input via Win32 `SendInput`, positions the
//! cursor with `SetCursorPos`, and reads the primary screen size. This is the
//! "blind" input layer: it drives whatever window has focus and clicks raw
//! coordinates. Finding controls *by name* (UI Automation) is Phase 2 — until
//! then the model works keyboard-first (shortcuts into focused apps) plus
//! coordinate clicks.
//!
//! Every action is surfaced as an Exec-tier tool, so the approval gate prompts
//! before any input is synthesized. `SendInput` is also subject to Windows UIPI:
//! input to a higher-integrity window is silently dropped, which we detect (the
//! returned count is short) and report rather than pretend success.

use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
    SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, INPUT_MOUSE, KEYBDINPUT, KEYEVENTF_KEYUP,
    KEYEVENTF_UNICODE, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP, MOUSEEVENTF_MIDDLEDOWN,
    MOUSEEVENTF_MIDDLEUP, MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP, MOUSEINPUT, VK_BACK,
    VK_CONTROL, VK_DELETE, VK_DOWN, VK_END, VK_ESCAPE, VK_F1, VK_HOME, VK_INSERT, VK_LEFT, VK_LWIN,
    VK_MENU, VK_NEXT, VK_PRIOR, VK_RETURN, VK_RIGHT, VK_SHIFT, VK_SPACE, VK_TAB, VK_UP,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{GetSystemMetrics, SM_CXSCREEN, SM_CYSCREEN};

/// A mouse button.
#[derive(Clone, Copy)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

impl MouseButton {
    /// (down-flag, up-flag) for `MOUSEINPUT::dwFlags`.
    fn flags(self) -> (u32, u32) {
        match self {
            MouseButton::Left => (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
            MouseButton::Right => (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
            MouseButton::Middle => (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
        }
    }

    pub fn parse(s: &str) -> Option<MouseButton> {
        match s.trim().to_ascii_lowercase().as_str() {
            "left" | "l" | "" => Some(MouseButton::Left),
            "right" | "r" => Some(MouseButton::Right),
            "middle" | "m" => Some(MouseButton::Middle),
            _ => None,
        }
    }
}

/// The primary screen's (width, height) in pixels.
pub fn screen_size() -> (i32, i32) {
    // SAFETY: GetSystemMetrics takes an index and has no preconditions.
    unsafe { (GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN)) }
}

/// Move the cursor to absolute screen coordinates.
pub fn move_cursor(x: i32, y: i32) -> Result<(), String> {
    // SAFETY: SetCursorPos takes two ints; failure is reported via the return.
    let ok = unsafe { windows_sys::Win32::UI::WindowsAndMessaging::SetCursorPos(x, y) };
    if ok != 0 {
        Ok(())
    } else {
        Err(format!("SetCursorPos({x}, {y}) failed"))
    }
}

/// Click (optionally double-click) the given button at the current cursor
/// position. Callers that want a positioned click call `move_cursor` first.
pub fn click(button: MouseButton, double: bool) -> Result<(), String> {
    let (down, up) = button.flags();
    let mut inputs = vec![mouse_input(down), mouse_input(up)];
    if double {
        inputs.push(mouse_input(down));
        inputs.push(mouse_input(up));
    }
    send(&inputs)
}

/// Type a Unicode string into the focused window (one keydown+keyup per UTF-16
/// unit via `KEYEVENTF_UNICODE`, so it is layout-independent).
pub fn type_text(text: &str) -> Result<(), String> {
    let mut inputs = Vec::with_capacity(text.len() * 2);
    for unit in text.encode_utf16() {
        inputs.push(unicode_input(unit, false));
        inputs.push(unicode_input(unit, true));
    }
    if inputs.is_empty() {
        return Ok(());
    }
    send(&inputs)
}

/// Send a key chord like `ctrl+s`, `win+r`, `alt+f4`, `enter`. Modifiers press
/// down in order, the single main key taps, then modifiers release in reverse.
pub fn press_keys(combo: &str) -> Result<(), String> {
    let parts: Vec<&str> = combo
        .split('+')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if parts.is_empty() {
        return Err("empty key combo".into());
    }
    let mut mods = Vec::new();
    let mut main = None;
    for p in &parts {
        match vk_for(p) {
            Some(Vk::Mod(vk)) => mods.push(vk),
            Some(Vk::Key(vk)) => {
                if main.is_some() {
                    return Err(format!("more than one non-modifier key in {combo:?}"));
                }
                main = Some(vk);
            }
            None => return Err(format!("unknown key {p:?} in combo {combo:?}")),
        }
    }
    let main = main.ok_or_else(|| format!("no main key in combo {combo:?}"))?;
    let mut inputs = Vec::new();
    for &m in &mods {
        inputs.push(key_input(m, false));
    }
    inputs.push(key_input(main, false));
    inputs.push(key_input(main, true));
    for &m in mods.iter().rev() {
        inputs.push(key_input(m, true));
    }
    send(&inputs)
}

// --- INPUT construction -----------------------------------------------------

fn mouse_input(flags: u32) -> INPUT {
    INPUT {
        r#type: INPUT_MOUSE,
        Anonymous: INPUT_0 {
            mi: MOUSEINPUT {
                dx: 0,
                dy: 0,
                mouseData: 0,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: 0,
            },
        },
    }
}

fn key_input(vk: u16, up: bool) -> INPUT {
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: vk,
                wScan: 0,
                dwFlags: if up { KEYEVENTF_KEYUP } else { 0 },
                time: 0,
                dwExtraInfo: 0,
            },
        },
    }
}

fn unicode_input(unit: u16, up: bool) -> INPUT {
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: 0,
                wScan: unit,
                dwFlags: KEYEVENTF_UNICODE | if up { KEYEVENTF_KEYUP } else { 0 },
                time: 0,
                dwExtraInfo: 0,
            },
        },
    }
}

fn send(inputs: &[INPUT]) -> Result<(), String> {
    // SAFETY: a contiguous, correctly-sized INPUT array + its element size, the
    // exact contract SendInput expects.
    let sent = unsafe {
        SendInput(
            inputs.len() as u32,
            inputs.as_ptr(),
            std::mem::size_of::<INPUT>() as i32,
        )
    };
    if sent as usize == inputs.len() {
        Ok(())
    } else {
        Err(format!(
            "SendInput delivered {sent}/{} events — the target window may be running at a higher \
             integrity level (run Camelid as administrator to drive it)",
            inputs.len()
        ))
    }
}

// --- key parsing ------------------------------------------------------------

enum Vk {
    Mod(u16),
    Key(u16),
}

fn vk_for(token: &str) -> Option<Vk> {
    let t = token.to_ascii_lowercase();
    match t.as_str() {
        "ctrl" | "control" => Some(Vk::Mod(VK_CONTROL)),
        "shift" => Some(Vk::Mod(VK_SHIFT)),
        "alt" | "menu" => Some(Vk::Mod(VK_MENU)),
        "win" | "super" | "cmd" | "meta" => Some(Vk::Mod(VK_LWIN)),
        "enter" | "return" => Some(Vk::Key(VK_RETURN)),
        "tab" => Some(Vk::Key(VK_TAB)),
        "esc" | "escape" => Some(Vk::Key(VK_ESCAPE)),
        "space" | "spacebar" => Some(Vk::Key(VK_SPACE)),
        "backspace" | "back" => Some(Vk::Key(VK_BACK)),
        "delete" | "del" => Some(Vk::Key(VK_DELETE)),
        "up" => Some(Vk::Key(VK_UP)),
        "down" => Some(Vk::Key(VK_DOWN)),
        "left" => Some(Vk::Key(VK_LEFT)),
        "right" => Some(Vk::Key(VK_RIGHT)),
        "home" => Some(Vk::Key(VK_HOME)),
        "end" => Some(Vk::Key(VK_END)),
        "pageup" | "pgup" => Some(Vk::Key(VK_PRIOR)),
        "pagedown" | "pgdn" => Some(Vk::Key(VK_NEXT)),
        "insert" | "ins" => Some(Vk::Key(VK_INSERT)),
        _ => {
            // Function keys f1..f12.
            if let Some(n) = t.strip_prefix('f').and_then(|s| s.parse::<u16>().ok()) {
                if (1..=12).contains(&n) {
                    return Some(Vk::Key(VK_F1 + (n - 1)));
                }
            }
            // A single alphanumeric char: its VK code is the ASCII-uppercase byte.
            let mut chars = t.chars();
            if let (Some(c), None) = (chars.next(), chars.next()) {
                if c.is_ascii_alphanumeric() {
                    return Some(Vk::Key(c.to_ascii_uppercase() as u16));
                }
            }
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_modifiers_and_keys() {
        assert!(matches!(vk_for("ctrl"), Some(Vk::Mod(_))));
        assert!(matches!(vk_for("WIN"), Some(Vk::Mod(_))));
        assert!(matches!(vk_for("enter"), Some(Vk::Key(_))));
        assert!(matches!(vk_for("f5"), Some(Vk::Key(_))));
        // 'a' maps to VK 0x41.
        assert!(matches!(vk_for("a"), Some(Vk::Key(0x41))));
        assert!(matches!(vk_for("7"), Some(Vk::Key(0x37))));
        assert!(vk_for("f13").is_none());
        assert!(vk_for("nope").is_none());
    }

    #[test]
    fn mouse_button_parses() {
        assert!(matches!(
            MouseButton::parse("left"),
            Some(MouseButton::Left)
        ));
        assert!(matches!(
            MouseButton::parse("RIGHT"),
            Some(MouseButton::Right)
        ));
        assert!(MouseButton::parse("scroll").is_none());
    }
}
