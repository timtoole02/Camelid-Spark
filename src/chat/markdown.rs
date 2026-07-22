//! A small, dependency-free Markdown renderer for assistant replies: fenced code
//! blocks, headings, bullets, block quotes, and inline `code` / **bold** /
//! *italic*. Produces theme-styled, width-wrapped [`Line`]s for ratatui.

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use super::theme::Theme;

/// Render `text` into styled lines wrapped to `width`.
pub fn render(text: &str, width: usize, th: Theme) -> Vec<Line<'static>> {
    let width = width.max(1);
    let mut out = Vec::new();
    let mut in_code = false;
    for raw in text.split('\n') {
        let trimmed = raw.trim_start();
        if trimmed.starts_with("```") {
            in_code = !in_code;
            out.push(Line::from(Span::styled(raw.to_string(), th.code_fence())));
            continue;
        }
        if in_code {
            out.push(Line::from(Span::styled(raw.to_string(), th.code())));
            continue;
        }
        if let Some(level) = heading_level(trimmed) {
            let content = trimmed[level..].trim_start();
            wrap_spans(&mut out, None, inline(content, th.heading(), th), width);
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("> ") {
            let prefix = Some(("┃ ".to_string(), th.quote()));
            wrap_spans(&mut out, prefix, inline(rest, th.quote(), th), width);
            continue;
        }
        if let Some(rest) = bullet(trimmed) {
            let prefix = Some(("• ".to_string(), th.bullet()));
            wrap_spans(&mut out, prefix, inline(rest, th.text(), th), width);
            continue;
        }
        wrap_spans(&mut out, None, inline(raw, th.text(), th), width);
    }
    out
}

fn heading_level(s: &str) -> Option<usize> {
    let hashes = s.chars().take_while(|&c| c == '#').count();
    if (1..=6).contains(&hashes) && s.chars().nth(hashes) == Some(' ') {
        Some(hashes)
    } else {
        None
    }
}

fn bullet(s: &str) -> Option<&str> {
    for marker in ["- ", "* ", "+ "] {
        if let Some(rest) = s.strip_prefix(marker) {
            return Some(rest);
        }
    }
    None
}

/// Parse one line of inline markdown into styled (text, style) segments.
fn inline(s: &str, base: Style, th: Theme) -> Vec<(String, Style)> {
    let chars: Vec<char> = s.chars().collect();
    let mut spans: Vec<(String, Style)> = Vec::new();
    let mut cur = String::new();
    let mut bold = false;
    let mut italic = false;
    let styled = |base: Style, bold: bool, italic: bool| {
        let mut st = base;
        if bold {
            st = st.add_modifier(Modifier::BOLD);
        }
        if italic {
            st = st.add_modifier(Modifier::ITALIC);
        }
        st
    };
    let flush = |cur: &mut String, spans: &mut Vec<(String, Style)>, st: Style| {
        if !cur.is_empty() {
            spans.push((std::mem::take(cur), st));
        }
    };
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '`' {
            flush(&mut cur, &mut spans, styled(base, bold, italic));
            if let Some(end) = chars[i + 1..].iter().position(|&c| c == '`') {
                let code: String = chars[i + 1..i + 1 + end].iter().collect();
                spans.push((code, th.code_inline()));
                i = i + 1 + end + 1;
                continue;
            }
            cur.push('`');
            i += 1;
            continue;
        }
        if c == '*' && chars.get(i + 1) == Some(&'*') {
            flush(&mut cur, &mut spans, styled(base, bold, italic));
            bold = !bold;
            i += 2;
            continue;
        }
        if c == '*' || c == '_' {
            flush(&mut cur, &mut spans, styled(base, bold, italic));
            italic = !italic;
            i += 1;
            continue;
        }
        cur.push(c);
        i += 1;
    }
    flush(&mut cur, &mut spans, styled(base, bold, italic));
    if spans.is_empty() {
        spans.push((String::new(), base));
    }
    spans
}

/// Reflow styled segments into width-bounded lines, with an optional first-line
/// prefix (continuation lines are indented to match).
fn wrap_spans(
    out: &mut Vec<Line<'static>>,
    prefix: Option<(String, Style)>,
    segments: Vec<(String, Style)>,
    width: usize,
) {
    let indent = prefix.as_ref().map(|(t, _)| t.chars().count()).unwrap_or(0);
    let avail = width.saturating_sub(indent).max(1);

    // Flatten to styled words (single spaces collapse; code stays intact).
    let mut words: Vec<(String, Style)> = Vec::new();
    for (text, style) in segments {
        for w in text.split(' ') {
            if w.is_empty() {
                continue;
            }
            if w.chars().count() > avail {
                for chunk in hard_split(w, avail) {
                    words.push((chunk, style));
                }
            } else {
                words.push((w.to_string(), style));
            }
        }
    }

    let mut first = true;
    let mut line: Vec<Span<'static>> = Vec::new();
    let mut col = 0usize;
    let start_line =
        |line: &mut Vec<Span<'static>>, first: bool, prefix: &Option<(String, Style)>| {
            if first {
                if let Some((t, s)) = prefix {
                    line.push(Span::styled(t.clone(), *s));
                }
            } else if indent > 0 {
                line.push(Span::raw(" ".repeat(indent)));
            }
        };

    if words.is_empty() {
        let mut l: Vec<Span<'static>> = Vec::new();
        start_line(&mut l, true, &prefix);
        out.push(Line::from(l));
        return;
    }

    start_line(&mut line, first, &prefix);
    for (word, style) in words {
        let wlen = word.chars().count();
        if col == 0 {
            line.push(Span::styled(word, style));
            col = wlen;
        } else if col + 1 + wlen <= avail {
            line.push(Span::raw(" "));
            line.push(Span::styled(word, style));
            col += 1 + wlen;
        } else {
            out.push(Line::from(std::mem::take(&mut line)));
            first = false;
            start_line(&mut line, first, &prefix);
            line.push(Span::styled(word, style));
            col = wlen;
        }
    }
    out.push(Line::from(line));
}

fn hard_split(word: &str, width: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut cur = String::new();
    let mut n = 0;
    for ch in word.chars() {
        if n == width {
            chunks.push(std::mem::take(&mut cur));
            n = 0;
        }
        cur.push(ch);
        n += 1;
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_without_panicking_and_wraps() {
        let th = Theme::Sandstorm;
        let md = "# Title\n\nSome **bold** and `code` and *italic* text that is quite long and should wrap nicely.\n\n- one\n- two\n\n```\nfn main() {}\n```\n> a quote";
        let lines = render(md, 24, th);
        assert!(lines.len() > 6);
        // No line should be empty of spans structurally (Line always constructed).
        for l in &lines {
            // width respected (allow a little slack for prefixes)
            let w: usize = l.spans.iter().map(|s| s.content.chars().count()).sum();
            assert!(w <= 26, "line too wide: {w}");
        }
    }

    #[test]
    fn inline_code_survives() {
        let lines = render("use `cargo build` now", 80, Theme::Mono);
        let joined: String = lines[0].spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(joined.contains("cargo build"));
    }
}
