//! Color themes for the TUI. Each theme maps a small set of roles to ANSI-256
//! colors; widgets and the markdown renderer ask the active theme for styles so
//! a single `/theme` switch restyles everything.

use ratatui::style::{Color, Modifier, Style};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Theme {
    Sandstorm,
    Mono,
    Ocean,
    Nord,
}

impl Theme {
    pub const ALL: &'static [Theme] = &[Theme::Sandstorm, Theme::Mono, Theme::Ocean, Theme::Nord];

    pub fn name(self) -> &'static str {
        match self {
            Theme::Sandstorm => "sandstorm",
            Theme::Mono => "mono",
            Theme::Ocean => "ocean",
            Theme::Nord => "nord",
        }
    }

    pub fn from_name(s: &str) -> Option<Theme> {
        Theme::ALL
            .iter()
            .copied()
            .find(|t| t.name().eq_ignore_ascii_case(s))
    }

    pub fn next(self) -> Theme {
        let i = Theme::ALL.iter().position(|&t| t == self).unwrap_or(0);
        Theme::ALL[(i + 1) % Theme::ALL.len()]
    }

    // --- role colors ------------------------------------------------------
    pub fn primary(self) -> Color {
        match self {
            Theme::Sandstorm => Color::Indexed(179),
            Theme::Mono => Color::Indexed(250),
            Theme::Ocean => Color::Indexed(37),
            Theme::Nord => Color::Indexed(110),
        }
    }
    pub fn title(self) -> Color {
        match self {
            Theme::Sandstorm => Color::Indexed(223),
            Theme::Mono => Color::Indexed(255),
            Theme::Ocean => Color::Indexed(51),
            Theme::Nord => Color::Indexed(153),
        }
    }
    pub fn dim(self) -> Color {
        Color::Indexed(245)
    }
    pub fn user(self) -> Color {
        match self {
            Theme::Sandstorm => Color::Indexed(110),
            Theme::Mono => Color::Indexed(252),
            Theme::Ocean => Color::Indexed(75),
            Theme::Nord => Color::Indexed(144),
        }
    }
    pub fn accent(self) -> Color {
        match self {
            Theme::Sandstorm => Color::Indexed(215),
            Theme::Mono => Color::Indexed(248),
            Theme::Ocean => Color::Indexed(80),
            Theme::Nord => Color::Indexed(180),
        }
    }
    pub fn code_fg(self) -> Color {
        match self {
            Theme::Sandstorm => Color::Indexed(151),
            Theme::Mono => Color::Indexed(252),
            Theme::Ocean => Color::Indexed(115),
            Theme::Nord => Color::Indexed(108),
        }
    }
    pub fn code_bg(self) -> Color {
        Color::Indexed(236)
    }
    pub fn highlight_bg(self) -> Color {
        Color::Indexed(238)
    }

    // --- styles -----------------------------------------------------------
    pub fn text(self) -> Style {
        Style::default()
    }
    pub fn heading(self) -> Style {
        Style::default()
            .fg(self.title())
            .add_modifier(Modifier::BOLD)
    }
    pub fn quote(self) -> Style {
        Style::default()
            .fg(self.dim())
            .add_modifier(Modifier::ITALIC)
    }
    pub fn bullet(self) -> Style {
        Style::default().fg(self.accent())
    }
    pub fn code(self) -> Style {
        Style::default().fg(self.code_fg()).bg(self.code_bg())
    }
    pub fn code_inline(self) -> Style {
        Style::default().fg(self.code_fg())
    }
    pub fn code_fence(self) -> Style {
        Style::default().fg(self.dim())
    }
}
