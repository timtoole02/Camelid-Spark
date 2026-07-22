//! The slash-command registry — the single source of truth for the `/` command
//! palette, `/help`, and command dispatch in both front ends.

/// One slash command's metadata.
pub struct Cmd {
    pub name: &'static str,
    pub args: &'static str,
    pub desc: &'static str,
    pub aliases: &'static [&'static str],
}

/// Every command the app exposes. Order is the palette's default order.
pub const COMMANDS: &[Cmd] = &[
    Cmd {
        name: "models",
        args: "",
        desc: "open the model browser (loaded + downloadable)",
        aliases: &[],
    },
    Cmd {
        name: "switch",
        args: "",
        desc: "switch instantly between models already loaded",
        aliases: &["loaded"],
    },
    Cmd {
        name: "model",
        args: "<id>",
        desc: "switch to a model by id",
        aliases: &[],
    },
    Cmd {
        name: "set",
        args: "<name> <value>",
        desc: "tune sampling: temperature top_p top_k max_tokens seed stream",
        aliases: &[],
    },
    Cmd {
        name: "system",
        args: "<text>",
        desc: "set the system prompt (applies next turn)",
        aliases: &["sys"],
    },
    Cmd {
        name: "retry",
        args: "",
        desc: "regenerate the last reply",
        aliases: &["regenerate", "regen"],
    },
    Cmd {
        name: "stop",
        args: "",
        desc: "cancel the in-flight generation",
        aliases: &["cancel"],
    },
    Cmd {
        name: "copy",
        args: "",
        desc: "copy the last reply to the clipboard",
        aliases: &["yank"],
    },
    Cmd {
        name: "reset",
        args: "",
        desc: "clear the conversation, keep the model",
        aliases: &["clear", "new"],
    },
    Cmd {
        name: "save",
        args: "[path]",
        desc: "save the session (settings + transcript) to JSON",
        aliases: &[],
    },
    Cmd {
        name: "load",
        args: "<path>",
        desc: "load a saved session",
        aliases: &[],
    },
    Cmd {
        name: "info",
        args: "",
        desc: "active model + settings detail",
        aliases: &[],
    },
    Cmd {
        name: "tokens",
        args: "",
        desc: "last response's token counts",
        aliases: &[],
    },
    Cmd {
        name: "theme",
        args: "[name]",
        desc: "cycle or pick a color theme",
        aliases: &[],
    },
    Cmd {
        name: "sidebar",
        args: "",
        desc: "toggle the settings sidebar",
        aliases: &[],
    },
    Cmd {
        name: "pull",
        args: "<alias>",
        desc: "download a supported model",
        aliases: &[],
    },
    Cmd {
        name: "agent",
        args: "",
        desc: "how to start the sandboxed tool-calling agent",
        aliases: &[],
    },
    Cmd {
        name: "help",
        args: "",
        desc: "show keys & commands",
        aliases: &["?"],
    },
    Cmd {
        name: "exit",
        args: "",
        desc: "quit the app",
        aliases: &["quit"],
    },
];

/// Resolve a typed command name (or alias) to its canonical [`Cmd`].
pub fn resolve(name: &str) -> Option<&'static Cmd> {
    let name = name.to_ascii_lowercase();
    COMMANDS
        .iter()
        .find(|c| c.name == name || c.aliases.contains(&name.as_str()))
}

/// Palette matches for the text typed after `/`. Prefix matches rank above
/// substring matches; an empty query returns everything in registry order.
pub fn matches(query: &str) -> Vec<&'static Cmd> {
    let q = query.trim().to_ascii_lowercase();
    if q.is_empty() {
        return COMMANDS.iter().collect();
    }
    let mut prefix = Vec::new();
    let mut contains = Vec::new();
    for cmd in COMMANDS {
        let names = std::iter::once(cmd.name).chain(cmd.aliases.iter().copied());
        if names.clone().any(|n| n.starts_with(&q)) {
            prefix.push(cmd);
        } else if names.clone().any(|n| n.contains(&q))
            || cmd.desc.to_ascii_lowercase().contains(&q)
        {
            contains.push(cmd);
        }
    }
    prefix.extend(contains);
    prefix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_handles_aliases() {
        assert_eq!(resolve("quit").unwrap().name, "exit");
        assert_eq!(resolve("regen").unwrap().name, "retry");
        assert_eq!(resolve("CLEAR").unwrap().name, "reset");
        assert!(resolve("nope").is_none());
    }

    #[test]
    fn matches_prefix_ranks_first() {
        let m = matches("se");
        assert!(!m.is_empty());
        assert_eq!(m[0].name, "set"); // prefix beats substring (e.g. "reset")
        assert!(matches("").len() == COMMANDS.len());
        // description search finds by keyword
        assert!(matches("clipboard").iter().any(|c| c.name == "copy"));
    }
}
