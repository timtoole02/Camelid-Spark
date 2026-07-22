//! Live Hugging Face GGUF browse for the experimental lane.
//!
//! This module is a **network concern only** — it discovers GGUF files on the Hub
//! so the Models tab can offer to download them. It makes NO claim about whether a
//! file will load or run: the `architecture`/`quant` it returns are best-effort
//! guesses from the filename (advisory, never authoritative). The authoritative
//! architecture is read from real GGUF metadata at load time, never here. Browse
//! results therefore always land in the experimental group with a permanent
//! "unverified, no parity claim" marker — discovering a file is not endorsing it.
//!
//! HTTP egress reuses the same `curl` subprocess pattern as `catalog.rs` (no new
//! dependency); every call tolerates offline by returning a typed error rather than
//! panicking, so the catalog endpoint can degrade to curated-only.

use std::path::PathBuf;

/// One GGUF file discovered on the Hub. `size_bytes` is the real content size
/// (LFS-aware). `architecture`/`quant` are filename guesses — advisory only, empty
/// when nothing matched.
#[derive(Debug, Clone, serde::Serialize)]
pub struct HfGgufFile {
    pub repo_id: String,
    pub filename: String,
    pub size_bytes: u64,
    pub downloads: u64,
    pub likes: u64,
    /// Filename-guessed architecture (advisory, NOT authoritative). Empty if unknown.
    pub architecture: String,
    /// Filename-guessed quant (advisory, NOT authoritative). Empty if unknown.
    pub quant: String,
}

/// A page of search results plus the opaque cursor for the next page (`None` when
/// the Hub advertised no further pages).
#[derive(Debug, Clone)]
pub struct HfSearchPage {
    pub files: Vec<HfGgufFile>,
    pub next_cursor: Option<String>,
}

/// Search the Hugging Face Hub for GGUF repos matching `query` and enumerate their
/// `*.gguf` files with real sizes. `limit` bounds the number of repos inspected;
/// `cursor` resumes a previous page. Network failure → typed `Err` (never panic),
/// so callers can fall back to curated-only.
pub async fn search_gguf(
    query: &str,
    limit: usize,
    cursor: Option<&str>,
) -> anyhow::Result<HfSearchPage> {
    let query = query.to_string();
    let cursor = cursor.map(|c| c.to_string());
    // curl is blocking; keep it off the async executor.
    tokio::task::spawn_blocking(move || search_gguf_blocking(&query, limit, cursor.as_deref()))
        .await
        .map_err(|err| anyhow::anyhow!("hugging face search task panicked: {err}"))?
}

struct RepoMeta {
    id: String,
    downloads: u64,
    likes: u64,
}

fn search_gguf_blocking(
    query: &str,
    limit: usize,
    cursor: Option<&str>,
) -> anyhow::Result<HfSearchPage> {
    let limit = limit.clamp(1, 50);

    // 1. Ask the Hub for repos matching the query (GGUF filter, full record so we
    //    pick up download/like counts). `cursor` paginates.
    let mut url = format!(
        "https://huggingface.co/api/models?search={}&filter=gguf&limit={}&full=true",
        urlencode(query),
        limit
    );
    if let Some(c) = cursor {
        url.push_str("&cursor=");
        url.push_str(&urlencode(c));
    }

    let (body, headers) = curl_get_with_headers(&url)?;
    let models: serde_json::Value = serde_json::from_slice(&body)
        .map_err(|err| anyhow::anyhow!("could not parse hugging face search response: {err}"))?;
    let next_cursor = parse_next_cursor(&headers);

    let repos: Vec<RepoMeta> = models
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|m| {
                    let id = m.get("id")?.as_str()?.to_string();
                    Some(RepoMeta {
                        downloads: m.get("downloads").and_then(|v| v.as_u64()).unwrap_or(0),
                        likes: m.get("likes").and_then(|v| v.as_u64()).unwrap_or(0),
                        id,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    // 2. For each repo, read its file tree for the real `*.gguf` sizes. One repo
    //    failing (private, renamed, transient) must not drop the whole page.
    let mut files = Vec::new();
    for repo in &repos {
        if let Ok(mut repo_files) = repo_gguf_files(repo) {
            files.append(&mut repo_files);
        }
    }

    Ok(HfSearchPage { files, next_cursor })
}

/// Enumerate the top-level `*.gguf` files in a repo with LFS-aware sizes, mirroring
/// the `remote_size()` logic in `catalog.rs`. Only top-level files are returned:
/// the downloader writes to `models/<filename>` and the local scan globs
/// `models/*.gguf`, so a nested path would download but never surface as a local
/// model. Sharded/subdir GGUFs are therefore skipped at browse time.
fn repo_gguf_files(repo: &RepoMeta) -> anyhow::Result<Vec<HfGgufFile>> {
    let url = format!(
        "https://huggingface.co/api/models/{}/tree/main?recursive=1",
        repo.id
    );
    let (body, _) = curl_get_with_headers(&url)?;
    let tree: serde_json::Value = serde_json::from_slice(&body)
        .map_err(|err| anyhow::anyhow!("could not parse hugging face tree response: {err}"))?;

    let mut out = Vec::new();
    let Some(entries) = tree.as_array() else {
        return Ok(out);
    };
    for entry in entries {
        let Some(path) = entry.get("path").and_then(|p| p.as_str()) else {
            continue;
        };
        if !path.to_lowercase().ends_with(".gguf") || path.contains('/') {
            continue;
        }
        // LFS/xet-backed files report the real content size under `lfs.size`; the
        // top-level `size` for those is just the pointer's byte count.
        let size = entry
            .get("lfs")
            .and_then(|lfs| lfs.get("size"))
            .and_then(|s| s.as_u64())
            .or_else(|| entry.get("size").and_then(|s| s.as_u64()))
            .unwrap_or(0);

        out.push(HfGgufFile {
            repo_id: repo.id.clone(),
            filename: path.to_string(),
            size_bytes: size,
            downloads: repo.downloads,
            likes: repo.likes,
            architecture: guess_architecture(path, &repo.id),
            quant: guess_quant(path).unwrap_or_default(),
        });
    }
    Ok(out)
}

/// Run `curl -fsSL <url>` returning `(body, raw_response_headers)`. Headers are
/// dumped to a temp file (`-D`) so the body stays clean JSON regardless of
/// redirects. Offline / HTTP error → typed `Err`.
fn curl_get_with_headers(url: &str) -> anyhow::Result<(Vec<u8>, String)> {
    let header_path = unique_temp_path("camelid-hf-hdr");
    let output = std::process::Command::new("curl")
        .args(["-fsSL", "-D"])
        .arg(&header_path)
        .arg(url)
        .output()
        .map_err(|err| anyhow::anyhow!("could not run curl (is it installed?): {err}"))?;

    let headers = std::fs::read_to_string(&header_path).unwrap_or_default();
    let _ = std::fs::remove_file(&header_path);

    if !output.status.success() {
        anyhow::bail!(
            "hugging face request failed (curl exited {})",
            output.status
        );
    }
    Ok((output.stdout, headers))
}

/// A process-unique temp path. Avoids time/random (unavailable in some sandboxes)
/// by combining the pid with a monotonic counter.
fn unique_temp_path(prefix: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{prefix}-{}-{n}.tmp", std::process::id()))
}

/// Pull the `cursor` query param out of the `rel="next"` entry of an HTTP `Link`
/// header, if any. Returns the decoded cursor (callers re-encode before reuse).
fn parse_next_cursor(headers: &str) -> Option<String> {
    for line in headers.lines() {
        if !line.to_lowercase().starts_with("link:") {
            continue;
        }
        let value = &line[line.find(':')? + 1..];
        for part in value.split(',') {
            if !part.to_lowercase().contains("rel=\"next\"") {
                continue;
            }
            let start = part.find('<')?;
            let end = part.find('>')?;
            return extract_query_param(&part[start + 1..end], "cursor");
        }
    }
    None
}

/// Decoded value of query parameter `key` in `url`, if present.
fn extract_query_param(url: &str, key: &str) -> Option<String> {
    let query = url.split('?').nth(1)?;
    for pair in query.split('&') {
        let mut it = pair.splitn(2, '=');
        if it.next()? == key {
            return Some(urldecode(it.next().unwrap_or("")));
        }
    }
    None
}

/// Percent-encode a query component (RFC 3986 unreserved set kept verbatim).
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Reverse `urlencode` (also tolerates `+` left as-is; HF cursors are %-encoded).
fn urldecode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 3 <= bytes.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Best-effort quant guess from a filename (advisory only). Longest tokens first so
/// `Q4_K_M` isn't shadowed by `Q4_K`.
fn guess_quant(filename: &str) -> Option<String> {
    let upper = filename.to_uppercase();
    const PATTERNS: &[&str] = &[
        "IQ2_XXS", "IQ3_XXS", "IQ2_XS", "IQ3_XS", "IQ4_XS", "IQ4_NL", "IQ1_S", "IQ1_M", "IQ2_S",
        "IQ2_M", "IQ3_S", "IQ3_M", "Q2_K_S", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_K_S", "Q4_K_M",
        "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_K", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q4_0", "Q4_1", "Q5_0",
        "Q5_1", "Q8_0", "BF16", "F16", "F32",
    ];
    PATTERNS
        .iter()
        .find(|p| upper.contains(**p))
        .map(|p| (*p).to_string())
}

/// Best-effort architecture guess from repo/filename tokens (advisory only). The
/// real architecture comes from GGUF metadata at load time; this never gates a
/// lane. More specific tokens are matched first.
fn guess_architecture(filename: &str, repo_id: &str) -> String {
    let hay = format!("{repo_id} {filename}").to_lowercase();
    const TABLE: &[(&str, &str)] = &[
        ("qwen3", "qwen3"),
        ("qwen2", "qwen2"),
        ("smollm3", "smollm3"),
        ("smollm", "smollm3"),
        ("gemma-4", "gemma4"),
        ("gemma4", "gemma4"),
        ("gemma-3", "gemma3"),
        ("gemma3", "gemma3"),
        ("phi-3", "phi3"),
        ("phi3", "phi3"),
        ("lfm2", "lfm2"),
        ("mixtral", "mixtral"),
        ("mistral", "mistral"),
        ("tinyllama", "llama"),
        ("llama", "llama"),
    ];
    for (needle, arch) in TABLE {
        if hay.contains(needle) {
            return (*arch).to_string();
        }
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guesses_common_quants_longest_first() {
        assert_eq!(guess_quant("model-Q4_K_M.gguf").as_deref(), Some("Q4_K_M"));
        assert_eq!(guess_quant("model-Q8_0.gguf").as_deref(), Some("Q8_0"));
        assert_eq!(
            guess_quant("model-IQ3_XXS.gguf").as_deref(),
            Some("IQ3_XXS")
        );
        assert_eq!(guess_quant("model-f16.gguf").as_deref(), Some("F16"));
        assert_eq!(guess_quant("model.gguf"), None);
    }

    #[test]
    fn guesses_architecture_advisory() {
        assert_eq!(
            guess_architecture("Qwen3-4B-Q8_0.gguf", "Qwen/Qwen3-4B-GGUF"),
            "qwen3"
        );
        assert_eq!(
            guess_architecture("model.gguf", "TheBloke/TinyLlama-1.1B-GGUF"),
            "llama"
        );
        assert_eq!(
            guess_architecture("gemma-3-1b-it-Q8_0.gguf", "ggml-org/x"),
            "gemma3"
        );
        assert_eq!(guess_architecture("mystery.gguf", "someone/mystery"), "");
    }

    #[test]
    fn urlencode_urldecode_roundtrip() {
        let raw = "abc/123+def=ghi?x&y";
        assert_eq!(urldecode(&urlencode(raw)), raw);
    }

    #[test]
    fn extracts_cursor_from_query() {
        assert_eq!(
            extract_query_param(
                "https://huggingface.co/api/models?search=q&cursor=AbC%3D%3D",
                "cursor"
            )
            .as_deref(),
            Some("AbC==")
        );
        assert_eq!(extract_query_param("https://x/api?foo=1", "cursor"), None);
    }

    #[test]
    fn parses_next_cursor_from_link_header() {
        let headers = "HTTP/2 200\r\nlink: <https://huggingface.co/api/models?search=q&cursor=NEXT%2B1>; rel=\"next\"\r\ncontent-type: application/json\r\n";
        assert_eq!(parse_next_cursor(headers).as_deref(), Some("NEXT+1"));
        assert_eq!(
            parse_next_cursor("HTTP/2 200\r\ncontent-type: application/json\r\n"),
            None
        );
    }
}
