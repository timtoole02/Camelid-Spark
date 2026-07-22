//! Minimal blocking HTTP/1.1 client for the local Camelid server.
//!
//! `camelid chat` is a thin client over the already-audited local API (see
//! `DECISIONS.md` D6 / `RECON_CHAT.md` §6). Rather than pull in an HTTP-client
//! crate, it reuses the same `TcpStream` + read-to-EOF + de-chunk shape the
//! receipt verifier uses (`src/receipt/verify.rs`), and adds an incremental
//! `data:` reader for the SSE chat lane so streamed terminal output rides the
//! exact wire path `/v1/chat/completions` already serves.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde::Deserialize;
use serde_json::{json, Value};

/// The subset of `/v1/health` the REPL consults (extra fields are ignored).
#[derive(Debug, Deserialize)]
pub struct Health {
    pub ok: bool,
    #[serde(default)]
    pub generation_ready: bool,
}

/// One supported/planned row from `/api/capabilities` → `model_compatibility`
/// (extra fields are ignored).
#[derive(Debug, Clone, Deserialize)]
pub struct CompatRow {
    pub id: String,
    #[serde(default)]
    pub family: String,
    #[serde(default)]
    pub quantization: String,
    #[serde(default)]
    pub status: String,
    /// Whether this exact row is verified to drive tool-calling (agent mode).
    /// Default false; promoted only with a real tool-call round-trip as evidence.
    #[serde(default)]
    pub tool_capable: bool,
}

impl CompatRow {
    /// The picker's supported predicate, read from the ledger at runtime: a row
    /// is offered only when the engine marks it `supported…`. `planned`,
    /// `active_validation_partial`, and `unsupported` rows are excluded.
    pub fn is_supported(&self) -> bool {
        self.status.starts_with("supported")
    }
}

#[derive(Debug, Deserialize)]
struct Capabilities {
    #[serde(default)]
    model_compatibility: Vec<CompatRow>,
}

#[derive(Debug, Deserialize)]
struct ModelList {
    #[serde(default)]
    data: Vec<LoadedInfo>,
}

/// One currently-loaded model from `GET /v1/models`.
#[derive(Debug, Clone, Deserialize)]
pub struct LoadedInfo {
    pub id: String,
    #[serde(default)]
    pub meta: Option<LoadedMeta>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LoadedMeta {
    #[serde(default)]
    pub n_ctx_train: Option<u32>,
    #[serde(default)]
    pub n_params: Option<u64>,
    #[serde(default)]
    pub size: Option<u64>,
}

impl LoadedInfo {
    pub fn context_length(&self) -> Option<u32> {
        self.meta.as_ref().and_then(|m| m.n_ctx_train)
    }
    /// A short human descriptor, e.g. "8.0B · ctx 8192 · 8.5 GB".
    pub fn descriptor(&self) -> String {
        let Some(meta) = &self.meta else {
            return String::new();
        };
        let mut parts = Vec::new();
        if let Some(p) = meta.n_params {
            parts.push(format!("{:.1}B", p as f64 / 1e9));
        }
        if let Some(c) = meta.n_ctx_train {
            parts.push(format!("ctx {c}"));
        }
        if let Some(s) = meta.size {
            parts.push(format!("{:.1} GB", s as f64 / 1e9));
        }
        parts.join(" · ")
    }
}

/// Outcome of a `/api/models/load` call.
pub enum LoadOutcome {
    /// The model loaded and exposes a Camelid-supported runtime config.
    Loaded { id: String },
    /// The model loaded as metadata but its architecture is not supported. The
    /// message is the engine's typed unsupported-state text, surfaced verbatim.
    Unsupported { message: String },
}

/// How a finished/halted chat stream ended.
#[derive(Debug, PartialEq, Eq)]
pub enum StreamEnd {
    /// `[DONE]` (or a clean close) was reached.
    Done,
    /// Aborted because the cancel flag was set (Ctrl-C mid-stream).
    Cancelled,
}

#[derive(Clone)]
pub struct Client {
    addr: SocketAddr,
}

impl Client {
    pub fn new(addr: SocketAddr) -> Self {
        Self { addr }
    }

    fn connect(&self, read_timeout: Duration) -> std::io::Result<TcpStream> {
        let stream = TcpStream::connect_timeout(&self.addr, Duration::from_secs(2))?;
        stream.set_read_timeout(Some(read_timeout))?;
        stream.set_write_timeout(Some(Duration::from_secs(30)))?;
        Ok(stream)
    }

    /// Send a request and read the whole response (status, JSON body). Used for
    /// the small control-plane calls (health, capabilities, load).
    fn request(
        &self,
        method: &str,
        path: &str,
        body: Option<&Value>,
        timeout: Duration,
    ) -> anyhow::Result<(u16, Value)> {
        let mut stream = self.connect(timeout)?;
        let raw_request = encode_request(
            method,
            path,
            &self.addr.to_string(),
            body,
            "application/json",
        )?;
        stream.write_all(&raw_request.0)?;
        stream.write_all(&raw_request.1)?;
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw)?;
        parse_http_response(&raw).map_err(|err| anyhow::anyhow!(err))
    }

    /// `GET /v1/health`. Returns `None` when the server is unreachable or
    /// reports not-ok, so callers can probe attach-vs-spawn cheaply.
    pub fn health(&self) -> Option<Health> {
        let (status, body) = self
            .request("GET", "/v1/health", None, Duration::from_secs(5))
            .ok()?;
        if status != 200 {
            return None;
        }
        let health: Health = serde_json::from_value(body).ok()?;
        health.ok.then_some(health)
    }

    /// `GET /api/capabilities` → the supported/planned ledger rows.
    pub fn capabilities(&self) -> anyhow::Result<Vec<CompatRow>> {
        let (status, body) =
            self.request("GET", "/api/capabilities", None, Duration::from_secs(10))?;
        anyhow::ensure!(status == 200, "/api/capabilities returned HTTP {status}");
        let caps: Capabilities = serde_json::from_value(body)?;
        Ok(caps.model_compatibility)
    }

    /// `GET /v1/models` → every model currently loaded in the server (for the
    /// instant loaded-model switcher). Empty on error.
    pub fn list_loaded(&self) -> Vec<LoadedInfo> {
        let Ok((status, body)) = self.request("GET", "/v1/models", None, Duration::from_secs(5))
        else {
            return Vec::new();
        };
        if status != 200 {
            return Vec::new();
        }
        serde_json::from_value::<ModelList>(body)
            .map(|list| list.data)
            .unwrap_or_default()
    }

    /// `GET /api/models/current` → the active `LoadedModel` JSON (for `/info`),
    /// or `None` when nothing is loaded.
    pub fn current_model(&self) -> Option<Value> {
        let (status, body) = self
            .request("GET", "/api/models/current", None, Duration::from_secs(5))
            .ok()?;
        (status == 200).then_some(body)
    }

    /// `POST /api/models/load`. On a recognized-but-unsupported architecture the
    /// server returns 200 with `unsupported_runtime` populated — surfaced here as
    /// `LoadOutcome::Unsupported` carrying the engine's typed message. A hard
    /// failure (bad path, unreadable GGUF) returns the server's error verbatim.
    pub fn load_model(&self, path: &str, id: Option<&str>) -> anyhow::Result<LoadOutcome> {
        let mut req = json!({ "path": path });
        if let Some(id) = id {
            req["id"] = json!(id);
        }
        let (status, body) = self.request(
            "POST",
            "/api/models/load",
            Some(&req),
            Duration::from_secs(600),
        )?;
        if status != 200 {
            let message =
                envelope_message(&body).unwrap_or_else(|| format!("load failed (HTTP {status})"));
            anyhow::bail!(message);
        }
        if let Some(unsupported) = body.get("unsupported_runtime").filter(|v| !v.is_null()) {
            let message = unsupported
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("loaded model architecture is not supported by this Camelid build")
                .to_string();
            return Ok(LoadOutcome::Unsupported { message });
        }
        let id = body
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("model")
            .to_string();
        Ok(LoadOutcome::Loaded { id })
    }

    /// Stream `POST /v1/chat/completions` with `stream=true`. `on_delta` is
    /// called with each assistant content delta as it arrives. `cancel` is polled
    /// between socket reads so Ctrl-C aborts cleanly. Returns how the stream
    /// ended plus the number of content deltas seen (one per generated token on
    /// Camelid's per-token SSE lane → the completion-token count).
    pub fn chat_stream(
        &self,
        request: &Value,
        cancel: &AtomicBool,
        mut on_delta: impl FnMut(&str),
    ) -> anyhow::Result<(StreamEnd, u32)> {
        // A short read timeout lets the loop wake to check `cancel` even while
        // the server is mid-generation and no bytes are arriving.
        let mut stream = self.connect(Duration::from_millis(250))?;
        let (head, body) = encode_request(
            "POST",
            "/v1/chat/completions",
            &self.addr.to_string(),
            Some(request),
            "text/event-stream",
        )?;
        stream.write_all(&head)?;
        stream.write_all(&body)?;

        let mut reader = SseReader::new(stream);
        if reader.read_headers(cancel)? {
            return Ok((StreamEnd::Cancelled, 0));
        }
        if reader.status != 200 {
            let message = reader.drain_error_body();
            anyhow::bail!(message);
        }

        let mut deltas: u32 = 0;
        let end = reader.stream(cancel, |line| {
            if let Some(payload) = line.strip_prefix("data:") {
                let payload = payload.trim();
                if payload == "[DONE]" {
                    return SseControl::Done;
                }
                if let Ok(chunk) = serde_json::from_str::<Value>(payload) {
                    if let Some(content) = chunk
                        .pointer("/choices/0/delta/content")
                        .and_then(Value::as_str)
                    {
                        if !content.is_empty() {
                            deltas += 1;
                            on_delta(content);
                        }
                    }
                }
            }
            SseControl::Continue
        })?;
        Ok((end, deltas))
    }

    /// POST a chat request and return the full assistant turn: text content PLUS
    /// any structured `tool_calls` (OpenAI shape). The server emits structured
    /// tool calls and EMPTIES `content` on a tool call, so an agent loop MUST read
    /// `tool_calls` here — reading only the text loses every tool call.
    pub fn chat_turn(&self, request: &Value) -> anyhow::Result<ChatTurn> {
        // Read deadline: a fast model answers in seconds; the 30-min ceiling is only
        // hit by the slow pure-f32 runnable oracle lane (e.g. a 9B qwen35 prefilling a
        // long tool prompt at ~1s/token), without masking a genuinely hung server.
        let (status, body) = self.request(
            "POST",
            "/v1/chat/completions",
            Some(request),
            Duration::from_secs(1800),
        )?;
        if status != 200 {
            anyhow::bail!(envelope_message(&body).unwrap_or_else(|| format!("HTTP {status}")));
        }
        Ok(parse_chat_turn(&body))
    }

    /// Text-only convenience over [`chat_turn`] for the plain chat UI (which never
    /// supplies tools, so `content` always carries the answer).
    pub fn chat_blocking(
        &self,
        request: &Value,
    ) -> anyhow::Result<(String, Option<u32>, Option<u32>)> {
        let turn = self.chat_turn(request)?;
        Ok((turn.content, turn.prompt_tokens, turn.completion_tokens))
    }
}

/// One assistant turn from `/v1/chat/completions`.
pub struct ChatTurn {
    pub content: String,
    /// Structured tool calls (OpenAI shape); `content` is empty when this is set.
    pub tool_calls: Vec<ToolCallOut>,
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
}

/// One structured tool call (OpenAI shape): a name + a JSON-encoded args string.
pub struct ToolCallOut {
    pub name: String,
    pub arguments: String,
}

/// Extract the assistant turn (content + structured tool calls + token counts)
/// from a `/v1/chat/completions` response body.
fn parse_chat_turn(body: &Value) -> ChatTurn {
    let content = body
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let tool_calls = body
        .pointer("/choices/0/message/tool_calls")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|tc| {
                    let name = tc
                        .pointer("/function/name")
                        .and_then(Value::as_str)?
                        .to_string();
                    let arguments = tc
                        .pointer("/function/arguments")
                        .and_then(Value::as_str)
                        .unwrap_or("{}")
                        .to_string();
                    Some(ToolCallOut { name, arguments })
                })
                .collect()
        })
        .unwrap_or_default();
    let prompt_tokens = body
        .pointer("/usage/prompt_tokens")
        .and_then(Value::as_u64)
        .map(|n| n as u32);
    let completion_tokens = body
        .pointer("/usage/completion_tokens")
        .and_then(Value::as_u64)
        .map(|n| n as u32);
    ChatTurn {
        content,
        tool_calls,
        prompt_tokens,
        completion_tokens,
    }
}

/// Pull `error.message` out of a Camelid `ErrorEnvelope` body, if present.
fn envelope_message(body: &Value) -> Option<String> {
    body.pointer("/error/message")
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// Build the request head (bytes up to and including the blank line) and the
/// separate body bytes.
fn encode_request(
    method: &str,
    path: &str,
    host: &str,
    body: Option<&Value>,
    accept: &str,
) -> anyhow::Result<(Vec<u8>, Vec<u8>)> {
    let body_bytes = match body {
        Some(value) => serde_json::to_vec(value)?,
        None => Vec::new(),
    };
    let mut head = format!(
        "{method} {path} HTTP/1.1\r\nHost: {host}\r\nAccept: {accept}\r\nConnection: close\r\n"
    );
    if !body_bytes.is_empty() {
        head.push_str("Content-Type: application/json\r\n");
        head.push_str(&format!("Content-Length: {}\r\n", body_bytes.len()));
    }
    head.push_str("\r\n");
    Ok((head.into_bytes(), body_bytes))
}

/// Incremental reader for a chunked `text/event-stream` body.
struct SseReader {
    stream: TcpStream,
    /// Raw bytes read from the socket but not yet consumed by the decoder.
    pending: Vec<u8>,
    status: u16,
    chunked: bool,
}

enum SseControl {
    Continue,
    Done,
}

impl SseReader {
    fn new(stream: TcpStream) -> Self {
        Self {
            stream,
            pending: Vec::new(),
            status: 0,
            chunked: false,
        }
    }

    /// Read until the header terminator, recording status + transfer-encoding and
    /// leaving any post-header body bytes in `pending`. Returns `true` if Ctrl-C
    /// cancelled while waiting (the server may prefill for many seconds before the
    /// first byte, so a read timeout here is normal and is retried, not an error).
    fn read_headers(&mut self, cancel: &AtomicBool) -> anyhow::Result<bool> {
        let mut scratch = [0u8; 4096];
        loop {
            if let Some(end) = find(&self.pending, b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&self.pending[..end]);
                let mut lines = head.split("\r\n");
                let status_line = lines.next().unwrap_or_default();
                self.status = status_line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|code| code.parse().ok())
                    .ok_or_else(|| {
                        anyhow::anyhow!("malformed HTTP status line: {status_line:?}")
                    })?;
                for line in lines {
                    if let Some((name, value)) = line.split_once(':') {
                        if name.trim().eq_ignore_ascii_case("transfer-encoding")
                            && value.to_ascii_lowercase().contains("chunked")
                        {
                            self.chunked = true;
                        }
                    }
                }
                self.pending.drain(..end + 4);
                return Ok(false);
            }
            if cancel.load(Ordering::Relaxed) {
                return Ok(true);
            }
            match self.stream.read(&mut scratch) {
                Ok(0) => anyhow::bail!("connection closed before HTTP headers completed"),
                Ok(n) => self.pending.extend_from_slice(&scratch[..n]),
                Err(err) if is_timeout(&err) => continue,
                Err(err) => return Err(err.into()),
            }
        }
    }

    /// On a non-200 status, read the rest of the (small) error body and extract a
    /// human-readable message.
    fn drain_error_body(&mut self) -> String {
        let mut scratch = [0u8; 4096];
        // Bounded read-to-close, tolerating the short read timeout (cap ~10s).
        let mut idle = 0;
        loop {
            match self.stream.read(&mut scratch) {
                Ok(0) => break,
                Ok(n) => {
                    idle = 0;
                    self.pending.extend_from_slice(&scratch[..n]);
                }
                Err(ref err) if is_timeout(err) => {
                    idle += 1;
                    if idle >= 40 {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        let body = if self.chunked {
            decode_chunked(&self.pending).unwrap_or_else(|_| self.pending.clone())
        } else {
            self.pending.clone()
        };
        serde_json::from_slice::<Value>(&body)
            .ok()
            .and_then(|v| envelope_message(&v))
            .unwrap_or_else(|| {
                format!(
                    "chat request failed (HTTP {}): {}",
                    self.status,
                    String::from_utf8_lossy(&body).trim()
                )
            })
    }

    /// Drive the body: de-chunk on the fly, split into lines, and hand each line
    /// to `on_line`. Stops on `[DONE]`, EOF, or a set cancel flag.
    fn stream(
        &mut self,
        cancel: &AtomicBool,
        mut on_line: impl FnMut(&str) -> SseControl,
    ) -> anyhow::Result<StreamEnd> {
        let mut decoder = ChunkDecoder::new(self.chunked);
        let mut line_acc: Vec<u8> = Vec::new();
        let mut scratch = [0u8; 4096];
        // Process anything that arrived alongside the headers first.
        let seeded = std::mem::take(&mut self.pending);
        if Self::feed(&seeded, &mut decoder, &mut line_acc, &mut on_line) {
            return Ok(StreamEnd::Done);
        }
        loop {
            if cancel.load(Ordering::Relaxed) {
                return Ok(StreamEnd::Cancelled);
            }
            match self.stream.read(&mut scratch) {
                Ok(0) => return Ok(StreamEnd::Done),
                Ok(n) => {
                    if Self::feed(&scratch[..n], &mut decoder, &mut line_acc, &mut on_line) {
                        return Ok(StreamEnd::Done);
                    }
                }
                Err(err) if is_timeout(&err) => continue,
                Err(err) => return Err(err.into()),
            }
        }
    }

    /// Feed raw bytes through the de-chunker into `line_acc`, emitting complete
    /// lines. Returns true once `on_line` signalled `Done`.
    fn feed(
        bytes: &[u8],
        decoder: &mut ChunkDecoder,
        line_acc: &mut Vec<u8>,
        on_line: &mut impl FnMut(&str) -> SseControl,
    ) -> bool {
        decoder.push(bytes, line_acc);
        loop {
            let Some(nl) = line_acc.iter().position(|&b| b == b'\n') else {
                return false;
            };
            let mut line = line_acc[..nl].to_vec();
            line_acc.drain(..nl + 1);
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            let text = String::from_utf8_lossy(&line);
            if matches!(on_line(&text), SseControl::Done) {
                return true;
            }
        }
    }
}

/// Incremental HTTP chunked-transfer decoder. When `chunked` is false it is a
/// pass-through (some servers may close with identity encoding).
struct ChunkDecoder {
    chunked: bool,
    raw: Vec<u8>,
    /// Remaining bytes of the current chunk's data, or `None` while reading the
    /// next size line.
    remaining: Option<usize>,
    done: bool,
}

impl ChunkDecoder {
    fn new(chunked: bool) -> Self {
        Self {
            chunked,
            raw: Vec::new(),
            remaining: None,
            done: false,
        }
    }

    /// Append `bytes`; push any newly-decoded body bytes onto `out`.
    fn push(&mut self, bytes: &[u8], out: &mut Vec<u8>) {
        if !self.chunked {
            out.extend_from_slice(bytes);
            return;
        }
        self.raw.extend_from_slice(bytes);
        while !self.done {
            match self.remaining {
                None => {
                    // Need a full `<hex>\r\n` size line.
                    let Some(eol) = find(&self.raw, b"\r\n") else {
                        return;
                    };
                    let size_line = String::from_utf8_lossy(&self.raw[..eol]);
                    let hex = size_line.split(';').next().unwrap_or_default().trim();
                    let size = usize::from_str_radix(hex, 16).unwrap_or(0);
                    self.raw.drain(..eol + 2);
                    if size == 0 {
                        self.done = true;
                        return;
                    }
                    self.remaining = Some(size);
                }
                Some(0) => {
                    // Consume the chunk's trailing CRLF, then read the next size.
                    if self.raw.len() < 2 {
                        return;
                    }
                    self.raw.drain(..2);
                    self.remaining = None;
                }
                Some(n) => {
                    if self.raw.is_empty() {
                        return;
                    }
                    let take = n.min(self.raw.len());
                    out.extend_from_slice(&self.raw[..take]);
                    self.raw.drain(..take);
                    self.remaining = Some(n - take);
                }
            }
        }
    }
}

fn is_timeout(err: &std::io::Error) -> bool {
    matches!(
        err.kind(),
        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
    )
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Parse a full HTTP/1.1 response read to EOF (status, JSON body), de-chunking
/// when needed. Mirrors `src/receipt/verify.rs::parse_http_response`.
fn parse_http_response(raw: &[u8]) -> Result<(u16, Value), String> {
    let header_end = find(raw, b"\r\n\r\n")
        .ok_or_else(|| "malformed HTTP response: missing header terminator".to_string())?;
    let head = String::from_utf8_lossy(&raw[..header_end]);
    let mut lines = head.split("\r\n");
    let status_line = lines.next().unwrap_or_default();
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .ok_or_else(|| format!("malformed HTTP status line: {status_line:?}"))?;

    let mut chunked = false;
    let mut content_length: Option<usize> = None;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let name = name.trim().to_ascii_lowercase();
        let value = value.trim();
        if name == "transfer-encoding" && value.to_ascii_lowercase().contains("chunked") {
            chunked = true;
        } else if name == "content-length" {
            content_length = value.parse().ok();
        }
    }

    let body_raw = &raw[header_end + 4..];
    let body = if chunked {
        decode_chunked(body_raw)?
    } else if let Some(length) = content_length {
        body_raw.get(..length).unwrap_or(body_raw).to_vec()
    } else {
        body_raw.to_vec()
    };
    if body.is_empty() {
        return Ok((status, Value::Null));
    }
    let value = serde_json::from_slice(&body)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&body).trim().to_string()));
    Ok((status, value))
}

fn decode_chunked(mut raw: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    loop {
        let line_end = find(raw, b"\r\n")
            .ok_or_else(|| "malformed chunked body: missing size line".to_string())?;
        let size_line = String::from_utf8_lossy(&raw[..line_end]);
        let size_text = size_line.split(';').next().unwrap_or_default().trim();
        let size = usize::from_str_radix(size_text, 16)
            .map_err(|_| format!("malformed chunk size: {size_text:?}"))?;
        raw = &raw[line_end + 2..];
        if size == 0 {
            return Ok(out);
        }
        let chunk = raw
            .get(..size)
            .ok_or_else(|| "malformed chunked body: truncated chunk".to_string())?;
        out.extend_from_slice(chunk);
        raw = raw.get(size + 2..).unwrap_or(&[]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_chat_turn_reads_structured_tool_calls() {
        // The server empties `content` and emits a structured tool_calls array
        // when the model calls a tool. The agent loop must read it from here.
        let body = json!({
            "choices": [{ "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "1", "type": "function",
                    "function": { "name": "read_file", "arguments": "{\"path\":\"notes.txt\"}" }
                }]
            }}],
            "usage": { "prompt_tokens": 11, "completion_tokens": 7 }
        });
        let turn = parse_chat_turn(&body);
        assert!(turn.content.is_empty());
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].name, "read_file");
        assert!(turn.tool_calls[0].arguments.contains("notes.txt"));
        assert_eq!(turn.completion_tokens, Some(7));
    }

    #[test]
    fn parse_chat_turn_reads_plain_content() {
        let body = json!({"choices":[{"message":{"role":"assistant","content":"hello"}}]});
        let turn = parse_chat_turn(&body);
        assert_eq!(turn.content, "hello");
        assert!(turn.tool_calls.is_empty());
    }

    /// Decode a chunked SSE body delivered in several arbitrary socket reads and
    /// confirm the `data:` deltas come out intact and in order, ending at
    /// `[DONE]`. This is the Phase 8 SSE/delta parser unit test.
    #[test]
    fn sse_deltas_survive_arbitrary_chunk_splits() {
        // Two events, each its own HTTP chunk, plus the terminating 0-chunk.
        let event_a = "data: {\"choices\":[{\"delta\":{\"content\":\"Cert\"}}]}\n\n";
        let event_b = "data: {\"choices\":[{\"delta\":{\"content\":\"ainly\"}}]}\n\n";
        let done = "data: [DONE]\n\n";
        let wire = format!(
            "{:x}\r\n{event_a}\r\n{:x}\r\n{event_b}\r\n{:x}\r\n{done}\r\n0\r\n\r\n",
            event_a.len(),
            event_b.len(),
            done.len(),
        );
        let bytes = wire.into_bytes();

        // Feed the wire bytes in 7-byte slices to exercise partial size lines and
        // split chunk data.
        let mut decoder = ChunkDecoder::new(true);
        let mut line_acc = Vec::new();
        let mut collected = String::new();
        let mut done_seen = false;
        for slice in bytes.chunks(7) {
            let finished = SseReader::feed(slice, &mut decoder, &mut line_acc, &mut |line| {
                if let Some(payload) = line.strip_prefix("data:") {
                    let payload = payload.trim();
                    if payload == "[DONE]" {
                        return SseControl::Done;
                    }
                    let v: Value = serde_json::from_str(payload).unwrap();
                    collected.push_str(
                        v.pointer("/choices/0/delta/content")
                            .unwrap()
                            .as_str()
                            .unwrap(),
                    );
                }
                SseControl::Continue
            });
            if finished {
                done_seen = true;
                break;
            }
        }
        assert!(done_seen, "stream must terminate on [DONE]");
        assert_eq!(collected, "Certainly");
    }

    #[test]
    fn parses_content_length_and_chunked_bodies() {
        let cl = b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\n\r\n{\"ok\":true}";
        let (status, value) = parse_http_response(cl).unwrap();
        assert_eq!(status, 200);
        assert_eq!(value["ok"], true);

        let chunked = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\n{\"a\":\r\n5\r\ntrue}\r\n0\r\n\r\n";
        let (status, value) = parse_http_response(chunked).unwrap();
        assert_eq!(status, 200);
        assert_eq!(value["a"], true);
    }

    #[test]
    fn supported_predicate_reads_the_ledger_status() {
        let supported = CompatRow {
            id: "tinyllama_1_1b_chat_q8_0".into(),
            family: "llama_bpe_decoder".into(),
            quantization: "Q8_0".into(),
            status: "supported_exact_row_smoke".into(),
            tool_capable: false,
        };
        let planned = CompatRow {
            id: "qwen2_5_7b_instruct_q8_0".into(),
            family: "qwen2".into(),
            quantization: "Q8_0".into(),
            status: "planned".into(),
            tool_capable: false,
        };
        assert!(supported.is_supported());
        assert!(!planned.is_supported());
    }
}
