//! Embedded web UI.
//!
//! The React/Vite frontend in `frontend/` is built to `frontend/dist` and
//! embedded directly into the `camelid` binary at compile time, so a single
//! `camelid serve` command gives the user the full chat surface with no Node
//! toolchain and no second process. The embedded assets are mounted as the
//! router's fallback: every real API route is matched first, and anything else
//! (`/`, client-side routes, static assets) is served from here.
//!
//! In debug builds `rust-embed` reads `frontend/dist` from disk at request
//! time; in release builds the files are baked into the binary. When the
//! frontend has not been built (see `build.rs`, which writes a placeholder
//! `index.html` so the crate always compiles), the placeholder is served with
//! instructions instead of a blank page.

use axum::{
    http::{header, StatusCode, Uri},
    response::{IntoResponse, Response},
};
use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "frontend/dist"]
struct WebAssets;

/// Router fallback: serve an embedded asset for `uri`, falling back to
/// `index.html` for client-side (SPA) routes so a deep link still loads the app.
pub async fn handler(uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };

    if let Some(response) = serve_asset(path) {
        return response;
    }

    // A request that names a concrete file (has an extension) and was not found
    // is a genuine 404 — don't mask it by returning the SPA shell.
    if looks_like_file(path) {
        return (StatusCode::NOT_FOUND, "Not found").into_response();
    }

    // Otherwise treat it as a client-side route and serve the app shell.
    serve_asset("index.html").unwrap_or_else(missing_ui_response)
}

fn serve_asset(path: &str) -> Option<Response> {
    let asset = WebAssets::get(path)?;
    let mime = asset.metadata.mimetype();
    // Vite emits content-hashed filenames under assets/, so they can be cached
    // forever; the HTML shell must always be revalidated to pick up new builds.
    let cache_control = if path.starts_with("assets/") {
        "public, max-age=31536000, immutable"
    } else {
        "no-cache"
    };
    Some(
        (
            [
                (header::CONTENT_TYPE, mime.to_string()),
                (header::CACHE_CONTROL, cache_control.to_string()),
            ],
            asset.data,
        )
            .into_response(),
    )
}

/// Best-effort: open `url` in the user's default browser. Failures are silent
/// — the ready banner already printed the URL, so a headless or locked-down
/// environment just falls back to the user clicking it themselves.
pub fn open_in_browser(url: &str) {
    let mut command = std::process::Command::new(open_command());
    #[cfg(target_os = "windows")]
    command.args(["/C", "start", ""]);
    command.arg(url);
    let _ = command.spawn();
}

#[cfg(target_os = "macos")]
fn open_command() -> &'static str {
    "open"
}

#[cfg(target_os = "windows")]
fn open_command() -> &'static str {
    "cmd"
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn open_command() -> &'static str {
    "xdg-open"
}

fn looks_like_file(path: &str) -> bool {
    path.rsplit('/')
        .next()
        .map(|segment| segment.contains('.'))
        .unwrap_or(false)
}

/// Shown only when the binary was built without a frontend build present
/// (placeholder `index.html`). Points the user at the one command that fixes it.
fn missing_ui_response() -> Response {
    const BODY: &str = "<!doctype html><meta charset=utf-8><title>Camelid</title>\
        <body style=\"font-family:system-ui;max-width:42rem;margin:4rem auto;padding:0 1rem;line-height:1.6\">\
        <h1>🐪 Camelid</h1>\
        <p>The API is running, but this binary was built without the web UI.</p>\
        <p>Build the frontend, then rebuild the binary:</p>\
        <pre style=\"background:#f4f4f5;padding:1rem;border-radius:8px\">cd frontend &amp;&amp; npm ci &amp;&amp; npm run build\ncargo build --release</pre>\
        <p>The OpenAI-style API is available now at <code>/v1/chat/completions</code>.</p>";
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        BODY,
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_concrete_files_vs_client_routes() {
        assert!(looks_like_file("assets/index-abc123.js"));
        assert!(looks_like_file("favicon.ico"));
        assert!(!looks_like_file("settings"));
        assert!(!looks_like_file("chat/new"));
        assert!(!looks_like_file(""));
    }

    #[tokio::test]
    async fn root_serves_the_app_shell() {
        let res = handler(Uri::from_static("/")).await;
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn client_side_route_falls_back_to_the_shell() {
        // A deep link with no file extension is a client-side route: serve the
        // app shell (200) rather than a 404, so reloads on a sub-route work.
        let res = handler(Uri::from_static("/settings")).await;
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn missing_static_file_is_not_masked() {
        let res = handler(Uri::from_static("/assets/definitely-not-built.js")).await;
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
    }
}
