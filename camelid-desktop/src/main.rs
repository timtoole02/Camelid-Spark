// Camelid Desktop — additive native Windows shell around the camelid engine.
//
// Lifecycle: open the WebView2 window on a bundled splash, spawn `camelid serve` on a
// loopback ephemeral port as a sidecar, health-gate `/v1/health`, then navigate the window
// to the engine's already-embedded UI (UI + API same-origin). The sidecar is killed on exit;
// a kill-on-close job object backstops crashes. See DECISIONS.md D11 and engine.rs.
//
// `windows_subsystem = "windows"` suppresses the console window in release builds; debug
// builds keep the console so engine stderr is visible while developing.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod engine;

use std::sync::Mutex;
use tauri::{Emitter, Manager};

use engine::Engine;

/// Managed state holding the running sidecar so it can be torn down on exit.
#[derive(Default)]
struct EngineState(Mutex<Option<Engine>>);

/// Report real startup progress to the splash. Never emits a "ready" state that isn't backed
/// by a passing health check.
fn emit_status(app: &tauri::AppHandle, message: &str) {
    let _ = app.emit("engine-status", serde_json::json!({ "message": message }));
}

/// Surface a real startup failure (with engine stderr) on the splash error pane.
fn emit_error(app: &tauri::AppHandle, error: &str) {
    let _ = app.emit("engine-status", serde_json::json!({ "error": error }));
}

fn main() {
    tauri::Builder::default()
        .manage(EngineState::default())
        .setup(|app| {
            let handle = app.handle().clone();
            // Start the sidecar off the UI thread so the splash paints immediately.
            std::thread::spawn(move || start_engine(handle));
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building camelid-desktop")
        .run(|app_handle, event| {
            if matches!(
                event,
                tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit
            ) {
                shutdown_engine(app_handle);
            }
        });
}

/// Resolve, spawn, and health-gate the sidecar; on success navigate the window to its UI.
fn start_engine(app: tauri::AppHandle) {
    emit_status(&app, "Locating engine\u{2026}");
    let resource_dir = app.path().resource_dir().ok();
    let engine_path = match engine::resolve_engine_path(resource_dir) {
        Ok(p) => p,
        Err(e) => {
            emit_error(&app, &e.detail());
            return;
        }
    };

    emit_status(&app, "Starting engine\u{2026}");
    match engine::spawn(&engine_path) {
        Ok(eng) => {
            let url = eng.base_url();
            if let Some(state) = app.try_state::<EngineState>() {
                if let Ok(mut guard) = state.inner().0.lock() {
                    *guard = Some(eng);
                }
            }
            emit_status(&app, "Engine ready. Loading\u{2026}");
            if let Some(window) = app.get_webview_window("main") {
                match tauri::Url::parse(&url) {
                    Ok(parsed) => {
                        if let Err(e) = window.navigate(parsed) {
                            emit_error(&app, &format!("could not load the engine UI: {e}"));
                        }
                    }
                    Err(e) => emit_error(&app, &format!("invalid engine URL {url}: {e}")),
                }
            } else {
                emit_error(&app, "internal error: main window not found");
            }
        }
        Err(e) => emit_error(&app, &e.detail()),
    }
}

/// Kill the sidecar cleanly on shutdown. Idempotent: `take()` ensures one shutdown.
fn shutdown_engine(app_handle: &tauri::AppHandle) {
    if let Some(state) = app_handle.try_state::<EngineState>() {
        if let Ok(mut guard) = state.0.lock() {
            if let Some(mut eng) = guard.take() {
                eng.shutdown();
            }
        }
    }
}
