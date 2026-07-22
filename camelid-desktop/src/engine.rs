// Sidecar lifecycle for the camelid engine.
//
// The desktop app does NOT reimplement any inference, tokenization, or HTTP surface. It
// spawns the shipped `camelid` server binary as a loopback-only sidecar, health-gates it,
// then points the WebView at the sidecar's already-embedded UI. Behavior is therefore
// byte-identical to running `camelid serve` + the web UI manually. See DECISIONS.md D11.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// Resolved engine binary stem. The crate/binary is `camelid`; the legacy
/// `backendinference` name must never be reintroduced (DECISIONS.md D2). Everything that
/// names the engine references THIS constant rather than scattering string literals.
pub const ENGINE_BINARY_STEM: &str = "camelid";

/// Platform file name for the engine binary, derived from [`ENGINE_BINARY_STEM`] so the
/// resolved name has exactly one source of truth (never a scattered literal).
pub fn engine_binary_file() -> String {
    if cfg!(windows) {
        format!("{ENGINE_BINARY_STEM}.exe")
    } else {
        ENGINE_BINARY_STEM.to_string()
    }
}

/// Health-gate budget: poll `/v1/health` for up to this long before declaring failure.
const HEALTH_TIMEOUT: Duration = Duration::from_secs(40);
/// Backoff between health polls (model load can dominate; keep polls cheap and patient).
const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(350);

/// A fatal error during sidecar startup, carrying any captured engine stderr so the splash
/// can surface the *real* failure rather than a fake "ready" state.
#[derive(Debug)]
pub struct EngineError {
    pub message: String,
    pub stderr: Option<String>,
}

impl EngineError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            stderr: None,
        }
    }
    fn with_stderr(message: impl Into<String>, stderr: Option<String>) -> Self {
        Self {
            message: message.into(),
            stderr,
        }
    }
    /// Human-readable detail block for the splash error pane.
    pub fn detail(&self) -> String {
        match &self.stderr {
            Some(s) if !s.trim().is_empty() => {
                format!("{}\n\n--- engine stderr ---\n{}", self.message, s.trim())
            }
            _ => self.message.clone(),
        }
    }
}

/// A running sidecar plus the loopback port it bound. Dropping/`shutdown` kills the child.
pub struct Engine {
    child: Child,
    port: u16,
    #[cfg(windows)]
    _job: Option<JobObject>,
}

impl Engine {
    /// The loopback base URL the WebView should navigate to. UI and API are same-origin.
    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}/", self.port)
    }

    /// Graceful-ish shutdown. On Windows there is no SIGTERM; the child is loopback-only and
    /// holds no external state, so `TerminateProcess` (via `Child::kill`) is the clean stop.
    /// The kill-on-close job object (set in `spawn`) is the backstop if the parent crashes.
    pub fn shutdown(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Locate the `camelid` engine binary. Resolution order:
/// 1. Beside the desktop executable (the bundled/portable case — and the dev case, since both
///    workspace binaries land in `target/<profile>/`).
/// 2. An explicit `resource_dir/sidecar/camelid.exe` (Tauri-bundled resource layout).
/// 3. Bare name on `PATH` (developer convenience).
pub fn resolve_engine_path(resource_dir: Option<PathBuf>) -> Result<PathBuf, EngineError> {
    let file = engine_binary_file();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let beside = dir.join(&file);
            if beside.is_file() {
                return Ok(beside);
            }
        }
    }
    if let Some(res) = resource_dir {
        let bundled = res.join("sidecar").join(&file);
        if bundled.is_file() {
            return Ok(bundled);
        }
    }
    // Fall back to PATH resolution by bare name; spawn will surface a clear error if absent.
    Ok(PathBuf::from(file))
}

/// Reserve an OS-assigned ephemeral port on loopback. We bind, read the assigned port, then
/// release it and hand the number to the sidecar. There is a small TOCTOU window between
/// release and the sidecar re-binding; this is the standard trade-off and is handled by the
/// health gate failing loudly (never silently) if the bind races.
pub fn pick_ephemeral_port() -> Result<u16, EngineError> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .map_err(|e| EngineError::new(format!("could not reserve a loopback port: {e}")))?;
    let port = listener
        .local_addr()
        .map_err(|e| EngineError::new(format!("could not read reserved port: {e}")))?
        .port();
    drop(listener);
    Ok(port)
}

/// Spawn `camelid serve --addr 127.0.0.1:<port> --no-open --models-dir <abs>`, bound to
/// loopback only, and health-gate it. Returns a running [`Engine`] or an [`EngineError`]
/// carrying engine stderr.
pub fn spawn(engine_path: &PathBuf) -> Result<Engine, EngineError> {
    let port = pick_ephemeral_port()?;
    let addr = format!("127.0.0.1:{port}");

    let mut command = Command::new(engine_path);
    command
        .arg("serve")
        .arg("--addr")
        .arg(&addr)
        .arg("--no-open");
    // The desktop's model store is the `models/` folder beside the engine binary (the
    // installed layout: camelid.exe and models/ side by side; NSIS upgrades preserve that
    // folder). The sidecar inherits an arbitrary launch-context CWD (Explorer, a shortcut,
    // the shell), so pin the directory explicitly as an ABSOLUTE path — otherwise the
    // engine's local-models scan, catalog downloads, and relative /api/models/load paths
    // resolve against whatever directory Windows happened to launch the app from.
    if let Some(models_dir) = sidecar_models_dir(engine_path) {
        command.arg("--models-dir").arg(models_dir);
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    no_console_window(&mut command);

    let mut child = command.spawn().map_err(|e| {
        EngineError::new(format!(
            "failed to launch the camelid engine at {}: {e}",
            engine_path.display()
        ))
    })?;

    // Tie the child's lifetime to ours: if the desktop process dies (even crashes), the OS
    // terminates the sidecar too, so no orphaned `camelid` process can survive.
    #[cfg(windows)]
    let job = JobObject::assign(&child).ok();

    match wait_for_health(port, &mut child) {
        Ok(()) => Ok(Engine {
            child,
            port,
            #[cfg(windows)]
            _job: job,
        }),
        Err(err) => {
            // Capture whatever the engine printed before failing, then ensure it's dead.
            let stderr = drain_stderr(&mut child);
            let _ = child.kill();
            let _ = child.wait();
            Err(EngineError::with_stderr(err.message, stderr.or(err.stderr)))
        }
    }
}

/// The desktop's models directory: the `models/` folder beside the engine binary, as an
/// absolute path. `None` when the engine was resolved as a bare `PATH` name (the developer
/// fallback in `resolve_engine_path`), where the engine's own exe-relative default is the
/// right authority. The directory deliberately does NOT need to exist yet — on a fresh
/// install it is created by the engine's first catalog download.
fn sidecar_models_dir(engine_path: &Path) -> Option<PathBuf> {
    let parent = engine_path
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())?;
    let models_dir = parent.join("models");
    if models_dir.is_absolute() {
        Some(models_dir)
    } else {
        std::path::absolute(&models_dir).ok()
    }
}

/// Poll `/v1/health` until it returns 200, the engine exits, or the budget elapses.
fn wait_for_health(port: u16, child: &mut Child) -> Result<(), EngineError> {
    let deadline = Instant::now() + HEALTH_TIMEOUT;
    loop {
        // If the engine already exited, fail immediately with its status (stderr added later).
        if let Ok(Some(status)) = child.try_wait() {
            return Err(EngineError::new(format!(
                "the camelid engine exited before becoming healthy (status: {status})"
            )));
        }
        if http_health_ok(port) {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(EngineError::new(format!(
                "the camelid engine did not report healthy on 127.0.0.1:{port} within {}s",
                HEALTH_TIMEOUT.as_secs()
            )));
        }
        std::thread::sleep(HEALTH_POLL_INTERVAL);
    }
}

/// Dependency-free loopback HTTP/1.1 GET of `/v1/health`; true iff the status line is 200.
fn http_health_ok(port: u16) -> bool {
    let addr: SocketAddr = match format!("127.0.0.1:{port}").parse() {
        Ok(a) => a,
        Err(_) => return false,
    };
    let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_millis(750)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(2000)));
    let _ = stream.set_write_timeout(Some(Duration::from_millis(2000)));
    let req =
        format!("GET /v1/health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    if stream.write_all(req.as_bytes()).is_err() {
        return false;
    }
    let mut buf = [0u8; 256];
    match stream.read(&mut buf) {
        Ok(n) if n > 0 => {
            let head = String::from_utf8_lossy(&buf[..n]);
            head.starts_with("HTTP/1.1 200") || head.starts_with("HTTP/1.0 200")
        }
        _ => false,
    }
}

/// Best-effort read of any buffered engine stderr (used to enrich a startup failure).
fn drain_stderr(child: &mut Child) -> Option<String> {
    let mut stderr = child.stderr.take()?;
    let mut buf = String::new();
    // The child has been (or is about to be) killed; this read returns what was buffered.
    let _ = stderr.read_to_string(&mut buf);
    if buf.trim().is_empty() {
        None
    } else {
        Some(buf)
    }
}

/// Suppress the spawned engine's console window on Windows (CREATE_NO_WINDOW).
#[cfg(windows)]
fn no_console_window(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn no_console_window(_command: &mut Command) {}

// ---------------------------------------------------------------------------------------
// Windows Job Object: kill-on-close backstop so a desktop crash can't orphan the sidecar.
// ---------------------------------------------------------------------------------------
#[cfg(windows)]
struct JobObject {
    handle: windows_sys::Win32::Foundation::HANDLE,
}

// The job handle is just a kernel object handle; closing it (the only thing we do, on Drop)
// is valid from any thread. Sending it across threads is sound, so the engine can live in
// Tauri's `Send + Sync` managed state.
#[cfg(windows)]
unsafe impl Send for JobObject {}

#[cfg(windows)]
impl JobObject {
    fn assign(child: &Child) -> Result<Self, ()> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::HANDLE;
        use windows_sys::Win32::System::JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
            SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        };

        unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job.is_null() {
                return Err(());
            }
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let ok = SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const core::ffi::c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );
            if ok == 0 {
                windows_sys::Win32::Foundation::CloseHandle(job);
                return Err(());
            }
            let proc_handle = child.as_raw_handle() as HANDLE;
            if AssignProcessToJobObject(job, proc_handle) == 0 {
                windows_sys::Win32::Foundation::CloseHandle(job);
                return Err(());
            }
            Ok(JobObject { handle: job })
        }
    }
}

#[cfg(windows)]
impl Drop for JobObject {
    fn drop(&mut self) {
        // Closing the last handle to the job triggers KILL_ON_JOB_CLOSE for the sidecar.
        unsafe {
            windows_sys::Win32::Foundation::CloseHandle(self.handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::sidecar_models_dir;
    use std::path::{Path, PathBuf};

    #[test]
    fn models_dir_sits_beside_an_absolute_engine_path() {
        let engine = if cfg!(windows) {
            PathBuf::from(r"C:\Apps\Camelid\camelid.exe")
        } else {
            PathBuf::from("/opt/camelid/camelid")
        };
        let dir = sidecar_models_dir(&engine).expect("absolute engine path yields a models dir");
        assert!(dir.is_absolute());
        assert_eq!(dir, engine.parent().unwrap().join("models"));
    }

    #[test]
    fn bare_path_engine_name_yields_no_models_dir() {
        // The PATH-resolution fallback: no directory to anchor to, so the engine's own
        // exe-relative default must stay in charge.
        assert_eq!(sidecar_models_dir(Path::new("camelid.exe")), None);
    }
}
