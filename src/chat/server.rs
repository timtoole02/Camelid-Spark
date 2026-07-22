//! Attach-or-spawn lifecycle for the local `camelid serve` the REPL drives.
//!
//! If a server is already answering `/v1/health` on the target address we attach
//! to it and leave it alone. Otherwise we spawn `camelid serve --addr <addr>
//! --no-open` as a child and tear it down on exit (Drop). This keeps terminal
//! generation on the already-audited HTTP lane (see `DECISIONS.md` D6).

use std::net::SocketAddr;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use super::client::Client;

pub struct ServerHandle {
    /// `Some` when we spawned the server (and must reap it); `None` when attached
    /// to a server someone else started.
    child: Option<Child>,
}

impl ServerHandle {
    /// Attach if `addr` is already healthy; otherwise spawn a child `serve` and
    /// wait (bounded) for it to come up.
    pub fn ensure(addr: SocketAddr, client: &Client) -> anyhow::Result<Self> {
        if client.health().is_some() {
            return Ok(Self { child: None });
        }

        let exe = std::env::current_exe().map_err(|err| {
            anyhow::anyhow!("cannot locate the camelid binary to spawn serve: {err}")
        })?;
        let mut command = Command::new(exe);
        command
            .arg("serve")
            .arg("--addr")
            .arg(addr.to_string())
            .arg("--no-open")
            // Supported gemma4 rows serve behind this flag (see /api/capabilities
            // notes); set it so picker-selected gemma4 rows actually generate.
            .env("CAMELID_GEMMA4_SERVE", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let child = command
            .spawn()
            .map_err(|err| anyhow::anyhow!("failed to spawn `camelid serve`: {err}"))?;

        // Poll health for up to ~30s.
        for _ in 0..150 {
            if client.health().is_some() {
                return Ok(Self { child: Some(child) });
            }
            std::thread::sleep(Duration::from_millis(200));
        }

        let mut child = child;
        let _ = child.kill();
        let _ = child.wait();
        anyhow::bail!("`camelid serve` did not become healthy at {addr} within 30s");
    }

    /// True when this handle owns a spawned server (vs an attached one).
    pub fn spawned(&self) -> bool {
        self.child.is_some()
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
