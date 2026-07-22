//! Windows UI Automation + screenshot for the computer-control agent (Phase 2).
//!
//! Gives a TEXT model "eyes and hands" for the GUI without pixels: it reads the
//! Windows UI Automation accessibility tree of a target window as TEXT (control
//! type, name, position) and invokes/clicks a control BY NAME. Plus a screenshot
//! tool that writes a PNG — for the operator or a future vision model, since the
//! text model can't read pixels.
//!
//! Implemented via trusted, embedded PowerShell using .NET `System.Windows.
//! Automation` + `System.Drawing` — no COM in Rust, no heavy `windows`-crate dep.
//! Model-supplied values (window title, control name, save path) are passed via
//! ENVIRONMENT VARIABLES and never interpolated into the script, so a crafted
//! name cannot inject PowerShell. The scripts below are fixed and trusted.

use std::io::{Read, Write};
use std::os::windows::io::AsRawHandle;
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use super::win_job::JobObject;

const CREATE_NO_WINDOW: u32 = 0x0800_0000;
const PS_TIMEOUT: Duration = Duration::from_secs(30);

fn system32(relative: &str) -> PathBuf {
    let root = std::env::var_os("SystemRoot").unwrap_or_else(|| "C:\\Windows".into());
    Path::new(&root).join("System32").join(relative)
}

/// Run a trusted embedded PowerShell `script` with `env` vars set, returning its
/// stdout on success or a typed error (stderr / non-zero exit / timeout). Mirrors
/// `run_windows_command`'s spawn: absolute interpreter path, stdin-fed script,
/// concurrent pipe drain, kill-on-close job object, hard timeout.
fn run_ps(script: &str, env: &[(&str, &str)]) -> Result<String, String> {
    let mut builder = Command::new(system32("WindowsPowerShell\\v1.0\\powershell.exe"));
    builder
        // -STA: UI Automation wants a single-threaded apartment. -Command - reads
        // the script from stdin (no command-line quoting).
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-STA",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "-",
        ])
        .creation_flags(CREATE_NO_WINDOW)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (k, v) in env {
        builder.env(k, v);
    }

    let mut child = builder
        .spawn()
        .map_err(|e| format!("spawn powershell failed: {e}"))?;

    let job = JobObject::new().ok();
    if let Some(ref j) = job {
        let _ = j.assign(child.as_raw_handle());
    }

    let out_reader = child.stdout.take().map(|mut p| {
        std::thread::spawn(move || {
            let mut b = Vec::new();
            let _ = p.read_to_end(&mut b);
            b
        })
    });
    let err_reader = child.stderr.take().map(|mut p| {
        std::thread::spawn(move || {
            let mut b = Vec::new();
            let _ = p.read_to_end(&mut b);
            b
        })
    });

    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(script.as_bytes());
        let _ = stdin.write_all(b"\r\n");
    }

    let deadline = Instant::now() + PS_TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break s,
            Ok(None) => {
                if Instant::now() >= deadline {
                    if let Some(ref j) = job {
                        j.terminate();
                    }
                    let _ = child.kill();
                    let _ = child.wait();
                    if let Some(h) = out_reader {
                        let _ = h.join();
                    }
                    if let Some(h) = err_reader {
                        let _ = h.join();
                    }
                    return Err(format!(
                        "UI Automation timed out after {}s",
                        PS_TIMEOUT.as_secs()
                    ));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => return Err(format!("wait failed: {e}")),
        }
    };

    let stdout = String::from_utf8_lossy(
        &out_reader
            .map(|h| h.join().unwrap_or_default())
            .unwrap_or_default(),
    )
    .into_owned();
    let stderr = String::from_utf8_lossy(
        &err_reader
            .map(|h| h.join().unwrap_or_default())
            .unwrap_or_default(),
    )
    .into_owned();

    if status.success() {
        Ok(stdout.trim_end().to_string())
    } else {
        let msg = if stderr.trim().is_empty() {
            stdout.trim().to_string()
        } else {
            stderr.trim().to_string()
        };
        Err(if msg.is_empty() {
            format!("powershell exited {}", status.code().unwrap_or(-1))
        } else {
            msg
        })
    }
}

/// Read the UIA tree (control view) of a target window as text. `window` is a
/// case-insensitive title substring; when None, the current foreground window.
pub fn inspect(window: Option<&str>) -> Result<String, String> {
    let mut env: Vec<(&str, &str)> = Vec::new();
    if let Some(w) = window {
        if !w.trim().is_empty() {
            env.push(("CAMELID_UIA_WINDOW", w));
        }
    }
    run_ps(INSPECT_SCRIPT, &env)
}

/// Find a control by name (exact, then case-insensitive substring) in the target
/// window and invoke it (InvokePattern), falling back to a center click.
pub fn click(window: Option<&str>, name: &str) -> Result<String, String> {
    let mut env: Vec<(&str, &str)> = vec![("CAMELID_UIA_NAME", name)];
    if let Some(w) = window {
        if !w.trim().is_empty() {
            env.push(("CAMELID_UIA_WINDOW", w));
        }
    }
    run_ps(CLICK_SCRIPT, &env)
}

/// Capture the primary screen to a PNG at `path`.
pub fn screenshot(path: &Path) -> Result<String, String> {
    let p = path.to_string_lossy();
    run_ps(SCREENSHOT_SCRIPT, &[("CAMELID_SHOT_PATH", p.as_ref())])
}

// --- trusted embedded scripts (params arrive via env, never interpolated) ----

#[rustfmt::skip]
const INSPECT_SCRIPT: &str = r#"
$ErrorActionPreference='Stop'
Add-Type -AssemblyName UIAutomationClient,UIAutomationTypes
$AE=[System.Windows.Automation.AutomationElement]
$want=$env:CAMELID_UIA_WINDOW
if($want){
  $root=$null
  foreach($w in $AE::RootElement.FindAll('Children',[System.Windows.Automation.Condition]::TrueCondition)){
    if($w.Current.Name -and $w.Current.Name.ToLower().Contains($want.ToLower())){$root=$w;break}
  }
  if($null -eq $root){Write-Error "no top-level window whose title contains '$want'";exit 1}
}else{
  $sig='[DllImport("user32.dll")] public static extern System.IntPtr GetForegroundWindow();'
  $U=Add-Type -MemberDefinition $sig -Name Fg -Namespace Cam -PassThru
  $root=$AE::FromHandle($U::GetForegroundWindow())
}
"window: $($root.Current.Name)"
$all=$root.FindAll('Descendants',[System.Windows.Automation.Condition]::TrueCondition)
$max=100
$cnt=[Math]::Min($all.Count,$max)
for($i=0;$i -lt $cnt;$i++){
  $c=$all.Item($i).Current
  $ct=$c.ControlType.ProgrammaticName -replace '.*\.',''
  $r=$c.BoundingRectangle
  if($r.IsEmpty -or [double]::IsInfinity($r.X) -or [double]::IsInfinity($r.Width)){
    "[$i] $ct | $($c.Name) | (offscreen)"
  }else{
    "[$i] $ct | $($c.Name) | @$([int]$r.X),$([int]$r.Y) $([int]$r.Width)x$([int]$r.Height)"
  }
}
if($all.Count -gt $max){"... ($($all.Count-$max) more elements; pass a window filter to narrow)"}
"#;

#[rustfmt::skip]
const CLICK_SCRIPT: &str = r#"
$ErrorActionPreference='Stop'
Add-Type -AssemblyName UIAutomationClient,UIAutomationTypes
$AE=[System.Windows.Automation.AutomationElement]
$want=$env:CAMELID_UIA_WINDOW
if($want){
  $root=$null
  foreach($w in $AE::RootElement.FindAll('Children',[System.Windows.Automation.Condition]::TrueCondition)){
    if($w.Current.Name -and $w.Current.Name.ToLower().Contains($want.ToLower())){$root=$w;break}
  }
  if($null -eq $root){Write-Error "no top-level window whose title contains '$want'";exit 1}
}else{
  $sig='[DllImport("user32.dll")] public static extern System.IntPtr GetForegroundWindow();'
  $U=Add-Type -MemberDefinition $sig -Name Fg -Namespace Cam -PassThru
  $root=$AE::FromHandle($U::GetForegroundWindow())
}
$name=$env:CAMELID_UIA_NAME
$cond=New-Object System.Windows.Automation.PropertyCondition($AE::NameProperty,$name)
$el=$root.FindFirst('Descendants',$cond)
if($null -eq $el){
  foreach($d in $root.FindAll('Descendants',[System.Windows.Automation.Condition]::TrueCondition)){
    if($d.Current.Name -and $d.Current.Name.ToLower().Contains($name.ToLower())){$el=$d;break}
  }
}
if($null -eq $el){Write-Error "no control named '$name' in the target window";exit 1}
try{
  $p=$el.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern)
  $p.Invoke()
  "invoked: $($el.Current.Name)"
}catch{
  $r=$el.Current.BoundingRectangle
  if($r.IsEmpty -or [double]::IsInfinity($r.X) -or [double]::IsInfinity($r.Width)){Write-Error "control '$name' has no Invoke pattern and no on-screen rectangle to click";exit 1}
  $cx=[int]($r.X+$r.Width/2);$cy=[int]($r.Y+$r.Height/2)
  $sig2='[DllImport("user32.dll")] public static extern bool SetCursorPos(int X,int Y);[DllImport("user32.dll")] public static extern void mouse_event(uint f,int dx,int dy,uint d,System.IntPtr e);'
  $M=Add-Type -MemberDefinition $sig2 -Name Clk -Namespace Cam2 -PassThru
  [void]$M::SetCursorPos($cx,$cy)
  $M::mouse_event(0x0002,0,0,0,[System.IntPtr]::Zero)
  $M::mouse_event(0x0004,0,0,0,[System.IntPtr]::Zero)
  "clicked: $($el.Current.Name) @ $cx,$cy"
}
"#;

#[rustfmt::skip]
const SCREENSHOT_SCRIPT: &str = r#"
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Windows.Forms,System.Drawing
$b=[System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bmp=New-Object System.Drawing.Bitmap($b.Width,$b.Height)
$g=[System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($b.Location,[System.Drawing.Point]::Empty,$b.Size)
$path=$env:CAMELID_SHOT_PATH
$bmp.Save($path,[System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose();$bmp.Dispose()
"saved $($b.Width)x$($b.Height) screenshot to $path"
"#;
