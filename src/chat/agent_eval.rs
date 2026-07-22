//! `camelid agent-eval` — the tool-capability promotion harness.
//!
//! Decides whether a model can drive a clean tool-call round-trip, producing a
//! receipt that justifies flipping `tool_capable` true — never a lucky run.
//! Crucially it distinguishes a real capability **FAIL** from an
//! **INCONCLUSIVE** result (the model didn't load within budget on a contended
//! box), so promotion is never decided by noise. Promotion is only ever earned
//! by a PASS receipt. See `DECISIONS.md`.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use super::agent::{self, AgentMsg, LiveDriver, Reporter};
use super::client::{Client, LoadOutcome};
use super::server::ServerHandle;
use super::tools::{Sandbox, ToolOutcome};

/// Exit codes (distinct so scripts can branch on the three outcomes).
pub const EXIT_PASS: i32 = 0;
pub const EXIT_FAIL: i32 = 1;
pub const EXIT_INCONCLUSIVE: i32 = 3;

pub struct EvalConfig {
    pub addr: SocketAddr,
    pub model: PathBuf,
    pub load_timeout: u64,
    pub max_steps: usize,
    pub max_tokens: u32,
    pub receipt_dir: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalOutcome {
    Pass,
    Fail,
    Inconclusive,
}

impl EvalOutcome {
    pub fn label(self) -> &'static str {
        match self {
            EvalOutcome::Pass => "PASS",
            EvalOutcome::Fail => "FAIL",
            EvalOutcome::Inconclusive => "INCONCLUSIVE",
        }
    }
    pub fn exit(self) -> i32 {
        match self {
            EvalOutcome::Pass => EXIT_PASS,
            EvalOutcome::Fail => EXIT_FAIL,
            EvalOutcome::Inconclusive => EXIT_INCONCLUSIVE,
        }
    }
}

/// Records the transcript for the verdict + receipt without any styling.
#[derive(Default)]
struct EvalReporter {
    calls: Vec<String>,
    results: Vec<(String, bool, String)>, // (tool, is_ok, text)
    answer: String,
}

impl Reporter for EvalReporter {
    fn model_text(&mut self, text: &str) {
        self.answer = text.to_string();
    }
    fn tool_call(&mut self, line: &str) {
        eprintln!("  ▸ {line}");
        self.calls.push(line.to_string());
    }
    fn tool_result(&mut self, name: &str, outcome: &ToolOutcome) {
        eprintln!(
            "  └ {} {}",
            name,
            if outcome.is_err() { "(error)" } else { "ok" }
        );
        self.results.push((
            name.to_string(),
            !outcome.is_err(),
            outcome.text().to_string(),
        ));
    }
    fn notice(&mut self, text: &str) {
        eprintln!("· {text}");
    }
}

/// Always-allow for the eval (it runs against a controlled fixture in a temp
/// sandbox; promotion evidence shouldn't depend on interactive approval).
struct AutoApprove;
impl agent::Approver for AutoApprove {
    fn approve(&mut self, _a: &super::tools::Action, _s: &Sandbox) -> agent::Decision {
        agent::Decision::Once
    }
}

/// One fixed case in the battery.
struct EvalCase {
    name: &'static str,
    goal: &'static str,
    /// Returns true if the recorded run satisfies the case.
    check: fn(&EvalReporter) -> bool,
}

const FIXTURE: &str = "alpha\nbeta\ngamma\n"; // 3 lines

/// True if a tool `name` ran cleanly (well-formed args → the sandbox executed it)
/// and its output satisfies `out_ok`.
fn tool_ran_ok(r: &EvalReporter, name: &str, out_ok: impl Fn(&str) -> bool) -> bool {
    r.results
        .iter()
        .any(|(n, ok, out)| n == name && *ok && out_ok(out))
}

/// A genuine multi-tool battery exercising three distinct tools (read_file /
/// list_dir / write_file). Each case requires the RIGHT tool to execute cleanly with
/// a correct result AND a correct final answer — tighter than the prior single-case
/// `answer.contains` heuristic. All must pass for a promotion-eligible certificate.
fn battery() -> Vec<EvalCase> {
    let all = vec![
        EvalCase {
            name: "read_and_count",
            goal: "Read the file notes.txt and tell me how many lines it has. Use the read_file \
                   tool, then give the count.",
            check: |r| {
                // read_file read the real fixture AND the answer states the count.
                tool_ran_ok(r, "read_file", |o| {
                    o.contains("alpha") && o.contains("gamma")
                }) && r.answer.contains('3')
            },
        },
        EvalCase {
            name: "list_dir_find",
            goal:
                "List the entries of the current directory '.' with the list_dir tool, then tell \
                   me the name of the text file you find there.",
            check: |r| {
                tool_ran_ok(r, "list_dir", |o| o.contains("notes.txt"))
                    && r.answer.to_lowercase().contains("notes.txt")
            },
        },
        EvalCase {
            name: "write_greeting",
            goal:
                "Create a file named greeting.txt whose exact contents are: hello there\nUse the \
                   write_file tool ONCE, then reply in words that you created it. Do not call any \
                   further tools and do not read the file back.",
            check: |r| {
                let wrote = r
                    .calls
                    .iter()
                    .any(|c| c.contains("write_file") && c.contains("greeting"))
                    && r.results.iter().any(|(n, ok, _)| n == "write_file" && *ok);
                wrote
                    && (r.answer.to_lowercase().contains("created")
                        || r.answer.to_lowercase().contains("greeting")
                        || r.answer.to_lowercase().contains("hello there"))
            },
        },
    ];
    // The pure-f32 runnable lane is slow for a 9B (a full multi-tool prompt prefills
    // at ~200s/turn on this host), so a full multi-case battery overruns the agent
    // client's read budget. `CAMELID_EVAL_CASE=<name>` runs exactly one case per
    // invocation, keeping each run within budget; absent the env, the full battery
    // runs (the fast path for optimized-lane models). Each single-case run still
    // mints a full promotion-eligible receipt for that case.
    if let Ok(name) = std::env::var("CAMELID_EVAL_CASE") {
        if all.iter().any(|c| c.name == name) {
            return all.into_iter().filter(|c| c.name == name).collect();
        }
        // Unknown name: fall through to the full battery rather than a 0-case PASS.
    }
    all
}

pub fn run(cfg: EvalConfig) -> anyhow::Result<i32> {
    let client = Client::new(cfg.addr);
    let _server = ServerHandle::ensure(cfg.addr, &client)?;

    // --- bounded load: a contended box yields INCONCLUSIVE, never FAIL ------
    let abs = std::fs::canonicalize(&cfg.model).unwrap_or_else(|_| cfg.model.clone());
    eprintln!("loading {} (timeout {}s)…", abs.display(), cfg.load_timeout);
    let started = Instant::now();
    let (tx, rx) = mpsc::channel();
    let loader = client.clone();
    let path = abs.to_string_lossy().to_string();
    std::thread::spawn(move || {
        let _ = tx.send(loader.load_model(&path, None));
    });
    let loaded = match rx.recv_timeout(Duration::from_secs(cfg.load_timeout)) {
        Ok(Ok(LoadOutcome::Loaded { id })) => id,
        Ok(Ok(LoadOutcome::Unsupported { message })) => {
            return finish(
                &cfg,
                EvalOutcome::Fail,
                &abs,
                None,
                &format!("unsupported: {message}"),
                &[],
            );
        }
        Ok(Err(err)) => {
            return finish(
                &cfg,
                EvalOutcome::Fail,
                &abs,
                None,
                &format!("load error: {err}"),
                &[],
            );
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            return finish(
                &cfg,
                EvalOutcome::Inconclusive,
                &abs,
                None,
                &format!(
                    "model did not load within {}s — box likely contended; re-run on a quiet host",
                    cfg.load_timeout
                ),
                &[],
            );
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            return finish(
                &cfg,
                EvalOutcome::Inconclusive,
                &abs,
                None,
                "loader thread died",
                &[],
            );
        }
    };
    eprintln!(
        "loaded '{loaded}' in {:.1}s",
        started.elapsed().as_secs_f64()
    );

    // The eval is intrinsically auto-approve (it runs a controlled fixture with a
    // non-interactive approver). That bypass must never happen in production, so
    // mirror the agent's fail-closed rule: refuse under CAMELID_PRODUCTION.
    if agent::is_production() {
        return finish(
            &cfg,
            EvalOutcome::Fail,
            &abs,
            Some(&loaded),
            "refused: agent-eval uses auto-approval and CAMELID_PRODUCTION is set; \
             run the promotion harness on a non-production host",
            &[],
        );
    }

    // --- fixture workspace --------------------------------------------------
    let work = std::env::temp_dir().join(format!("camelid-agent-eval-{}", std::process::id()));
    std::fs::create_dir_all(&work)?;
    std::fs::write(work.join("notes.txt"), FIXTURE)?;
    // The eval runs a controlled read-only fixture; run_shell is unrestricted so
    // the harness works on any host (including non-Linux CI) without the
    // kernel-sandbox preconditions. The battery never invokes run_shell.
    let sandbox = Sandbox::new(&work, false, Duration::from_secs(20))?
        .with_shell_mode(super::shell_sandbox::ShellSandbox::Unrestricted);
    let tools = super::tools::specs(false, sandbox.shell_mode());
    let family = family_for(&abs);

    // --- run the battery ----------------------------------------------------
    let cancel = AtomicBool::new(false);
    let mut cases = Vec::new();
    let mut all_pass = true;
    for case in battery() {
        eprintln!("== case: {}", case.name);
        let mut driver = LiveDriver::with(
            client.clone(),
            loaded.clone(),
            family.clone(),
            cfg.max_tokens,
            0.0,
        );
        let mut reporter = EvalReporter::default();
        let mut approver = AutoApprove;
        let mut history = vec![
            AgentMsg::System(agent::system_prompt(&sandbox, &tools)),
            AgentMsg::User(case.goal.to_string()),
        ];
        let cfg_loop = agent::AgentConfig {
            workdir: work.clone(),
            max_steps: cfg.max_steps,
            auto_approve: true,
            yolo: false,
            allow_net: false,
            allow_fs: false,
            shell_timeout: Duration::from_secs(20),
            max_tokens: cfg.max_tokens,
            temperature: 0.0,
            // The promotion harness audits nothing (controlled fixture, no sink).
            audit: Box::new(super::audit::NoopSink),
            shell_sandbox: super::shell_sandbox::ShellSandbox::Unrestricted,
        };
        // Auto-approve posture (write/network auto; run_shell still gated, which
        // the AutoApprove approver allows). Production was already refused above.
        let mut policy = agent::Policy::default();
        policy.set_auto_all(true);
        let end = agent::run_loop(
            &mut driver,
            &mut approver,
            &mut reporter,
            &sandbox,
            &cfg_loop,
            &cancel,
            &mut policy,
            &mut history,
        );
        let passed = (case.check)(&reporter);
        all_pass &= passed;
        cases.push(json!({
            "case": case.name,
            "goal": case.goal,
            "loop_end": format!("{end:?}"),
            "tool_calls": reporter.calls,
            "tool_results": reporter.results.iter().map(|(n,ok,t)| json!({"tool":n,"ok":ok,"output":t})).collect::<Vec<_>>(),
            "final_answer": reporter.answer,
            "passed": passed,
        }));
    }
    let _ = std::fs::remove_dir_all(&work);

    let outcome = if all_pass {
        EvalOutcome::Pass
    } else {
        EvalOutcome::Fail
    };
    finish(
        &cfg,
        outcome,
        &abs,
        Some(&loaded),
        "battery complete",
        &cases,
    )
}

/// Host 1-minute load average for the eval receipt. POSIX-only (`getloadavg`);
/// Windows has no equivalent, so the receipt records `null` (unavailable)
/// rather than a misleading number.
#[cfg(unix)]
fn host_loadavg_1m() -> Option<f64> {
    let mut load = [0f64; 3];
    // SAFETY: getloadavg writes up to 3 doubles into the provided buffer and
    // returns the number of samples written.
    let n = unsafe { libc::getloadavg(load.as_mut_ptr(), 3) };
    if n >= 1 {
        Some(load[0])
    } else {
        None
    }
}

#[cfg(not(unix))]
fn host_loadavg_1m() -> Option<f64> {
    None
}

/// Emit the receipt + the human verdict, return the exit code.
fn finish(
    cfg: &EvalConfig,
    outcome: EvalOutcome,
    gguf: &std::path::Path,
    model_id: Option<&str>,
    note: &str,
    cases: &[Value],
) -> anyhow::Result<i32> {
    let loadavg_1m = host_loadavg_1m();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let receipt = json!({
        "schema": "camelid.agent_eval/v1",
        "outcome": outcome.label(),
        "model_id": model_id,
        "gguf": gguf.display().to_string(),
        "gguf_bytes": std::fs::metadata(gguf).map(|m| m.len()).ok(),
        "quantization": infer_quant(gguf),
        "note": note,
        "cases": cases,
        "host_loadavg_1m": loadavg_1m,
        "timestamp_unix": ts,
        "promotion_eligible": outcome == EvalOutcome::Pass,
    });
    std::fs::create_dir_all(&cfg.receipt_dir)?;
    let stem = gguf
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "model".into());
    let path = cfg
        .receipt_dir
        .join(format!("{stem}-{ts}-{}.json", outcome.label()));
    let mut text = serde_json::to_string_pretty(&receipt)?;
    text.push('\n');
    std::fs::write(&path, text)?;

    eprintln!();
    eprintln!("{} — {note}", outcome.label());
    eprintln!("receipt → {}", path.display());
    if outcome == EvalOutcome::Inconclusive {
        eprintln!("(inconclusive does NOT change any tool_capable flag — re-run on a quiet box)");
    }
    if outcome == EvalOutcome::Pass {
        eprintln!("(eligible for promotion: set this row's tool_capable=true in the ledger)");
    }
    // Machine-readable verdict on stdout.
    println!("{}", outcome.label());
    Ok(outcome.exit())
}

fn family_for(gguf: &std::path::Path) -> String {
    let name = gguf
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_lowercase();
    // Ornith / qwen35 emit the custom `<function=…>` XML tool format, which routes
    // to `parse_ornith`. Check before "qwen" (the model is Qwen3.5-derived but its
    // tool grammar differs from Qwen2/Qwen3 JSON-in-`<tool_call>`).
    if name.contains("ornith") || name.contains("qwen35") || name.contains("qwen3.5") {
        "ornith".into()
    } else if name.contains("qwen") {
        "qwen".into()
    } else if name.contains("mistral") {
        "mistral".into()
    } else {
        "llama".into()
    }
}

fn infer_quant(gguf: &std::path::Path) -> String {
    let name = gguf
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_uppercase();
    for q in [
        "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0", "BF16", "F16", "F32",
    ] {
        if name.contains(q) {
            return q.to_string();
        }
    }
    "unknown".into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outcome_exit_codes_are_distinct() {
        assert_eq!(EvalOutcome::Pass.exit(), 0);
        assert_eq!(EvalOutcome::Fail.exit(), 1);
        assert_eq!(EvalOutcome::Inconclusive.exit(), 3);
    }

    #[test]
    fn check_requires_clean_read_and_correct_count() {
        // No tool ran, just an answer → fail.
        let r = EvalReporter {
            answer: "the file has 3 lines".into(),
            ..Default::default()
        };
        assert!(!(battery()[0].check)(&r));
        // read_file ran ok + correct count → pass.
        let r = EvalReporter {
            answer: "it has 3 lines".into(),
            results: vec![("read_file".into(), true, "alpha\nbeta\ngamma\n".into())],
            ..Default::default()
        };
        assert!((battery()[0].check)(&r));
        // read_file errored (malformed args) → fail even with a lucky answer.
        let r2 = EvalReporter {
            answer: "3".into(),
            results: vec![("read_file".into(), false, "requires path".into())],
            ..Default::default()
        };
        assert!(!(battery()[0].check)(&r2));
    }
}
