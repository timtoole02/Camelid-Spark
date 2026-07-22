# Incident: hard system hang during BASALT Phase 0 (2026-07-16)

**Outcome:** Tim's machine became unresponsive and was recovered by holding the power button.
System down 14:28:36 → 14:36:24 local (Kernel-Power event 41, `BugcheckCode=0`,
`LongPowerButtonPressDetected=true`, EventLog 6008 "previous system shutdown was unexpected").
No BSOD, no WHEA errors, no Resource-Exhaustion-Detector event — signature of a hard hang +
manual power-off, not a kernel panic.

## Root-cause chain

1. **13:47** — the Phase 0 workflow's `refusal:baseline` subagent ran the pin's
   `llama-cli.exe -m qwen3-0.6b-NVFP4-basalt-refusal.gguf -p "..."` for a scripted sanity
   generation. In pin build 9632 (acd79d603) `llama-cli` is **conversation-only**: the agent's
   first attempt with `--no-conversation` was rejected with *"please use llama-completion
   instead"* (preserved at the head of `pin_sanity_excerpt.txt`), and the retry without the
   flag launched an interactive REPL.
2. The REPL hit EOF on its redirected stdin and spun in an infinite `> ` prompt loop —
   **7,017,535 bytes (~1.17 M prompt marks) of `pin_sanity.log` written by 13:48**.
3. **14:05** — the subagent was killed by an upstream API error (529) after ~17.6 min with the
   llama-cli call still in flight. On Windows, agent death does not kill in-flight child
   processes: the spinning `llama-cli` was **orphaned**.
4. The orchestrating session (me) read the dead agent's logs, noted the interactive misfire,
   but **did not sweep the orphaned process** before launching two follow-up agents at 14:11
   (both lightweight: file-metadata commands and HF API curls — ruled out as contributors from
   their transcripts).
5. **14:28** — machine unresponsive; power-cycled at ~14:35.

The proximate crasher is the orphaned spinning `llama-cli`; the process-management failure
(no orphan sweep after an agent death) is the enabling mistake and was the orchestrator's.

## Corrective rules (recorded in the session memory as hard rules)

- Scripted llama.cpp generation uses **`llama-completion.exe`, never `llama-cli.exe`**
  (build 9632+ REPL spins on EOF/closed stdin). Every scripted model run carries an explicit
  timeout and kill-by-PID.
- **After any agent/task dies mid-run, the first action is a child-process sweep**
  (`Get-Process` match on `llama|camelid`, StartTime vs the dead agent's window, kill by PID —
  never blanket taskkill; the desktop app runs a camelid sidecar).
- Subagent prompts that include model runs must embed the RAM-check, REPL-ban, and
  orphan-sweep rules verbatim.

## Evidence

- Kernel-Power 41 EventData (BugcheckCode=0, LongPowerButtonPressDetected=true) and
  EventLog 6008 — captured 2026-07-16 post-reboot.
- `pin_sanity_excerpt.txt` (this directory) — first 2 KB of the 7 MB spin log, including the
  `--no-conversation` rejection message; full log deleted after excerpting (scratchpad,
  7 MB of repeated `> ` prompts).
- Workflow transcript `wf_05e96fc0-bf3/agent-a68723d48c2a71a8e.jsonl` (session directory,
  not bundled) — records the exact llama-cli invocations.
- Post-reboot state check: no orphaned `llama*`/`camelid` processes; 5.5 GB free of 15.7 GB.

## Impact on Phase 0

No evidence was lost or corrupted. The NVFP4 artifact (`models/qwen3-0.6b-NVFP4-basalt-refusal.gguf`),
the quantize log, and all five recon lane outputs predate the hang and are intact. The two
follow-up lanes (refusal receipt completion; upstream NVFP4 checkpoint scout) were interrupted
after only read-only commands and can resume cleanly — with the corrected `llama-completion`
sanity recipe.
