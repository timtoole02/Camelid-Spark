# Redaction note — phase3/proxy.log (2026-07-15)

`proxy.log` (the logging-proxy capture of the opencode end-to-end run described in
`e2e-validation.md`) contained operator home-directory paths inside one logged
request body (line 40, a `REQ_BODY` JSON line embedding the client's system prompt
and tool descriptions). `scripts/audit-evidence-bundle-privacy.mjs` flagged the
forward-slash occurrences as `mac_home_path` findings.

## What was redacted

Six occurrences on that single line, all replaced with the placeholder
`REDACTED_HOME` substituted for the home-directory path component (the `Users`
segment plus the account name):

- 1 × a `file:///` URL embedding the operator's forward-slash Windows home path
  (a tool `<location>` pointing back into this repo checkout) — now
  `file:///C:/REDACTED_HOME/Camelid/…`
- 3 × JSON-escaped backslash Windows home paths (an AppData temp path and two
  repo-checkout paths) — now the `C:\\REDACTED_HOME\\…` form
- 2 × generic "My Documents" documentation-example home paths quoted inside the
  client's embedded system-prompt text (not a real leak; redacted uniformly so
  the privacy audit's home-path pattern stays quiet)

Every other byte of the file is unchanged. No parity/evidence semantics were
altered: the SSE frames, usage fields, token counts, and response bodies the
bundle's conclusions rest on are untouched, and no checksum manifest in this
bundle covers `proxy.log`.

## Git status of the log (why there is no history concern)

`proxy.log` has never been tracked or committed — it is ignored by the repo-root
`.gitignore` (`*.log`) and exists only on the capture host's disk. The leak was
therefore local-only, not public, and no git-history rewriting is needed. This
note is the tracked record of the redaction. (Related pre-existing quirk, left
as-is: the tracked `e2e-validation.md` cites `proxy.log` as evidence even though
the log is untracked; whether to commit a scrubbed copy is a maintainer decision.)

After redaction, `node scripts/audit-evidence-bundle-privacy.mjs` reports
`finding_count: 0`.
