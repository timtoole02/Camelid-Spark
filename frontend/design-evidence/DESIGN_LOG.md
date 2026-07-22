# Camelid Frontend Overhaul — Design Log

One entry per phase: what changed, what was tried and rejected, and any
deviation from the build spec with reasoning.

---

## Phase 1 — Design system and visual identity (2026-06-12)

Branch: `feat/frontend-phase-1-design-system` (on top of Phase 0 baseline `c837d05`).

### Concept

Instrument panel, not chat toy. The identity is carried by three things:
1. **The Evidence Chip** (`components/ui/EvidenceChip.jsx` + `lib/evidenceStatus.js` +
   `styles/evidence.css`) — every support/evidence claim now renders through one
   component with a row-scoped mono label, a state icon (color-independent), and a
   click-to-verify popover citing the claim source (capability row id, scope, contract
   copy). Presentation-only: it displays gate state, never computes it.
2. **The status color doctrine** — copper is reserved exclusively for
   supported/verified; desaturated amber for evidence-only/bounded; cool steel blue
   for interactive/informational; muted slate for unsupported (a normal state, never
   alarming); red for errors only.
3. **Mono as a first-class citizen** — IBM Plex Mono carries row ids, statuses,
   endpoint paths, pin-badge evidence lattices, and chip labels.

### Token table (dark canonical / light override)

| Token | Dark | Light | Role |
| --- | --- | --- | --- |
| `--color-bg` | `#0e1216` | `#f6f8fa` | near-black blue-grey base |
| `--color-bg-elevated` | `#141a21` | `#ffffff` | cards, popovers |
| `--color-surface-strong` | `#1c242e` | `#e1e8ef` | strongest panel |
| `--color-text` | `#dde5ed` | `#1b2530` | body text |
| `--color-text-muted` | `#9caab9` | `#4d5d6d` | secondary |
| `--color-text-faint` | `#8190a0` | `#56656f` | captions (AA-fixed) |
| `--color-accent` | `#8fb6dc` | `#2b5c84` | steel blue, interactive/info |
| `--color-verified` | `#dfa371` | `#96531c` | **copper — supported/verified only** |
| `--color-evidence` | `#cfb56a` | `#75601a` | desaturated amber — bounded evidence |
| `--color-planned` | `#8b9aab` | `#546576` | planned/groundwork/target |
| `--color-unsupported` | `#8694a3` | `#5a6a79` | calm honest unsupported |
| `--color-ready` | `#8cc9a0` | `#20713c` | operational runtime health |
| `--color-warning` | `#d8b86a` | `#82610e` | operational warnings |
| `--color-error` | `#e9928a` | `#b3261e` | errors only |
| `--font-display` | Space Grotesk Variable | — | view titles, wordmark |
| `--font-ui` | Inter Variable | — | body |
| `--font-mono` | IBM Plex Mono 400/500/600 | — | ids, statuses, paths |
| `--radius-sm/md/lg/xl/2xl` | 6/8/12/16/20px | — | tightened from 8/12/18/24/32 |

Each evidence color also has `-soft` (fill) and `-border` variants. Full list in
`src/styles/tokens.css`. `scripts/contrast-check.mjs` (`npm run smoke:contrast`)
asserts WCAG AA for all 124 text/status-on-surface pairs in both themes — passing.

### What changed

- `tokens.css` rewritten dark-first: dark is the canonical `:root` palette;
  light is the `[data-theme="light"]` + `prefers-color-scheme: light` override
  (the two light blocks stay byte-identical, mirroring the old dark-block rule).
  All legacy variable names kept so every existing sheet still renders.
- Theme default changed from `system` to `dark` (`useTheme.js`); cycle order is
  now dark → light → system. System-following still works exactly as before.
- Fonts self-hosted via Fontsource (`main.jsx` imports); the Google Fonts CDN
  `@import` was removed. **Baseline correction:** BASELINE.md §7 claimed the
  baseline had "no CDN calls" — wrong; `tokens.css:11` imported Plus Jakarta
  Sans/Outfit from fonts.googleapis.com at runtime. Phase 1 makes the offline
  claim actually true.
- Evidence Chip replaced ad-hoc claim renders in: TopBar (support gate, now
  visible on every tab, not just chat), chat composer status strip, ApiView
  (feature rows, compatibility rows, selected-model contract), SystemView
  (rows + guarded features), AnalyticsView (guarded rows/features), ModelsView
  (tracked-row status pills, 3B acceptance card, external-routing pill).
  Operational model-state pills (downloading/loaded/needs-attention) stay
  `status-pill` — restyled mono/uppercase but still operational-green: runtime
  state is not a support claim and must not look like one (I4 in reverse).
- `pin-badge` evidence lattices restyled to mono micro-labels; their "ready"
  tone now maps to evidence amber, not green — a passing bounded pack is
  bounded evidence, not blanket readiness.
- Brand sparkle re-colored from the legacy purple Gemini-style gradient to the
  instrument gradient (steel → brass → copper), matching `--camelid-aurora`.
- Shell: hairline-bordered glass top bar (56px) with the gate chip + model
  button; display-face titles; tightened radii; backend-unreachable
  empty/error states added to ApiView and SystemView (the views that had none).
- New tooling: `scripts/contrast-check.mjs`, `scripts/capture-views.mjs`
  (puppeteer-core against system Chrome; seeds `camelid-theme` before boot;
  forces a fresh document load per view because hash routing is mount-time).
  `puppeteer-core` added as devDependency; fonts are the only new runtime deps.

### Tried and rejected

- One warm accent for both "supported" and "evidence": rejected — the spec's
  distinction (copper vs desaturated amber) is the product's honesty made
  visible; merging them re-blurs supported vs bounded-evidence.
- Keeping light as the canonical `:root` palette: rejected; dark-first means the
  canonical definition is dark, and it also makes the no-attribute default dark.
- Uppercase mono for all pin-badges: rejected as too loud across 100+ badges;
  pin-badges stay lowercase mono, only EvidenceChip/status-pill labels are caps.
- Replacing every pin-badge with EvidenceChips: deferred — the per-lane evidence
  lattice in ModelsView is Phase 4's drill-down material; converting 100+
  badges now would duplicate that work without the contract-driven popover data.

### Gate results

- `npm run build` clean — JS 145.85 kB gz (baseline 143.76; Phase 7 budget 229.9),
  CSS 19.45 kB gz. Fonts ship as separate self-hosted woff2 assets.
- Smokes: streaming, model-state, capability-readiness, 3b-closure, integration,
  observatory all PASS; `smoke:tiny` PASS against a live backend with the
  unsupported tiny fixture — chat verifiably stays blocked, with the topbar and
  composer chips honestly reading "no matching COMPATIBILITY.md row".
- Readiness-gate logic: `git diff` over `chatGate.js`, `capabilities.js`,
  `modelState.js`, `capabilityReadiness.js` is empty — byte-identical to the
  Phase 0 record.
- `smoke:ui`: unchanged pre-existing failure (stale README-copy assertion).
  Deeper finding while reading it: everything after that first failing assertion
  has been dead for a while — the tail reads `src/styles/components.css` (deleted
  before this overhaul) and asserts pre-redesign TopBar internals
  (`exactHintDetail`) that no longer exist. The smoke needs a deliberate
  re-baseline as its own change; nothing was deleted or weakened in Phase 1.
- Contrast: all 124 pairs AA in both themes (`npm run smoke:contrast`).
- Screenshots: `design-evidence/phase-1/` — all 10 views × dark/light × 1440/390.
  Self-critique fixes during capture: purple sparkle (fixed), capture script
  initially re-shot the chat view 40× because hash navigation doesn't remount
  (fixed in the harness). Known carry-over: observatory run-details panel still
  overflows at 390px (recorded at baseline; Phase 7 responsive audit scope).

---

## Phase 2 — Chat experience (2026-06-12)

Branch: `feat/frontend-phase-2-chat-experience` (pre-work commits 547a233 + 755a7b1,
feature commit follows this entry).

### Pre-work (committed separately)

- BASELINE.md errata appended (offline-fonts claim, smoke:ui health overstatement).
- Deleted three zero-importer pre-redesign orphans (AppSidebar, ConversationDeleteDialog,
  GlobalNotice); scrubbed stale components.css comment references.
- capture-views.mjs gained a sha256 self-check that fails the run when two captured
  views are pixel-identical (the 40-identical-screenshots failure mode); negative-tested.
- smoke:ui re-baselined: full port/retire ledger in commit 755a7b1; negative-tested
  (injected copper-token violation fails the run). It is now in the standing gate set.

### What shipped

- **Markdown**: tables (header detection + instrument-grid styling), ordered lists with
  preserved start numbers, links (http/https/mailto only — any other scheme degrades to
  visible plain text), italics/strikethrough, and per-language syntax highlighting
  (python/rust/bash/json families joined js/html/css) — all still rendered as React
  elements, so there is no innerHTML path at all. New `smoke:markdown` (SSR via vite
  ssrLoadModule) covers tables/lists/links/highlighting/injection-escaping.
- **Metadata footer** on completed assistant messages: model id, the Evidence Chip for
  the row that was active at send time (row id + status snapshot, never paths), token
  counts labeled `usage` (backend) vs `usage est.` (client estimate), TTFT, tok/s,
  duration, and a persistent CLIENT-MEASURED tag (I4).
- **Message actions**: copy (existing), regenerate (truncates the thread at the prior
  user turn and resends through the same gate-checked sendMessage path — no second send
  path exists), edit-and-resend on user rows (inline textarea, Enter resends, Esc cancels).
- **Conversation export** (Markdown/JSON) from Chat history: field-whitelist serializer
  (`lib/conversationExport.js`) so filesystem paths are excluded by construction; smoke:ui
  now feeds it a conversation salted with path fields and asserts none survive (I7), plus
  the telemetry-not-evidence note in both formats.
- **Generation controls drawer**: system-prompt editor with local presets (leads the
  request; the code-first policy prompt appends behind it), and the sampling lane —
  every parameter renders as a guarded "no contract row" Evidence Chip because
  /api/capabilities advertises no sampling rows (BACKEND_ASKS.md #1). The unlock path is
  fully wired (`lib/samplingContract.js`: exact-id row match, per-model persistence,
  contract-gated request overrides) but inert until the contract grows.
- **Keyboard**: Enter/Shift+Enter (existing), Esc cancels stream (existing), Cmd/Ctrl+K
  stub jumps to the composer and says the palette ships in Phase 7 (no fake palette).

### Tried and rejected

- marked/DOMPurify/highlight.js: rejected — the renderer is already React-element-based
  (sanitized by construction); extending it keeps runtime deps at zero and the offline
  property trivial.
- Editable sampling controls with a "values are experimental" disclaimer: rejected —
  I3 says guarded surfaces, not caveated live ones. Controls unlock per-parameter only
  when the contract advertises the exact row.
- A second "regenerate" request path in the hook: rejected — regenerate/edit-resend
  reuse sendMessage with truncate+override options so the chat gate, code-first policy,
  streaming, and abort handling stay single-sourced.

### Gate results

- Build clean; JS **150.66 kB gz** (Phase 1: 145.85; ceiling 229.9), CSS 20.03 kB gz.
- All 10 smokes green: streaming, model-state, capability-readiness, 3b-closure,
  integration (one regex made markup-tolerant for the new keyword highlighting — the
  escaped-content assertion is intact), observatory, **markdown (new)**, **ui
  (re-baselined, now standing)**, contrast, tiny (chat verifiably blocked for the
  unsupported fixture).
- Readiness-gate libs: empty git diff.
- Live manual pass against the loaded supported 3B row, driven through the real UI
  (p2-manual-*.png): chat unlocked; mid-stream abort renders the interrupted warning
  and keeps partial content; metadata footer renders with TTFT 336ms / tok/s / supported
  chip; regenerate replaced the reply without duplicating turns. Structured SSE
  `event: error` mid-stream stays covered by smoke:streaming + smoke:integration
  (the backend offers no way to trigger one on demand — noted, not hand-waved).
- Screenshots: chat + history × dark/light × 1440/390 via the harness (self-check
  passed, 8 distinct) + live-stream evidence set + controls-drawer shot.

---

## Phase 3 — Model management (2026-06-12)

Branch: `feat/frontend-phase-3-model-management`.

### What shipped

- **Card-level Evidence Chips on local GGUFs** (`ModelCardEvidence` in ModelsView):
  every local model card — both "Local runtime" and "Still needs setup" — resolves its
  exact model/quant against the live contract. Matched rows show their real status
  chip; unmatched models get the calm muted "no exact supported row" chip plus a
  "view the compatibility ledger" jump (currently #api; re-targets to the Phase 4 view
  when it exists). Not an error state (I2).
- **Model inspector drawer** (`components/models/ModelInspector.jsx`): fetches
  `/api/models/current` + `/api/models/tokenizer` on open. File section (path with a
  "local-only display; never exported" note, GGUF version, quant from file_type,
  tensor count/offsets, model-native context length explicitly caveated against the
  bounded-pack contract), tokenizer section (model, vocab size, special ids, config
  flags), and the full 35-key KV grid with long values summarized client-side (the raw
  payload is 5.6 MB of vocab/merges arrays — rendered as "[…, N items]"). A banner
  chip pins the framing: descriptive metadata — not support evidence (I2/I4).
- **Tokenizer playground** (`components/models/TokenizerPlayground.jsx`): live
  encode/decode against `/api/models/tokenizer/{encode,decode}` (feature row
  `tokenizer_encode_decode`, cited by the panel chip). Text → token count, per-token
  id+piece chips (per-id decode — faithful for BPE since decode is a fixed id→bytes
  map; capped at 200 with an honest truncation note), add_special/parse_special
  toggles, and a byte-exact round-trip verdict computed over the full sequence.
  Works whenever a tokenizer is loaded — chat support not required, and the chip copy
  says token output does not widen generation support.
- **Load/switch flow**: already had typed-guardrail error surfacing
  (getGuardrailErrorMessage → load_error + notice) and busy states; unchanged. The
  active model's "unmistakable everywhere" treatment comes from the Phase 1 topbar
  gate + composer chips + the active-model-card highlight.

### Tried and rejected

- Storing `/api/models/current` in dashboard state for the inspector: rejected — the
  payload is 5.6 MB; the drawer fetches on open and summarizes immediately instead of
  keeping vocab arrays resident in React state.
- Per-token pieces via incremental prefix decodes: rejected — O(n) requests with no
  correctness gain over per-id decode for BPE; per-id chunks of 16 keep it simple and
  the full-sequence round-trip still catches any normalization drift.
- Evidence chips on catalog-preview cards: deferred — catalog entries are not local
  GGUFs; their "Catalog quant:" labels already stay non-promotional, and Phase 4's
  ledger view is the right home for browsing claims.

### Gate results

- Build clean; JS **153.38 kB gz** (Phase 2: 150.66; ceiling 229.9).
- All 9 offline smokes green + `smoke:tiny` PASS; smoke:ui extended with Phase 3
  assertions (card chip presence + calm no-row copy + ledger link; inspector labeled
  not-support-evidence and barred from gate computation; playground cites
  tokenizer_encode_decode and disclaims generation support; both new components in
  the brand-hygiene sweep).
- Readiness-gate libs: empty git diff.
- Wrong-row demonstration through the real UI: tiny fixture loaded
  (generation_ready=true) → composer reads "Runtime ready, support gated ·
  tiny-generation · No matching COMPATIBILITY.md row", send stays locked with a
  draft present, and the library card shows the calm unsupported chip + ledger link
  (chat-blocked-tiny / library-blocked-tiny screenshots).
- Live checks against the loaded 3B: inspector renders 35 KV rows with arrays
  summarized and context length present; playground round-trips
  "Hello Camelid, parity is the product." at 10 tokens, byte-exact ✓.
- Screenshots: library × dark/light × 1440/390 (harness self-check passed) +
  inspector + playground + blocked-state evidence.

---

## Phase 4 — Compatibility & evidence explorer (2026-06-12)

Branch: `feat/frontend-phase-4-compatibility-explorer`. The signature view.

### What shipped

- **views/CompatibilityView.jsx** — the live ledger, new first-class tab
  (`#compatibility`, registered in HASH_TABS/VALID_TABS/TopBar/sidebar). Everything on
  the screen is the `/api/capabilities` payload at render time: the support-contract
  block (current gate / support policy / unsupported policy verbatim), a stat strip
  (14 rows · 9 supported · 5 "tracked, honestly not claimed"), and one ledger row per
  exact lane. Smoke-enforced: the view source contains zero hardcoded row ids or
  support statuses, and the integration smoke renders it against a mock contract
  (rows come from the mock; trap fields like broad_family lists must NOT render) and
  against a null contract (fail-closed "Ledger unavailable", zero rows).
- **Proven / Not claimed at equal visual weight** — two same-width columns per row;
  the not-claimed column renders the row's `full_support_blockers` copy verbatim with
  `support_scope` underneath. Supported rows get a copper left edge (rule lives in
  evidence.css — the copper-reservation smoke caught it in views.css, which is exactly
  what that assertion is for).
- **Per-row drill-down** — a 13-track evidence checklist (metadata/tokenizer/tensors/
  generation/prompt-token parity/frontend load/template-shape pack/bounded 512–8192
  context packs/perf-RSS), each an Evidence Chip citing the row id plus the
  `*_pack_id` evidence-bundle identifier where the contract advertises one; latest
  checked bucket → result, and the row's readiness-gate sentence.
- **Promotion path** — for non-supported rows only, the contract's `next_step` copy in
  a dashed planned-tinted panel, captioned "an honest checklist, not a promise."
- **Cross-linking** — every Evidence Chip in the app now carries "View in the evidence
  ledger →" in its popover whenever it cites a row id, dispatching a
  `camelid:open-ledger` event the app shell listens for (no prop drilling through
  dozens of chip sites). The ledger scrolls to, highlights, and auto-expands the row;
  api-feature ids resolve to the ledger's feature section. ModelCardEvidence's
  "view the compatibility ledger" link re-targeted from #api to the new view.
- **"How to read this ledger" explainer** in product voice: exact-row support, bounded
  packs ≠ native context, perf ≠ throughput promises, unsupported is a normal state.

### Tried and rejected

- Hash-fragment row addressing (#compatibility/<row>): rejected — hash routing is
  mount-time-only here; the event + focus-state approach deep-links from live chips
  without rearchitecting navigation.
- Rendering manifest paths from README/COMPATIBILITY.md copy: rejected — that would be
  doc-derived support claims the contract doesn't make. The contract exposes pack ids
  (cited); manifest references are BACKEND_ASKS.md #2 and render automatically when
  `*_pack_manifest` fields appear.

### Gate results

- Build clean; JS **156.08 kB gz** (Phase 3: 153.38; ceiling 229.9).
- 10/10 smokes green (integration + ui extended with the ledger assertions above);
  readiness-gate libs empty diff; `smoke:tiny` still proves fail-closed chat.
- Live UI checks: 14 rows render from the live contract with not-claimed on every
  row; drill-down shows 13 tracks with real pack ids; deep-link from a library
  tracked-row chip lands focused + auto-expanded on llama32_3b_instruct_q8_0. One
  honest negative: the composer chip with the tiny fixture loaded cites no row, so it
  correctly offers no ledger link.
- Screenshots: compatibility × dark/light × 1440/390 (self-check passed) + drill-down
  + deep-link focus shots in design-evidence/phase-4/.

---

## Phase 5 — API workbench (2026-06-12)

Branch: `feat/frontend-phase-5-api-workbench`.

### What shipped

- **components/api/ApiWorkbench.jsx + lib/apiExamples.js** — nine routes of the live
  surface as workbench cards: /v1/health, /v1/models, /v1/chat/completions,
  /v1/completions, /api/capabilities, /api/models/tokenizer/encode, and the three
  fail-closed routes (/v1/embeddings, /v1/responses, /v1/messages) rendered as typed
  guarded rows citing the fail_closed_native_compatibility_routes feature row.
- **Examples** in curl / Python / JS-fetch, pre-filled with the live API base and
  loaded model id, copy button per card. Generation examples mirror the chat request
  shape (greedy temperature=0, streaming). Gate evidence: the chat-completions curl
  example was extracted from the rendered page DOM and executed verbatim — it streamed
  SSE chunks from the live 3B.
- **Try-it gating (I1/I3)**: generation endpoints run only when the shared exact-row
  chat gate is green — including /v1/completions, where a found sharp edge made this
  matter: the raw backend route answers for ANY loaded model (it generated <unk> tokens
  from the unsupported tiny fixture), so the workbench card states explicitly that the
  route answers but the UI keeps generation examples gated like chat. Read-only routes
  run whenever the backend answers; fail-closed routes never run. Verified live in both
  directions: tiny fixture → both generation try-its guarded with typed copy while
  health stayed runnable; 3B loaded → chat try-it unlocked.
- **Request inspector (I4)**: rendered request, status, headers/total timings, pretty
  JSON bodies, and a timestamped SSE chunk log (capped at 80 lines with an honest
  truncation note) under a pinned "operational telemetry — not compatibility evidence"
  chip. Live run: 9 chunk lines, headers 162ms, total 1100ms.
- ApiView's four static endpoint cards were absorbed into the workbench; the
  readiness-gated curl block and every smoke-asserted gate string stayed live (one
  asserted phrase was restored into the section copy when the cards went away — the
  smoke caught it).

### Tried and rejected / deviations

- "Python · openai sdk" as the visible tab label: rejected by the pre-existing
  integration-smoke brand assertion on rendered markup. Deliberate resolution: the tab
  label is just "Python"; the SDK example itself (which must name the class it
  instantiates — that is technical compatibility content, not product copy) renders
  only when the tab is selected, so the default markup stays brand-clean.
  lib/apiExamples.js is consciously excluded from the brand sweep with an inline
  comment; the workbench component itself remains swept.
- Gating /v1/completions as 'tokenizer-level' (it technically runs for any loaded
  model): rejected — I1 says generation examples gate exactly like chat; a route that
  emits unsupported-model tokens is precisely what the gate exists to keep out of the
  paved path.

### Gate results

- Build clean; JS **159.41 kB gz** (Phase 4: 156.08; ceiling 229.9).
- 10/10 smokes green; integration smoke extended with both gating directions
  (blocked fixture → data-tryit-ready=false for both generation routes + typed copy +
  health true; green 3B fixture → chat try-it true); smoke:ui extended with workbench
  assertions. Readiness-gate libs: empty diff.
- Copy-paste gate: page-extracted curl ran verbatim against the live backend (SSE).
- Screenshots: api × dark/light × 1440/390 (self-check passed) + gated-state +
  SSE-inspector shots in design-evidence/phase-5/. Backend left re-gated on the tiny
  fixture (smoke:tiny re-run green at close).

---

## Phase 6 — Observability dashboard (2026-06-12)

Branch: `feat/frontend-phase-6-observability`.

### What shipped

- **lib/telemetryLog.js** — the session store. Records arrive ONLY from real traffic:
  the chat send path (success, interruption, and error all record), workbench try-it
  runs, and the live health polls. In-memory ring buffers (500 requests / 240 polls),
  nothing persists across reloads, no seeding path exists (smoke bars the obvious
  fabrication routes and the empty state promises "never seeds or invents data").
- **views/TelemetryView.jsx** (`#telemetry`, first-class tab): summary tiles
  (requests, error rate, median TTFT / tok/s / duration — all client-measured and
  labeled), SVG sparkline trends, per-model breakdown by model id (captioned: grouping
  implies nothing about support), backend reachability strip, and the request log.
- **Request log**: time, endpoint, model, outcome, duration, token counts; prompt
  content REDACTED by default with a per-session reveal toggle; Export JSON goes
  through a field whitelist that cannot include prompt content or paths — smoke-
  enforced behaviorally with a salted record (secret prompt + /Volumes path → absent
  from export, whitelisted fields + not-evidence note present).
- **Health with backoff**: the dashboard refresh loop became self-scheduling — 2.5s
  while the backend answers, doubling to a 20s ceiling on consecutive failures, reset
  on success. Every poll outcome (latency or failure) lands in the reachability strip.
- **I4 pinned everywhere**: page-level chip + per-panel captions; a smoke assertion
  bars perf numbers from rendering inside Evidence Chips in this view. Bounded
  perf/RSS contract evidence stays in the Compatibility ledger, explicitly pointed to.

### Tried and rejected

- Persisting telemetry to localStorage: rejected — "session metrics" should die with
  the session; persistence would also turn yesterday's numbers into ambient pseudo-
  evidence.
- Folding into AnalyticsView: rejected — Analytics is conversation usage over stored
  history; this is live operational traffic. Mixing them blurs the I4 boundary the
  spec draws.
- A separate health poller for the panel: rejected — the dashboard already polls; a
  second poller would double traffic and make the history lie about cadence. The
  existing loop gained backoff instead.

### Gate results

- Build clean; JS **163.18 kB gz** (Phase 5: 159.41; ceiling 229.9).
- 10/10 smokes green incl. the behavioral export-whitelist check; gate libs empty diff.
- "Demonstrably real requests" shown live end-to-end in one browser session: fresh
  session renders the empty state with only real health polls in the strip; one real
  chat send (3B, "telemetry check") + one workbench health try-it populate exactly 2
  log rows, the tiles (TTFT median 284ms · 3.9 tok/s · 387ms), and a 3B per-model row;
  prompt redacted by default, reveal toggle shows it. Failure history demonstrated in
  an isolated profile against a dead API base: 4 failed polls in 16s (backoff visibly
  stretching the cadence), red strip cells.
- Screenshots: telemetry × dark/light × 1440/390 + populated/empty/unreachable shots
  in design-evidence/phase-6/.

---

## Phase 7 — Polish, command palette, accessibility, performance (2026-06-12)

Branch: `feat/frontend-phase-7-polish`. The closing phase.

### What shipped

- **Command palette** (Cmd/Ctrl+K): navigate all 12 views, new conversation, theme
  cycle, switch model (hint stays gate-honest: "readiness still gates send"), and jump
  to any compatibility row — reusing the same `camelid:open-ledger` event the chips
  use, live-verified to land focused on the row. Combobox/listbox semantics, arrow/
  enter/esc keyboard model.
- **"?" shortcut overlay** documenting the full keyboard map (outside text fields).
- **Accessibility**: Lighthouse a11y **100 on chat, 98 on compatibility** (gate ≥95).
  The one real fix it surfaced: interactive Evidence Chips now carry an explicit
  aria-label (the topbar gate chip loses its visible label at mobile widths and was
  name-less). Composer status strip became a polite live region; heading order
  normalized in the ledger; icon-per-state chips (Phase 1) already satisfied
  color-independence.
- **Performance**: route-level code splitting — chat stays eager, the other 11 views
  load on first visit. Initial JS chunk **104.10 kB gz** (was 163.32 monolithic);
  total across all chunks **176.39 kB gz** vs the 229.9 budget (1.6× the 143.76
  baseline — met with 23% headroom). Long-conversation windowing (latest 60 turns +
  "show earlier" expander; the telemetry log was already windowed) instead of a
  virtualization dependency. Fixed an ineffective dynamic import in the poll loop.
- **Responsive**: the baseline-recorded observatory run-details overflow at 390px is
  fixed (panel stacks under the canvas ≤700px; live-measured 0px horizontal overflow).
  Full audit captured at 390/768/1024/1440.
- **Identity**: SVG favicon (instrument sparkle, steel→brass→copper on the dark base)
  + theme-color meta; wordmark already carried by Space Grotesk since Phase 1.
- **frontend/README.md**: new views/features/shortcuts section with the explicit
  statement that readiness-gate semantics are unchanged (smoke-asserted).

### Tried and rejected

- A virtualization library for message lists: rejected — windowing achieves the
  perf goal with zero dependencies and no scroll-anchoring edge cases during
  streaming.
- Chasing compatibility from 98 to 100: the residual flag is a heading-order
  nit inside contract-rendered sections; restructuring real content hierarchy for a
  scanner point wasn't worth bending the ledger's semantics. 98 ≥ 95 gate.

### Gate results

- 10/10 smokes green (smoke:ui extended with palette/overlay/code-split/README
  assertions); readiness-gate libs: empty diff — byte-identical through all 8 phases.
- Lighthouse a11y: chat 100, compatibility 98 (both ≥95).
- Bundle budget met: 176.39 kB gz total / 104.10 initial vs 229.9 ceiling.
- Final screenshot set: 12 views × dark/light × 1440/390 (48 shots, self-check
  distinct) + 24-shot responsive audit at 768/1024 + palette/shortcuts/observatory-
  fix evidence. design-evidence/phase-7/.

### Before / after (the whole overhaul)

Phase 0 baseline → Phase 7: a Gemini-styled chat shell with ad-hoc status badges and
a Google-Fonts CDN dependency became an instrument-panel operator console with one
claim component (the Evidence Chip, cited everywhere, deep-linked to a live-contract
evidence ledger), a chat surface with telemetry-honest footers and contract-gated
controls, a model inspector + tokenizer playground, a gated API workbench whose curl
examples run verbatim, a real-traffic-only session telemetry dashboard, a command
palette, AA-contrast-smoked dual themes on self-hosted fonts, and a 10-smoke gate
suite (from 8, one of which was dead) — at 104 kB gz initial JS against a 143.76 kB
baseline monolith, with the fail-closed chat gate byte-identical throughout.

---

## Final acceptance (2026-06-12, after Phase 7)

1. `npm run build` clean — initial JS 104.10 kB gz, total 176.39 kB gz. ✓
2. `smoke:streaming`, `smoke:contrast`, re-baselined `smoke:ui` green (with the full
   10-smoke suite). ✓
3. Backend up → `smoke:tiny` green: the unsupported fixture loads, reports
   generation_ready=true, and chat verifiably stays blocked. ✓
4. Supported-row manual pass (Llama 3.2 3B Instruct Q8_0 — the supported
   `supported_exact_row_smoke` row; the spec names TinyLlama, any supported exact row
   satisfies the gate-green condition): load ✓, inspector ✓, streaming chat ✓, Esc
   abort mid-stream renders interrupted state ✓, regenerate completes with telemetry
   footer ✓, conversation export (path-free, smoke-enforced) ✓, workbench try-it
   unlocked + request inspector ✓, telemetry dashboard populating from the session ✓,
   Compatibility ledger matching /api/capabilities exactly (14/14 row ids, 11/11
   feature rows) ✓.
5. frontend/README.md source-of-truth section re-read against the shipped app: every
   listed behavior holds — health/models/capabilities consumption, meta-as-descriptive,
   no native-route unlocks, load via /api/models/load, gate visible in the top bar on
   every tab with a direct jump to the contract (via the ledger), API tab first-class,
   readiness-gated examples, file_type quant normalization, exact-row wins shown
   row-scoped, streaming + typed SSE error handling, and the fail-closed chat gate.
   No drift found. ✓

The overhaul is complete: Phases 0–7 shipped, all invariants I1–I7 held at every
gate, and the readiness-gate libraries are byte-identical to the Phase 0 record.

---

## Phase 6.1 — Observatory rework: The Flow Bench (2026-06-12)

Branch: `feat/frontend-phase-6.1-flow-bench`.

### Defects found and fixed first (own commits; full detail in phase-6.1/DEFECTS.md)
1. Per-mount SSE store + unmount disconnect made every run invisible unless the view
   was open when it happened (and wiped state on navigation). Fixed: shared
   app-lifetime store bootstrapped from the app shell; smoke-guarded.
2. Connection churn (41 EventSource cycles / 21 mounts) — same root cause, same fix.
   Teardown itself was verified clean (0 rAF unmounted, opened==closed).
3. During the rework, my own destroy() called WEBGL_lose_context — strict-mode
   double-mount then reused the same dead context on the same canvas. Removed; guard
   added (`isContextLost()` → Canvas2D fallback).

### Event → fluid mapping as shipped
| Real event | Behavior |
| --- | --- |
| start | steel-blue prompt droplet at one of 5 inlets, drifting with the bench current |
| first_content | the droplet bursts where it stands — TTFT is the drift distance |
| progress | grey-white generation ink advected by a jet whose power tracks real tok/s |
| end ok | inks mix; field diffuses toward ambient |
| end interrupted | thread cuts with a counter-jet (curls back, visibly truncated) |
| end error | low-saturation red bloom, re-splatted ~2.6s so it refuses to mix |
| idle | zero injections; 0.988/frame dissipation settles to near-still drift |
| late join | a request started on another tab renders from the inlet onward — a real product need (send a chat, switch here to watch); deviation from the strict table, logged here |

One shared emitter: the lifecycle bus in lib/telemetryLog (ids minted at send time;
chat + workbench unified in their own commit). The sim has no other input; counts and
timings only. Copper/amber barred from the engine by smoke assertion.

### Implementation
Self-written WebGL curl-noise dye advection (~250 lines incl. GLSL; divergence-free by
construction, no pressure solve needed) + Canvas2D particle fallback sharing the
choreography. No dependency added. Route chunk 8.24 → **6.88 kB gz (net −1.36)**;
other routes unchanged; global budget untouched. DPR capped at 2; pauses on
document.hidden; reduced-motion (system or manual toggle) renders one static field
frame — measured 0 rAF requests. Hover on a log row draws that request's traced ink
thread on a 2D overlay — the art↔data link.

### Gate evidence
- 10/10 smokes green (observatory smoke's two idle-copy checks re-pointed to the new
  honest-idle copy, intent preserved; new Flow Bench assertions added to smoke:ui);
  gate libs empty diff; smoke:tiny green, 3B reloaded after.
- Truth check (executable no-synthetic-data): N real requests driven, sim ledger 'end'
  ids matched the rail's request log one-to-one (2/2, zero phantom events).
- Performance: 75fps measured at default settings (idle-settling field, M4, dpr 2);
  0 rAF in 1s after 20 navigate-away/return cycles (no loop/listener leaks).
- Tuning iterations recorded: additive splats first blew out to white (fixed:
  intensity-tinted injections + soft tonemap), then over-damped to murky smoke
  (fixed: luminance-driven alpha). Final: luminous ink on the dark bench.
- Evidence: idle/hover/reduced-motion stills, harness shots (both themes/widths,
  GPU-enabled capture), 40-frame sequence + flowbench-stream-recording.mp4 of a real
  streamed request (late-join thread → token flow → completion mix) in
  design-evidence/phase-6.1/. Every motion in it maps to a logged request.

---

## Phase 6.2 — Flow Bench visual remediation (2026-06-12)

### Before-critique (design-evidence/phase-6.2/before/, one real streamed request)
t=0: clean idle (0% lit — honest). TTFT: a single soft grey puff, 2.45% of pixels,
already the visual peak. Mid-stream: the puff has FADED to 1.58% while tokens are
still flowing — injection loses to dissipation. Completion: 0.8%, a ghost. Why weak:
(1) magnitude — dye dissipation 0.988/frame (~50%/s) outruns the 0.07–0.11-tinted
injections, and ambient drift ~0.07 canvas/s means ink dies where it spawns;
(2) contrast — mean ink RGB(99,102,104) on #0e1216 ≈ 2.9:1, under the 3:1 floor;
(3) NOT the wiring — instrumentation counted start=1, first_content=1, progress=100,
end=1 reaching the sim for one request; canvas is true-size (816×520@dpr1 headless)
and tokens resolve to real values at init. It reads as "is something supposed to be
happening?" — exactly the failure the phase names.

### Phase 6.2 outcome (after 6 documented iterations + one root-cause hunt)

**The real bug, found by elimination:** advection had been silently DEAD since 6.1.
`render()` enables blending for screen compositing; blending into a (half-)float sim
target without EXT_float_blend is INVALID_OPERATION and the driver silently drops the
draw — so every advect/splat after the first rendered frame no-opped. The "flow" was
JS-advanced splat positions; the ping-pong then flickered two stale dye generations
(48.8% of pixels flipping between consecutive frames at constant coverage — the
tell). Isolation tests passed because they never called render() between steps. Fix:
`gl.disable(BLEND)` at the head of every sim pass + canonical front/back swap +
FBO-completeness check with byte fallback + time-based (frame-rate-independent) decay
+ concentration-tracked alpha so dark pigment is visible on the light theme.

**Iterations** (frames + verdicts in iterations/iter1–5): 1 stronger params — ribbon
appears but cotton-soft; 2 sharper — ink exits right edge; 3 weaker bench current —
reference-quality swirl (vs a dead sim); 4 first tune against LIVE advection —
billowing fronts + filaments, core blown white; 5 displayGain 1.55 + 70% steel-blue
lean (operator request: visible color) + alphaGain 4.6 + dissipation 0.9992.

**Final parameter table** = FLOW_CONFIG in lib/observatory/flowBench.js (dev tuning
panel ships in dev builds only — verified absent from dist).

**smoke:flow (permanent gate)**: dark ttft-departure 3.9% (floor 1%), completion
24.0% (floor 15%), top-decile contrast 7.87:1 (floor 3:1), idle settle
35.3%→0.0%→0.000% frame delta after 60s; light 23.0% / 7.26:1. fps 79 at final
settings; 0 rAF leaks after 20 nav cycles; reduced-motion renders the frozen tuned
swirl. Thumbnail-test recordings (both themes) + hover-link proof shot in
design-evidence/phase-6.2/.

---

## Phase 8A — Original Camelid mark (2026-06-12)

**Inventory of the retired four-point sparkle** (all replaced, grep count now 0):
Avatar.jsx (assistant avatar), ChatWorkspace hero (52px) + pending row (30px),
MessageTurn avatar via Avatar, SidebarRail brand lockup (24px), SettingsView (28px),
TopologyCanvas empty state (22px), public/favicon.svg, plus CSS: pulseSparkle
keyframes (deleted), .camelid-sparkle-icon, hero class (renamed).

**Direction: the animal.** components/ui/CamelidMark.jsx — an upright-neck llama
glyph with the signature splayed ear pair, three strokes on a strict 24px grid,
stroke-only, currentColor (neutral ink default; copper never decorates). Legible at
16px (favicon ships it on the dark base in steel ink); wordmark lockup = mark +
Space Grotesk "Camelid" (sidebar/TopBar).

**Similarity check (one sentence each, per the brief):**
- Gemini: theirs is a four-point star; ours is an animal head with ears.
- ~Open~AI provider mark: theirs is an interlocking-loop knot; ours has no loops.
- Copilot: theirs is rounded twin chat-shapes; ours is a stroke-drawn quadruped head.
- Meta AI: theirs is a blue-gradient ring; ours is no ring and no gradient.
- Mistral: theirs is a pixel-block flag; ours has no blocks or flag geometry.
- Perplexity: theirs is a wireframe polyhedron; ours is organic-figurative, not
  geometric-abstract.
(No comparison needed more than a sentence; no redesign triggered.)

**Animation states** (CSS transform/opacity on SVG sub-elements only; reduced motion
renders all states static — state is also carried by existing text affordances):
| state | motion |
| --- | --- |
| idle | static |
| awaiting (post-send, pre-TTFT) | 2.2s breathing scale/opacity |
| streaming | ears flick alternately, advanced by each rAF-coalesced token batch (data-step parity) — rhythm = real cadence |
| error/abort | one 320ms settle back to rest |

Gate: zero old-mark instances (grep + smoke:ui assertions incl. favicon gradient
stops); 4-state screenshots both themes + streaming recording in
design-evidence/phase-8/.

## Phase 8B — chat fluidity (2026-06-12)

Baseline (PERF_BASELINE.md) showed the original architecture already healthy on the
test machine (rAF-coalesced flushes since Phase 0; 0 long tasks at 695 tokens) — so
this pass removed the structural risks rather than chasing numbers: block-memoized
markdown (stable prefix parses once via React.memo keyed on its string; boundary =
last block break outside an open fence), open-fence highlighting deferred to fence
close (decision: highlighting was the only per-flush O(block) cost left), footer
space reservation (zero layout shift), jump-to-latest affordance, contain:layout
style on turns, and the honest pacing buffer (lib/streamPacing.js — ≤150ms lag bound
+ instant byte-identical drain, enforced by a behavioral smoke; metrics keep real
arrival times per I4; first easing curve failed its own lag-bound smoke and was
steepened to 60%/step). Full table in PERF_AFTER.md; before/after recordings of the
identical greedy prompt.

---

## Phase 9 — Response-length control (2026-06-12)

**Step 0 data contract (probed before any UI):** model context = `/v1/models`
`meta.n_ctx_train` (now merged onto model records — descriptive, I2-disclaimed in the
marker label); verified bound = max VALIDATED `bounded_context_*_pack` window on the
exact matched row (3B: 2,048; unvalidated windows never count; the exact-artifact gate
applies — a fixture without the GGUF basename gets no bound, which the smoke proves);
KV-cost and system memory DO NOT EXIST on the API → BACKEND_ASKS.md #3 filed with
exact field names/units, and the memory indicators render ABSENT with an explanatory
line — no client-side estimation, no fake gauge.

**Boundary behavior (probed live):** the backend does NOT clamp — any request where
`prompt_tokens + max_tokens > context_length` returns typed
`context_length_exceeded` (verified with the 64-token fixture at max_tokens 1M and at
50). The UI mirrors that truthfully: red states block send with the typed error's
language.

**The control:** log-scale slider (position↔tokens round-trips, smoke-checked) with
detents at 256/1k/4k/16k/64k/256k/1M and light snap; paired numeric field (1→1M,
arrows step, Shift = ×10); markers from real data only (verified bound gets the one
allowed Evidence-Chip treatment; model max labeled "from model metadata, not a
support claim"). Per-model persistence (`camelid.maxTokens.<id>`, legacy global key
as fallback); stored values are clamped visually, never rewritten.

**Validation:** red = will-not-fail-silently errors (over model context), amber =
allowed-but-untested (over verified bound), each with icon + product-voice message;
composer-level send-time check uses the backend's real rule with the client prompt
estimate (labeled estimated) and disables send with the inline red strip.

**Conscious smoke port:** the Phase-2-era "no max-token strings in ChatWorkspace"
regex predates send-time validation; narrowed to its intent (no PICKER in chat — the
control lives in Settings).

**Matrix (design-evidence/phase-9/, both themes):** normal 1,024 ok · 8,192 amber
(beyond verified 2,048) · 200,000 red (over 131,072) · missing-data fallback (absent
lines, captured offline — while a model is loaded the gate's selection snap-back
makes metadata-less selection unreachable, which is itself correct behavior) ·
composer red strip with real numbers (131,072 limit + estimated prompt) and send
disabled. Gate: 11/11 smokes green, gate libs empty diff, main chunk 104.68 kB gz.

---

# Backend addendum — Agent security hardening (2026-06-16)

These entries cover backend (Rust) agent-loop hardening, logged here because the
build spec routed the design decisions and threat model to DESIGN_LOG. Code lives
in `src/chat/{tools,agent,shell_sandbox,audit}.rs`. Companion docs: `AUDIT_EVENTS.md`
(event schema), `DECISIONS.md` (D9, agent loop).

## run_shell OS sandbox (`shell_sandbox`)

### Decision

`run_shell` is the only tool that hands the model a general-purpose execution
primitive. The file tools enforce a canonical-root jail *in code*; a shell
command cannot be jailed that way, so it gets a *kernel*-enforced sandbox.

Three modes, config `--shell-sandbox`:

- **`disabled`** — `run_shell` is not registered with the model at all
  (`tools::specs` omits it). Defense in depth: `run_shell` execution also refuses
  if reached.
- **`sandboxed`** — **the default.** On Linux (x86_64/aarch64) the command runs
  with, in child `pre_exec` order: chroot into the workspace *iff* it is a usable
  rootfs (else cwd-confinement, surfaced as a caveat); rlimits (no core, 30 CPU-s,
  1 GiB AS, 64 fds); supplementary-group drop + setgid + setuid to nobody (65534)
  when started as root, with a re-`setuid(0)` check that the drop stuck;
  `PR_SET_NO_NEW_PRIVS`; then a seccomp-BPF filter that **EPERMs** the
  `ptrace`/`mount`/`socket` families (+ `unshare`/`setns`/module/kexec/reboot/swap)
  and **kills** on arch mismatch (defeats x32/compat bypass). The existing
  parent-side wall-clock kill-on-deadline loop is retained.
- **`unrestricted`** — explicit opt-in; cwd-pinned + timed only. Logs a startup
  warning.

**Fail closed.** seccomp availability is preflighted (`prctl(PR_GET_SECCOMP)`);
if seccomp is unavailable, or the host is not Linux/x86_64-aarch64, sandboxed mode
**refuses to run `run_shell`** rather than silently downgrading to unrestricted.
The startup banner and the returned tool error report the *actual* enforced layers
(`EnforcedShell::summary`) — the UI never claims a sandbox the kernel didn't apply.

### Threat model

| Threat | Vector | Mitigation |
| --- | --- | --- |
| Prompt-injected destructive command | tool result / file content steers the model to emit a `run_shell` call | Approval tier (`confirm`/`deny`) gates the call before execution; injection tests in `agent.rs` assert result text never auto-executes. |
| Workspace escape via shell | `cd /`, absolute paths, `..` | chroot (rootfs) or chdir-confined cwd; the file tools' canonical-root check is unaffected. |
| Local privilege escalation | exploit a setuid binary, load a kernel module, `ptrace` another process | uid/gid drop to nobody + `NO_NEW_PRIVS`; seccomp EPERMs `ptrace`, module load, `mount`. |
| Network exfiltration / C2 from the shell | open a socket from the command | seccomp blocks `socket`/`socketpair` (the validating test opens a raw socket and asserts EPERM). Note: this is independent of `--allow-net`, which only governs the `http_fetch` tool. |
| Resource exhaustion (fork bomb, fill disk, hang) | runaway command | rlimits (CPU-s, AS, fds) + parent wall-clock timeout + no core dumps. |
| Silent loss of protection | seccomp/uid-drop unavailable on host | fail closed: refuse to run; never downgrade silently. |

### Residual risks / follow-ups

- **chroot needs a provisioned rootfs.** Chrooting into a bare scratch dir leaves
  no shell to exec, so chroot engages only when the workspace has `/bin/sh`;
  otherwise cwd-confinement is used and the caveat is surfaced. Production-preferred
  alternative (mount namespace + bind-mount of a minimal rootfs) is a documented
  follow-up.
- **Requires root to drop privileges and chroot.** When started unprivileged the
  uid-drop/chroot layers can't engage; seccomp + rlimits + cwd-confinement still do,
  and the enforced-layer report says so honestly.
- **Verification status.** The Linux enforcement is built behind
  `cfg(target_os="linux", arch ∈ {x86_64,aarch64})` and **type-checks on the Linux
  target** (verified via `cargo check --target x86_64-unknown-linux-gnu` on the
  Windows dev box), but its *runtime* behavior — that seccomp actually EPERMs
  `socket()` — is exercised only by the `socket_is_blocked_under_seccomp` test,
  which must be run by Linux CI. Until that runs green, sandboxed mode is trusted
  on Linux only after CI; everywhere else it correctly fails closed. The Windows
  dev box exercises the fail-closed paths (`*_fails_closed_off_linux`).

## Related: approval tiers + auto-approve (Task 2) and audit events (Task 3)

- **Approval tiers** (`tools::ApprovalPolicy`): every tool resolves to an
  `auto`/`confirm`/`deny` tier through one chokepoint the loop consults before
  executing. `--auto-approve` promotes `confirm`→`auto` but **never** exec-risk
  tools (`run_shell` stays gated) and is **refused (fail closed) under
  `CAMELID_PRODUCTION`**.
- **Audit events** (`audit::AuditSink`): each executed tool emits
  `agent.tool_call`/`agent.tool_result` carrying a SHA-256 *digest* of the args
  (never raw args — they can hold secrets), the applied tier, outcome and duration.
  Pluggable sink (no-op default / non-blocking webhook that drops on backpressure).
  Schema in `AUDIT_EVENTS.md`.

## VRAM headroom policy + contention (Task 4)

- **Headroom policy** (`cuda_vram::evaluate`, pure + unit-tested): at resident
  load the projected device allocation (resident weights + scratch + sized KV) is
  checked against free VRAM and a configurable floor
  (`CAMELID_MIN_VRAM_HEADROOM_MIB`, default 512 MiB) **before** allocating. A
  violation **refuses the resident load with a named shortfall** (MiB) and falls
  back to CPU — no mid-load OOM. Wired into `src/inference.rs`'s resident sizing.
- **Contention harness** (`cuda_vram::measure_contention`, `cfg(cuda)`): occupies
  a model-sized allocation, attempts a second on the same device ×5, records
  clean-fail vs OOM with median + variance. Findings + schema:
  `qa/cuda/CONTENTION_FINDINGS.md` (numbers pending on the CUDA host).

## CPU-vs-CUDA parity (Task 5)

### Tolerance gate (rationale)

The shipped CUDA Q8_0 decode kernel mirrors the CPU reference op-for-op and is
compiled with `--fmad=false`, so the f32 logits are **bit-identical** and the
greedy argmax / token IDs match exactly (see `src/cuda.rs` header). The default
gate (`ToleranceGate::bit_exact`) therefore allows **zero token divergences** and
**zero logit delta**.

Why a tolerance at all, then? Floating-point matmul *reduction order* is not
associative, so any path that does not preserve the CPU order (a future
cuBLAS-backed matmul; the offload streaming path if it ever reorders reductions)
is not expected to be bit-exact. For those, `ToleranceGate::argmax_stable(tol)`
defines parity as: **greedy token IDs must still match** (argmax is robust to tiny
logit noise) with a bounded max absolute logit delta. The gate records which
regime applied, so a divergence is an explained, documented outcome — not a bare
pass/fail. This treats divergence with a tolerance rationale rather than as a
binary, per the spec.

### Artifact + harness

- Schema `camelid.cpu_cuda_parity/v1` (`ParityArtifact::to_json`): model, fixture,
  verdict, tolerance regime, **first-divergence index**, tokens compared,
  divergence count, length-mismatch flag, and both token streams.
- `compare_tokens` reports the first-divergence index token-by-token (pure,
  unit-tested — 6/6 on the Windows dev box).
- `tests/cuda_cpu_parity.rs` (ignored) gates a CPU stream against a CUDA stream
  generated over the frozen fixtures and writes `qa/cuda/parity-latest.json`,
  reusing the repo's diag-file shape.

### Status

Implemented + unit-tested on the Windows dev box; type-checks under
`--features cuda`. The **token-by-token run over the frozen fixtures on real CPU
and CUDA backends is pending the CUDA host** — until then the GPU correctness
story should state parity is *gated by a ready harness with an explicit
bit-exact tolerance*, not that the run has been performed. The dev box has no GPU
build path exercised here; the existing `qa/parity_*_diag.json` files are
camelid-vs-llama.cpp and are not a CPU-vs-CUDA substitute.

## Phase 7 (runnable lane) — Receipt lane distinction: runnable vs supported (2026-06-16)

Part of the runnable-lane build (`RUNNABLE_LANE_SPEC.md`). The runnable lane runs any
covered GGUF deterministically as a generic f32 graph and is externally anchored to HF
transformers (it is the promotion oracle for the supported lane). Its receipts must be
**unmistakable** from a supported, llama.cpp-parity-verified receipt — and must never
earn copper.

### Schema (backend)

- New `ExecutionLane { Runnable, Supported }` enum and an additive
  `ParityReceipt.execution_lane: Option<ExecutionLane>` (`src/receipt/mod.rs`).
- Additive on purpose, following the `execution_trace`/`signature` precedent:
  `#[serde(default, skip_serializing_if = "Option::is_none")]`. **Absent = supported**
  (the legacy default), so every receipt written before this field digests and verifies
  byte-for-byte unchanged (`absent_execution_lane_is_omitted_and_keeps_digest_stable`).
- `Some(Runnable)` is part of the canonical body, so it is bound into `receipt_id` and
  cannot be stripped to pass a runnable run off as supported without changing the id
  (`runnable_lane_serializes_and_is_digest_bound`). The supported serving path
  (`src/api/mod.rs`) leaves the lane absent.

### UI (frontend)

- New first-class evidence state **`runnable`** (`src/lib/evidenceStatus.js`): amber
  (the 🟡 legend state), classified from `runnable`/`runnable_*`, with its own label and
  claim copy. It can never fall through to the copper `supported` state.
- `.ev-chip--runnable` (`src/styles/evidence.css`) uses the amber `--color-evidence`
  tokens — the copper `--color-verified` token is never spent on it.
- `ParityReceiptCard` (`src/components/chat/render/ParityReceipt.jsx`) reads
  `receipt.execution_lane`; a runnable receipt gets a distinct "Runnable lane" badge,
  a "Runnable receipt" title, and runnable-specific copy ("cross-checked execution, not
  a supported parity contract — never copper"). Badge styled amber in `chat.css`.

### Status doctrine

Copper stays reserved exclusively for `supported`/`supported_*`. Runnable is amber,
distinct from both supported (copper) and unsupported (slate). A runnable receipt
attests *deterministic, oracle-anchored execution*, not cross-validated support.

### Gate results

- Backend: `cargo test --lib receipt::` — 29/29 (3 new: digest-stable-when-absent,
  digest-bound-when-runnable, runnable≠supported digests).
- Frontend: `node scripts/ui-regression-smoke.mjs` passes, with new assertions that
  runnable classifies to its own state (never copper), the runnable chip uses amber not
  `--color-verified`, and the receipt card detects + labels the runnable lane.

## Runnable lane — Smoke-admission vs. oracle qualification (2026-06-16)

Two distinct, deliberately-separate notions of "this model is OK" in the runnable lane.
They must never be conflated, and the receipts/UI keep them apart.

### Oracle qualification (Phase 5, per architecture)

- A **one-time, per-(architecture, quant, tokenizer) gate**. Proves the runnable f32
  graph is *numerically equivalent to HF transformers* — greedy token sequences match
  exactly on frozen fixtures, logit max-abs-diff ~1e-4 (`tests/runnable_parity.rs`,
  artifacts `qa/runnable/<arch>-parity.json`).
- This is what *earns* an architecture the right to be trusted. Currently qualified:
  llama, qwen3, gemma3 (Q8_0). phi3 is implemented + coherence-validated; its HF
  bit-parity is pending a larger-RAM machine (Phi-3-mini is 3.8B → ~15 GB f32).
- It is the promotion oracle for the supported lane.

### Smoke-admission (new capability, per model file)

- A **per-GGUF** check: "does THIS file admit, load, and execute cleanly here." It does
  NOT prove correctness — it attests *deterministic execution*. (`src/runnable/smoke.rs`)
- Guardrail: smoke-admission runs **only on combos that are already oracle-qualified**.
  A GGUF on any other combo is refused with `combo not yet anchored` — so a green smoke
  can never be mistaken for "this architecture is correct" (that claim belongs to the
  oracle, not the smoke).
- Checks, in order: (1) covered-set admission gate; (2) load — all tensors present,
  shapes consistent, dequant succeeds on every tensor; (3) greedy forward on a fixed
  tiny prompt with finite logits and a sane range; (4) coherence — greedy-decode ~24
  tokens, fail if degenerate (a short-period repetition loop or too-few distinct
  tokens). The smoke prompt is rendered through the GGUF's own chat template so
  instruction-tuned models get a fair coherence test.
- A pass emits a **runnable receipt** (`execution_lane = Runnable`, never copper) whose
  `parity` block is `not_compared` — honest that no reference was consulted. It is, by
  construction, unmistakable from a supported parity receipt (Phase 7).

### The distinction in one line

Oracle qualification answers "is this architecture's math right?" (vs HF, once).
Smoke-admission answers "does this specific file run cleanly?" (per file, never a
correctness claim). The runnable receipt a smoke emits attests the latter only.

## Models tab — lane-distinction UI (Gate 4, 2026-06-17)

The Models tab renders four DERIVED local sections (Supported / Compatible / Run
smoke-admission / Not yet runnable). Membership is computed from `/api/models/local`
lane facts + the `/api/capabilities` support contract — never a hand-authored array.

**Use/load is lane-scoped, on purpose.** Only **Supported** rows get a "Use for chat"
button. It calls `POST /api/models/load { id, path }` — the parity-locked supported chat
backend — and the row that matches `/api/models/current` shows "● Loaded" with a neutral
(never copper) active accent. Copper stays exclusively in the EvidenceChip.

**Compatible (runnable) rows deliberately have no chat button.** The runnable lane is the
generic f32 engine and exposes no HTTP serve/generate route (only `runnable-smoke` + the
CLI). Loading a runnable-only model through `/api/models/load` would run it on the
*supported* backend and visually imply parity it doesn't have — so we don't. Those rows
keep their runnable receipt (amber, `parity: not_compared`) plus an explicit "CLI only —
no HTTP serve yet" note. The missing endpoint is logged as BACKEND_ASKS #4, not faked.

**Model copy says what the model is GOOD AT, never where it runs.** Catalog blurbs and the
local `describeModel()` line were scrubbed of all hardware/parity/serve-flag jargon
("16 GB Mac", "greedy parity", "GPU-resident", "CAMELID_GEMMA4_SERVE", "memory-infeasible").
They now state strengths by family ("Strong at reasoning, coding…", "multilingual chat,
reasoning, coding, and math"). System/lane facts live in the chip + meta line, not the
description.
