# Frontend Baseline — Phase 0 Recon

Recorded: 2026-06-12
Source head: `16aa661f7e44` — "Merge PR #253: Inference Observatory — live telemetry stream + real-time visualization tab" (2026-06-11)
Toolchain: Node v22.22.2, npm 10.9.7, Vite 8.0.10, React 19.2, no router library, no other runtime deps.

This file is the Phase 0 deliverable of the frontend overhaul spec. Every later phase gate
re-verifies the readiness-gate logic recorded in §2 against this document. No code was
changed in Phase 0.

---

## 1. View / route structure

Hash-based navigation, no router library. `App.jsx` holds the active tab in `useState`;
`HASH_TABS` (`src/App.jsx:24`) whitelists deep-linkable views. The hash is read once on
mount (`src/App.jsx:72-77`) and written on tab change; the active tab also persists to
`localStorage.camelid.activeTab` for browser-tab restore. Mobile nav is a top-bar
hamburger + scrim below an 860px breakpoint.

| Hash / tab ID | View component | Purpose |
| --- | --- | --- |
| `chat` (default) | `views/ChatWorkspace.jsx` | Chat surface, streaming, composer, gate messaging |
| `library` | `views/ModelsView.jsx` | Model catalog, local GGUF registration, load/activate |
| `api` | `views/ApiView.jsx` | `/api/capabilities` contract, readiness lanes, curl examples |
| `analytics` | `views/AnalyticsView.jsx` | Usage telemetry (explicitly usage-only, not evidence) |
| `history` | `views/HistoryView.jsx` | Conversation list, search/rename/delete |
| `memory` | `views/MemoryView.jsx` | Local memory snippets |
| `system` | `views/SystemView.jsx` | Runtime health, support lanes, Q8 policy |
| `settings` | `views/SettingsView.jsx` | API base override, backend launcher, theme, max-tokens |
| `cluster` | `views/ClusterView.jsx` | Topology editor (client-side model, dev-hook probes) |
| `observatory` | `views/InferenceObservatoryView.jsx` | Live inference telemetry (SSE) visualization |

Desktop nav: `components/layout/SidebarRail.jsx` (collapsible) + `components/TopBar.jsx`
(carries runtime + support-gate status outside Chat/Models — required by frontend/README.md).
The API tab is first-class in desktop/sidebar/mobile nav (contract requirement).

## 2. Readiness-gate logic (assert unchanged at every later gate)

**Entry point:** `getChatGateState(capabilities, model, runtime)` — `src/lib/chatGate.js:4-24`.

`chatUnlocked` is true iff **runtimeReady && contractSupported**, where:

- `runtimeLoaded` = `runtime.loaded_now` && `modelRuntimeIdMatches(model, runtime)`
  (`src/lib/modelState.js:27-31` — `runtime.active_model_id` must equal `model.id` or
  `model.runtime_model_name`).
- `runtimeGenerationReady` = `runtime.generation_ready` && same id match.
- `runtimeReady` = `isRunnableInCurrentRuntime(model, runtime)` && `runtimeLoaded` &&
  `runtimeGenerationReady` (`src/lib/modelState.js:38-42`; for local models requires
  `model.status === 'ready'`, a local path, and the model itself reporting
  `loaded_now`/`generation_ready`).
- `contractSupported` = `isCompatibilitySupportedForModel(capabilities, model)`
  (`src/lib/capabilities.js:711-717`): the hint from `findCompatibilityHint`
  (`src/lib/capabilities.js:591-667`) must be `kind === 'compatibility'` with
  `exact === true`, AND the matched row's `status` must satisfy
  `isSupportedCapabilityStatus` (`supported` or `supported_*`,
  `src/lib/capabilities.js:212-215`).

Properties of `findCompatibilityHint` that later phases must preserve:

- Exact row-id identity match first (`findExactCompatibilityRowByIdentity`), then
  Llama-BPE exact size/version/instruct detection against the
  `EXACT_LLAMA_PROMOTION_ROWS` whitelist (`capabilities.js:87-91`), then named
  future-exact-row matchers (Mistral/Mixtral/Qwen/Gemma), then family fallbacks.
- Family fallbacks return `kind: 'family'` hints which are **never** `exact`, so they can
  never unlock chat — resemblance is not evidence.
- Quant awareness: `quant_missing` / `quant_mismatch` hints never unlock; quant labels come
  from `general.file_type` normalization (`quantLabelFromGgufFileType`,
  `capabilities.js:43-47`).
- Exact-artifact gate: the three tracked Llama rows additionally require the exact GGUF
  basename (`EXACT_ARTIFACT_GATED_ROWS`, `capabilities.js:93-97`) or the hint degrades to
  `artifact_mismatch` (not unlocked).
- Readiness classification for row display lives in `src/lib/capabilityReadiness.js`
  (`classifyCapabilityRow` — status/scope fields only, never family name;
  `classifyInputModality` — anything non-text is fail-closed unsupported-multimodal;
  `isExactRowSupported` — exact `row.id` equality only).

Backend surfaces consumed by the gate: `GET /v1/health` (`loaded_now`,
`generation_ready`, `active_model_id`), `GET /api/capabilities`
(`model_compatibility` rows, `support_contract`, `api_features`). At baseline the live
contract reports **14 compatibility rows, 3 supported model families, 5 planned families,
9 guarded API features**.

## 3. Component inventory

### UI primitives (`src/components/ui/`)
- `Button.jsx` — variants primary/tonal/ghost/outline, sizes sm/md/lg
- `IconButton.jsx` — icon-only button
- `Chip.jsx` — status pill; tones neutral/accent/ready/warn/error/info
- `Card.jsx` — container card
- `StatusDot.jsx` — state indicator; tones ready/warn/error/info/neutral/offline; pulse
- `Avatar.jsx`, `Field.jsx`, `Modal.jsx`, `ConfirmDialog.jsx`, `Notice.jsx`,
  `Tooltip.jsx`, `ThemeToggle.jsx`, `EmptyState.jsx`, `icons.jsx` (20+ inline SVG icons)

### Layout (`src/components/layout/` + top-level)
- `SidebarRail.jsx` — conversation list, search, new chat, theme toggle, collapse
- `ConversationListItem.jsx` — sidebar conversation card with rename/delete
- `BackendBanner.jsx` — offline banner with dev-hook quick-start
- `TopBar.jsx` — tab nav, conversation title, model selector, gate status
- `AppSidebar.jsx`, `GlobalNotice.jsx`, `ConversationDeleteDialog.jsx`

### Chat (`src/components/chat/`)
- `MessageTurn.jsx` — message bubble; assistant messages render via `lib/markdown.jsx`
- `render/StreamingIndicator.jsx`, `render/ParityReceipt.jsx` (receipt verification),
  `render/Diagnostics.jsx`

### Models (`src/components/models/`)
- `SupportedModels.jsx` — exact-row compatibility grid with readiness lanes
  (template / context / throughput)

### Cluster (`src/components/cluster/`)
- `TopologyCanvas.jsx` (custom SVG canvas, drag), `NodeCard.jsx`, `NodeInspector.jsx`,
  `NodeInventory.jsx`, `AddServerWizard.jsx`, `DiscoverDevices.jsx`, `ClusterDrawer.jsx`

### Observatory (`src/components/observatory/`)
- `InferenceCanvas.jsx` (token particles / layer heatmap / KV trail),
  `MetricsOverlay.jsx`, `ProofOverlay.jsx`, `DetailsPanel.jsx`

## 4. Data hooks (`src/hooks/`)

- `useDashboardData.js` (1348 lines) — the monolithic data layer: fetches `/v1/health`,
  `/v1/models`, `/api/capabilities`, `/api/models/current`, catalog downloads; polls every
  2.5s; owns conversations/memories/models state + `sendMessage`/`activateModel`; ~50
  returned properties, prop-drilled from `App.jsx` (no context providers anywhere).
- `useTheme.js` — system/light/dark preference in `localStorage.camelid-theme`; applies
  `[data-theme]` attribute; removes it for system mode so `prefers-color-scheme` wins.
- `useBackendLauncher.js` — dev-server-only backend launch/stop via `/__camelid/backend/*`
  vite hook; polls status at 2.5s.
- `useClusterTopology.js` — client-side topology model persisted to
  `localStorage.camelid.clusterTopology`; merges `public/cluster-import.json` once;
  real probe/telemetry calls via dev hook.
- `useInferenceTelemetry.js` — `InferenceTelemetryStore` subscribed to
  `/api/telemetry/stream` SSE; 250ms live metric sampling.
- `useNotice.js` — toast state.

## 5. State management

All React state lives in `App.jsx` + `useDashboardData`; persistence is localStorage only
(no IndexedDB). Keys in use:

`camelid.sidebarCollapsed`, `camelid-theme`, `camelid.activeTab`,
`camelid.selectedConversationId`, `camelid.selectedModelId`, `camelid.apiBase`,
`camelid.launchCommand`, `camelid.clusterTopology`, `camelid.clusterImports`,
`camelid.localModels`, `camelid.conversations`, `camelid.memories`, `camelid.maxTokens`.

API base resolution: `VITE_CAMELID_API_BASE` build-time default →
`localStorage.camelid.apiBase` runtime override via SettingsView (I5 contract). Default
`http://127.0.0.1:8181`. Local model paths are saved in `camelid.localModels` only
(local-only surface).

## 6. Lib modules (`src/lib/`)

Gate-participating (see §2): `chatGate.js`, `capabilities.js`, `capabilityReadiness.js`,
`modelState.js`.

Chat/data: `chatCompletionStream.js` (SSE parser — handles partial frames, `usage`
chunks, structured `event: error`, non-streaming JSON fallback), `chatState.js`
(`NEW_CHAT_SENTINEL`), `conversationStorage.js` (normalization/migration).

Presentation: `formatters.js`, `loadedModelDisplay.js`, `markdown.jsx` (custom renderer:
code blocks, bold, inline code — no tables/sanitized-HTML pipeline yet),
`supportedModels.js` (installable catalog metadata), `acceptanceTargets.js`.

Cluster: `clusterModel.js` (typed enums + CRUD + validation + auto-layout),
`devCluster.js`, `devBackend.js` (dev-hook fetchers).

Telemetry: `inferenceTelemetry.js` (store/reducer for `camelid.telemetry/v1` SSE) +
`lib/observatory/*` (5 renderer modules: tokenParticles, layerVisualizer, kvCacheTrail,
samplerBloom, clusterConstellation).

## 7. Styles

`src/styles.css` imports, in order: `styles/tokens.css` (CSS custom properties — colors
incl. light/dark via `[data-theme]` + `prefers-color-scheme`, type scale, spacing, radii,
shadows, transitions), `base.css`, `ui.css`, `shell.css`, `chat.css`, `views.css`,
`cluster.css`, `observatory.css`. A tokens file already exists; Phase 1 evolves it rather
than introducing one. Note: repo memory records that legacy `components.css` references
were preserved for dashboard views during the redesign — verify before removing any
legacy CSS.

Fonts at baseline: system font stack only — no font files vendored, no CDN calls
(already offline-correct).

## 8. Smoke scripts (`frontend/scripts/`)

| Script (`npm run …`) | Asserts | Needs backend |
| --- | --- | --- |
| `smoke` → `smoke.mjs` | Frontend reachable, model load, health/models, capabilities gate, chat guard; `--load-tiny` = `smoke:tiny` | Yes (+ dev server) |
| `smoke:streaming` | SSE parsing: partial chunks, usage, JSON fallback, `event: error` after headers | No |
| `smoke:model-state` | `modelState.js` runnable/loaded/label logic | No |
| `smoke:capability-readiness` | Row classification; family prefix NEVER supported; modality fail-closed | No |
| `smoke:3b-closure` | 1B/3B exact-row promotion + artifact gating UI logic | No |
| `smoke:integration` | Full mocked chat flow incl. streaming state updates | No |
| `smoke:observatory` | Telemetry store renders only from real events; no fabricated telemetry | No |
| `smoke:ui` | UI class-structure regression + README copy assertions | No |

## 9. Baseline results (2026-06-12, head `16aa661f7e44`)

**Build:** `npm run build` clean in 1.78s, 87 modules, single chunk:

| Artifact | Size | gzip |
| --- | --- | --- |
| `dist/assets/index-*.js` | 507.78 kB | **143.76 kB** |
| `dist/assets/index-*.css` | 115.37 kB | **17.20 kB** |
| `dist/index.html` | 0.81 kB | 0.49 kB |

Phase 7 bundle budget (≤1.6× baseline JS gz): **229.9 kB gzipped JS**. No code splitting
at baseline (Vite warns about the >500 kB chunk).

**Smokes:**

| Smoke | Result |
| --- | --- |
| `smoke:streaming` | PASS |
| `smoke:model-state` | PASS |
| `smoke:capability-readiness` | PASS |
| `smoke:observatory` | PASS |
| `smoke:3b-closure` | PASS |
| `smoke:integration` | PASS |
| `smoke` (backend up, tiny fixture loaded) | PASS — chat correctly blocked |
| `smoke:tiny` | PASS — fixture loads, `generation_ready=true`, `contract_supported=false`, chat gate stays blocked |
| `smoke:ui` | **PRE-EXISTING FAIL** — asserts repo `README.md` matches `/product-forward while still reflecting the local-first runtime contract/i`; that phrasing was replaced by the perf-section README rewrite. Stale assertion, known follow-up predating this overhaul; not part of the I6 gate set (build / smoke:streaming / smoke:tiny). |

**Live contract at capture:** 14 `model_compatibility` rows; supported gate copy matches
COMPATIBILITY.md (TinyLlama current gate; Llama 3.2 1B bounded 512–8192 packs; 3B
`supported_exact_row_smoke`; 8B bounded 512/1024/2048; Mistral 7B v0.3
`supported_exact_row_smoke`; Mixtral one-token evidence only, fail-closed).

## 10. Screenshots

`design-evidence/phase-0/` — every view at desktop 1440×900 and mobile 390×844, headless
Chrome, default (light/system) theme. Backend state during capture: camelid running on
127.0.0.1:8181 with the unsupported `tiny-generation` smoke fixture loaded — i.e. the
honest "backend online, generation-ready, chat blocked by contract" state.

Files: `{chat,library,api,analytics,history,memory,system,settings,cluster,observatory}-{desktop-1440,mobile-390}.png`

Note: the observatory pair was captured in a second pass the same day (same head, same
backend state — tiny fixture loaded, chat gate blocked, observatory in its honest
"waiting for live telemetry" empty state). Baseline finding for the Phase 7 responsive
audit: at 390px the observatory Run Details panel overflows the viewport horizontally.

---

## Errata (appended in Phase 2 pre-work — trust BASELINE.md only together with this section)

1. **§7 fonts claim was false.** At baseline, `tokens.css:11` imported Plus Jakarta
   Sans/Outfit from fonts.googleapis.com at runtime — the baseline did NOT render
   fully offline. Phase 1 replaced the CDN import with self-hosted Fontsource files;
   offline rendering became true (and testable) only from Phase 1 onward.
2. **§9 understated how broken `smoke:ui` was.** It recorded a single stale README
   assertion, but every assertion after that first failure was dead code: the script
   read the already-deleted `src/styles/components.css` (readFileSync would throw)
   and asserted pre-redesign TopBar internals (`exactHintDetail`) that no longer
   exist in `components/TopBar.jsx`. The smoke was re-baselined in Phase 2 pre-work
   and only then re-joined the standing gate set.
