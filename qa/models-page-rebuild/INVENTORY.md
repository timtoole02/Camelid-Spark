# Models page rebuild — Phase 0 inventory

Baseline: `before-1280.png` (full-page, 1280px, dev build against live `camelid serve`
with Llama-3.2-3B-Instruct-Q8_0 loaded + generation-ready). DOM dump of
`.models-view` children confirmed 13 top-level blocks.

## Panel inventory (render order) and fate

| # | Panel | Component | Data source | Fate |
|---|---|---|---|---|
| 1 | Hero header "Exact rows, real readiness…" | inline `cxv-head` | static copy | REPLACE — short page header, no contract prose |
| 2 | `models-hero-ledger` support-contract summary | inline | `/api/capabilities` | DELETE |
| 3 | Local models by lane | `LocalLaneSections.jsx` | `/api/models/local` + capabilities | FOLD — becomes Zones 2–3; `laneOf`/rows reused |
| 4 | "Supported models" download cards | `SupportedModels.jsx` + `lib/supportedModels.js` | hand-authored `SUPPORTED_MODELS` + localStorage records + live scan | DELETE section — curated data folds into Zone 5 as decoration only |
| 5 | Catalog — acquire from HuggingFace | `CatalogLaneBrowse.jsx` | `/api/models/catalog` (+downloads poll) | REWORK — becomes Zone 5; per-row polling moves to global Zone 4 |
| 6 | Search toolbar + truth strip + summary pills | inline (FILTERS) | localStorage-merged `models` + runtime | DELETE |
| 7 | `models-status-grid` (Loaded now / Next chat) | inline | runtime + `models` | DELETE — Unload + readiness collapse into Zone 1 |
| 8 | Llama 3.2 3B acceptance panel | inline + `lib/acceptanceTargets` | hand-authored target | DELETE panel (lib file stays — consumed by `lib/loadedModelDisplay.js`) |
| 9 | Tracked exact Q8 compatibility rows | inline | capabilities | DELETE — ledger content lives in the Compatibility view |
| 10 | "Local runtime" model grid | inline (legacy FILTERS grid) | localStorage-merged `models` | DELETE |
| 11 | Tokenizer playground | `TokenizerPlayground.jsx` | `/api/tokenize` | RELOCATE — Diagnostics disclosure |
| 12 | `models-catalog-panel-clean` catalog preview | inline | `/api/models/catalog` | DELETE |
| 13 | `models-setup-grid`: Import local GGUF + Hosted API fieldset | inline | `registerModel` / stub | **FLAGGED — not listed in the conductor doc.** Disposition: import-by-path moves into the Diagnostics disclosure (capability preserved — it is the only way to load a GGUF outside `models/`); the permanently-disabled Hosted API fieldset is deleted (dead UI; `connectExternalModel` is a stub notice) |
| 14 | "API links" section (`apiLinkModels`) | inline (legacy grid family) | localStorage external records | DELETE (hosted routing not wired) |
| 15 | "Still needs setup" section (`setupModels`) | inline (legacy grid family) | localStorage-merged `models` | DELETE |
| 16 | Model inspector (modal) | `ModelInspector.jsx` | `/api/models/inspect` | RELOCATE — Diagnostics disclosure |

## External consumers of deletion-slated code

- `lib/acceptanceTargets.js` — imported by `lib/loadedModelDisplay.js`
  (`resolveLoadedModelDisplayName`); the lib stays, only the ModelsView panel goes.
- `SUPPORTED_MODELS` — only `SupportedModels.jsx` imports it; folds into Zone 5.
- `TokenizerPlayground` / `ModelInspector` — only `ModelsView.jsx`; relocated.
- `UnsupportedBlocker` — `LocalLaneSections.jsx` + `ModelInspector.jsx`; kept.
- `useDashboardData` props consumed by ModelsView today: runtime, capabilities,
  refreshDashboard, registerForm/setRegisterForm, externalForm/setExternalForm,
  registerModel, connectExternalModel, models, selectedModelId/setSelectedModelId,
  loadingModelId, activateModel, unloadCurrentModel, installModel,
  installCatalogModel, cancelModelDownload, apiBase, setTab. The rebuilt page
  drops most of these (App.jsx call site updated in Phase 4); chat-side users of
  the same hook (ChatWorkspace, CommandPalette, TopBar) are untouched.

## Backend facts captured for wiring (no backend changes)

- `GET /api/models/catalog/downloads` items: `{ id (=catalog_id), repo_id,
  filename, total_bytes, bytes_downloaded, status }`; terminal
  `completed`/`failed` entries are returned once and then dropped from the map —
  the poller must treat "entry disappeared" as "check the live local scan".
- `POST /api/models/catalog/cancel` body is `{ id }` (catalog_id).
- Download lands as `models/<filename>.part`, promoted on curl success only.
