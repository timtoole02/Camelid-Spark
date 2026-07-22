# Neural Field — performance receipt (Phase 5)

Date: 2026-07-02. Dev machine: Windows 11, i7-class 16 logical cores, RTX 3060 Laptop 6GB.
Renderer: Canvas2D, zero new dependencies. Measurement: headless Chrome (puppeteer-core,
`--enable-gpu`), 1600×900 CSS px viewport, during a **real 50-token TinyLlama 1.1B Q8_0
generation** against the live backend (`scripts/neural-field-evidence.mjs`, perfRun).

Two instruments, both recorded over the full run window:

- **cadence** — rAF frame-to-frame delta (includes vsync wait; floor is ~16.7ms at 60Hz).
- **drawCost** — JS cost of the Neural Field frame body (event drain + choreography step +
  projection + full draw), measured with `performance.now()` around the tick's work
  (dev-only seam in `NeuralField.jsx`). This is the number the ≤16.7ms budget constrains,
  since compositing is on the GPU.

| DPR | cadence avg | cadence p95 | drawCost avg | drawCost p95 | frames |
| --- | --- | --- | --- | --- | --- |
| 1 | 16.63ms | 16.8ms | **1.63ms** | **2.3ms** | 57 |
| 2 | 16.87ms | 16.8ms | 1.8ms | 3.3ms | 69 |

(The GPU-lane run at ~45 tok/s is the stressful case: 12 concurrent sweep fronts, edge
underlays, motes. CPU-lane runs carry fewer simultaneous effects.)

## Gate: p95 ≤ 16.7ms at DPR 1 — **PASS**

- drawCost p95 at DPR 1 = 2.3ms, ~14% of budget. Headroom is large enough that neither
  prescribed reduction step (18→12 nodes per disc, dropping faux-glow underlays) was needed.
- cadence held the 60Hz vsync floor throughout (avg 16.63ms); the 16.8ms p95 is vsync
  scheduling jitter, not dropped frames — an empty page measures the same cadence. No
  frame in either run exceeded one vsync interval by more than jitter.
- Drawable count at TinyLlama's 22 layers: 396 nodes + 378 edges ≈ 774 sorted drawables,
  well inside the ≤4,000 sort budget.
- `ctx.shadowBlur` is not used anywhere in the renderer (grep-clean); all glow is wide
  low-alpha underlay strokes/fills.

Raw numbers: `capture-summary.json` (same directory).
