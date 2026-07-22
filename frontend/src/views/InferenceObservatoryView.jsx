import { useEffect, useMemo, useState } from 'react'
import FlowBench from '../components/observatory/FlowBench'
import NeuralField from '../components/observatory/NeuralField'
import DetailsPanel from '../components/observatory/DetailsPanel'
import { useInferenceTelemetry } from '../hooks/useInferenceTelemetry'
import { getTelemetrySnapshot, subscribeTelemetry, summarizeTelemetry } from '../lib/telemetryLog'
import { getChatGateState } from '../lib/chatGate'
import { EvidenceChip } from '../components/ui/EvidenceChip'
import { IconObservatory } from '../components/ui/icons'

/* Observatory (Phase 6.1 — "The Flow Bench"): inference rendered as liquid.
   The centerpiece canvas and the instrument rail consume the SAME lifecycle
   bus as the Telemetry view, so the art and the numbers cannot disagree. The
   backend-reported run panel (camelid.telemetry/v1 SSE) remains below as a
   separate, explicitly backend-side instrument. */

const fmtMs = (value) => (Number.isFinite(value) ? (value >= 1000 ? `${(value / 1000).toFixed(1)}s` : `${Math.round(value)}ms`) : '—')
const fmtRate = (value) => (Number.isFinite(value) ? `${value >= 10 ? Math.round(value) : value.toFixed(1)} tok/s` : '—')

/* Renderer mode. Neural Field is the default: its Phase 5 gate passed
   2026-07-02 (frames + PERF p95 2.3ms@DPR1 + truthfulness audit + build —
   see design-evidence/neural-field/). A stored choice always wins. */
const RENDERER_KEY = 'camelid.observatory.renderer'
const RENDERERS = ['flowbench', 'neuralfield']

function initialRenderer() {
  try {
    const stored = window.localStorage.getItem(RENDERER_KEY)
    return RENDERERS.includes(stored) ? stored : 'neuralfield'
  } catch {
    return 'neuralfield'
  }
}

const HEADER_COPY = {
  flowbench: {
    title: 'The Flow Bench',
    sub: 'Inference as liquid: prompt ink drifts until the first token bursts it, generation ink flows at the real decode rate, errors bloom and refuse to mix. Every motion traces to a request in the log — an idle backend settles to stillness, never fake motion.',
  },
  neuralfield: {
    title: 'The Neural Field',
    sub: 'The loaded model as geometry, inference as light: real layer counts shape the tunnel, real KV occupancy fills the column, real sampler candidates bloom at the outlet. The 18-node discs are stylized layer cross-sections (head counts are not in telemetry), and on GPU-resident lanes the traversal is a token-paced sweep — the token really crossed every layer; per-layer timing is not observable there. Idle = the network at rest.',
  },
}

export default function InferenceObservatoryView({ apiBase, runtime = null, selectedModel = null, capabilities = null }) {
  const store = useInferenceTelemetry(apiBase)
  const [snapshot, setSnapshot] = useState(getTelemetrySnapshot)
  const [highlightId, setHighlightId] = useState(null)
  const [detailsCollapsed, setDetailsCollapsed] = useState(true)
  const [systemReduced] = useState(() => typeof window !== 'undefined' && Boolean(window.matchMedia?.('(prefers-reduced-motion: reduce)').matches))
  const [manualReduced, setManualReduced] = useState(false)
  const reducedMotion = systemReduced || manualReduced
  const [renderer, setRenderer] = useState(initialRenderer)

  const pickRenderer = (mode) => {
    setRenderer(mode)
    try {
      window.localStorage.setItem(RENDERER_KEY, mode)
    } catch { /* persistence is best-effort */ }
  }

  useEffect(() => subscribeTelemetry(() => setSnapshot(getTelemetrySnapshot())), [])

  const { requests } = snapshot
  const summary = useMemo(() => summarizeTelemetry(requests), [requests])
  const recent = useMemo(() => requests.slice(-9).reverse(), [requests])
  const gate = getChatGateState(capabilities, selectedModel, runtime)
  const activeModelId = runtime?.active_model_id || null

  return (
    <section className="observatory-view cxv flowbench-view">
      <header className="cxv-head">
        <div className="cxv-head__copy">
          <p className="cxv-kicker"><IconObservatory size={14} /> Observatory</p>
          <h1>{HEADER_COPY[renderer].title}</h1>
          <p className="cxv-sub">{HEADER_COPY[renderer].sub}</p>
        </div>
        <div className="cxv-head__actions">
          <div className="observatory-renderer-toggle" role="group" aria-label="Centerpiece renderer">
            <button
              type="button"
              className={renderer === 'flowbench' ? 'is-active' : ''}
              aria-pressed={renderer === 'flowbench'}
              onClick={() => pickRenderer('flowbench')}
            >
              Flow Bench
            </button>
            <button
              type="button"
              className={renderer === 'neuralfield' ? 'is-active' : ''}
              aria-pressed={renderer === 'neuralfield'}
              onClick={() => pickRenderer('neuralfield')}
            >
              Neural Field
            </button>
          </div>
          <EvidenceChip
            state="neutral"
            label="operational telemetry — not compatibility evidence"
            source={{ note: 'The fluid renders counts and timings from this session’s real requests only — never token text, never support claims. Bounded contract evidence lives in the Compatibility ledger.' }}
            size="sm"
          />
        </div>
      </header>

      <div className="flowbench-stage">
        {renderer === 'neuralfield'
          ? <NeuralField apiBase={apiBase} reducedMotion={reducedMotion} />
          : <FlowBench reducedMotion={reducedMotion} highlightId={highlightId} />}
        <aside className="flowbench-rail" aria-label="Live instruments">
          <div className="flowbench-rail__tiles">
            <div className="cxv-stat"><span>Requests</span><strong>{summary.total}</strong><small>{summary.errors} error{summary.errors === 1 ? '' : 's'}</small></div>
            <div className="cxv-stat"><span>TTFT med</span><strong>{fmtMs(summary.medianTtftMs)}</strong><small>client-measured</small></div>
            <div className="cxv-stat"><span>Decode med</span><strong>{fmtRate(summary.medianTokensPerSec)}</strong><small>client-measured</small></div>
          </div>
          <div className="flowbench-rail__model">
            <span className="flowbench-rail__model-id">{activeModelId || 'no model loaded'}</span>
            {gate.hint?.target && (
              <EvidenceChip
                status={gate.hint.target.status}
                state={gate.contractSupported ? 'supported' : null}
                source={{ rowId: gate.hint.target.id }}
                size="sm"
              />
            )}
          </div>
          <ol className="flowbench-rail__log" aria-label="Recent requests — hover to highlight the ink thread">
            {recent.length === 0 && <li className="flowbench-rail__empty">No session traffic yet — the bench stays still until a real request runs.</li>}
            {recent.map((record) => (
              <li
                key={record.id}
                className={`flowbench-rail__row ${record.outcome !== 'ok' ? 'is-error' : ''} ${highlightId === record.id ? 'is-highlit' : ''}`}
                onMouseEnter={() => setHighlightId(record.id)}
                onMouseLeave={() => setHighlightId(null)}
              >
                <code>{record.id}</code>
                <span>{record.kind === 'chat' ? 'chat' : record.endpoint}</span>
                <span>{record.outcome}</span>
                <span>{fmtMs(record.durationMs)}</span>
              </li>
            ))}
          </ol>
          <div className="flowbench-rail__foot">
            <button type="button" className="cxturn__action" onClick={() => setManualReduced((value) => !value)} aria-pressed={manualReduced}>
              {reducedMotion ? 'Motion: static field' : 'Motion: live'}
            </button>
            <a className="flowbench-rail__link" href="#telemetry">full request log &amp; health history →</a>
          </div>
        </aside>
      </div>

      <div className="flowbench-backend">
        <DetailsPanel store={store} collapsed={detailsCollapsed} onToggle={() => setDetailsCollapsed((value) => !value)} />
        <p className="tele-note">backend-reported stream (camelid.telemetry/v1) — a separate backend-side instrument, also not support evidence</p>
      </div>
    </section>
  )
}
