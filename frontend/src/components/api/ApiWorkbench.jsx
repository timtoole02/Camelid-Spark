import { useMemo, useState } from 'react'
import { EvidenceChip } from '../ui/EvidenceChip'
import { copyText } from '../../lib/markdown'
import { workbenchEndpoints } from '../../lib/apiExamples'
import { beginRequest, emitFirstContent, emitProgress, recordWorkbenchRun } from '../../lib/telemetryLog'

/* API workbench (Phase 5).

   Per-endpoint cards: copyable curl / python / js examples pre-filled with the
   live API base and loaded model id, plus a try-it runner feeding the request
   inspector below. Gating (I1/I3):
   - generation endpoints run only when the shared chat gate is green —
     otherwise the button renders guarded with typed copy;
   - read-only endpoints run whenever the backend answers;
   - fail-closed routes never run and say why.
   The inspector output is operational telemetry — never compatibility
   evidence (I4); its banner chip says so. */

const LANG_TABS = [
  { key: 'curl', label: 'curl' },
  { key: 'python', label: 'Python' },
  { key: 'js', label: 'JS · fetch' },
]

const CHUNK_LOG_LIMIT = 80

function nowMs() {
  return performance.now()
}

async function runEndpoint(endpoint, apiBase, lifecycleId) {
  const base = (apiBase || '').replace(/\/$/, '')
  const url = `${base}${endpoint.path}`
  const init = endpoint.body
    ? { method: endpoint.method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(endpoint.body) }
    : { method: endpoint.method }
  const startedAt = nowMs()
  const record = {
    request: { method: endpoint.method, url, body: endpoint.body || null },
    status: null,
    headersMs: null,
    totalMs: null,
    chunks: [],
    bodyText: null,
    error: null,
    truncated: false,
  }
  try {
    const response = await fetch(url, init)
    record.status = `${response.status} ${response.statusText}`
    record.headersMs = nowMs() - startedAt
    if (lifecycleId) emitFirstContent(lifecycleId, record.headersMs)
    const contentType = response.headers.get('content-type') || ''
    if (endpoint.sse && contentType.includes('text/event-stream') && response.body) {
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        const at = nowMs() - startedAt
        if (lifecycleId) emitProgress(lifecycleId, { tokens: record.chunks.length, tokensPerSec: record.chunks.length / Math.max(at / 1000, 0.001) })
        for (const line of decoder.decode(value, { stream: true }).split('\n')) {
          if (!line.trim()) continue
          if (record.chunks.length >= CHUNK_LOG_LIMIT) {
            record.truncated = true
            continue
          }
          record.chunks.push({ at, line: line.length > 220 ? `${line.slice(0, 220)}…` : line })
        }
      }
    } else {
      const text = await response.text()
      record.bodyText = text.length > 4000 ? `${text.slice(0, 4000)}… (${text.length.toLocaleString()} chars total)` : text
      try {
        record.bodyText = JSON.stringify(JSON.parse(text), null, 2).slice(0, 4000)
      } catch { /* keep raw text */ }
    }
  } catch (error) {
    record.error = String(error.message || error)
  }
  record.totalMs = nowMs() - startedAt
  return record
}

function tryItState(endpoint, { backendOnline, chatUnlocked, tokenizerAvailable }) {
  if (endpoint.gate === 'blocked') {
    return { runnable: false, reason: 'Fail-closed by contract — this route only returns its typed error.' }
  }
  if (!backendOnline) {
    return { runnable: false, reason: 'Backend unreachable; nothing to try against.' }
  }
  if (endpoint.gate === 'chat' && !chatUnlocked) {
    return { runnable: false, reason: 'Requires a loaded supported model — gated exactly like chat: loaded_now, generation_ready, active_model_id, and the exact supported /api/capabilities row.' }
  }
  if (endpoint.gate === 'tokenizer' && !tokenizerAvailable) {
    return { runnable: false, reason: 'Requires a loaded tokenizer; load a local GGUF first.' }
  }
  return { runnable: true, reason: null }
}

function EndpointCard({ endpoint, gateState, onRun, running }) {
  const [lang, setLang] = useState('curl')
  const [copied, setCopied] = useState(false)
  const example = endpoint.examples?.[lang]

  const copyExample = async () => {
    if (!example) return
    await copyText(example)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1500)
  }

  return (
    <article className={`wb-endpoint ${endpoint.gate === 'blocked' ? 'wb-endpoint--blocked' : ''}`}>
      <header className="wb-endpoint__head">
        <div className="wb-endpoint__route">
          <span className="cxv-tag">{endpoint.method}</span>
          <code className="wb-endpoint__path">{endpoint.path}</code>
        </div>
        {endpoint.gate === 'chat' && (
          <EvidenceChip
            state={gateState.runnable ? 'supported' : 'unsupported'}
            label={gateState.runnable ? 'gate green' : 'gate blocked'}
            asText
            size="sm"
          />
        )}
        {endpoint.gate === 'blocked' && (
          <EvidenceChip
            status="fail_closed"
            label="fail-closed"
            source={{ rowId: endpoint.featureRowId, note: endpoint.summary }}
            size="sm"
          />
        )}
      </header>
      <p className="wb-endpoint__summary">{endpoint.summary}</p>

      {endpoint.examples && (
        <>
          <div className="wb-endpoint__langs" role="tablist" aria-label="Example language">
            {LANG_TABS.filter((tab) => endpoint.examples[tab.key]).map((tab) => (
              <button
                key={tab.key}
                type="button"
                role="tab"
                aria-selected={lang === tab.key}
                className={`wb-endpoint__lang ${lang === tab.key ? 'is-active' : ''}`}
                onClick={() => setLang(tab.key)}
              >
                {tab.label}
              </button>
            ))}
            <button type="button" className="wb-endpoint__copy" onClick={copyExample}>{copied ? 'Copied' : 'Copy'}</button>
          </div>
          <pre className="wb-endpoint__example"><code>{example}</code></pre>
        </>
      )}

      <div className="wb-endpoint__actions">
        {gateState.runnable ? (
          <button
            type="button"
            className="cxv-button"
            data-tryit-ready="true"
            data-endpoint={endpoint.id}
            disabled={running}
            onClick={() => onRun(endpoint)}
          >
            {running ? 'Running…' : 'Try it'}
          </button>
        ) : (
          <p className="wb-endpoint__guarded" data-tryit-ready="false" data-endpoint={endpoint.id}>
            {gateState.reason}
          </p>
        )}
      </div>
    </article>
  )
}

export function ApiWorkbench({ apiBase, modelId, backendOnline, chatUnlocked, tokenizerAvailable }) {
  const endpoints = useMemo(() => workbenchEndpoints({ apiBase, modelId }), [apiBase, modelId])
  const [runningId, setRunningId] = useState(null)
  const [inspection, setInspection] = useState(null)

  const onRun = async (endpoint) => {
    if (runningId) return
    setRunningId(endpoint.id)
    setInspection({ endpointId: endpoint.id, pending: true })
    const lifecycleId = beginRequest({ kind: 'workbench', endpoint: endpoint.path, modelId: endpoint.gate === 'chat' ? modelId : null })
    const record = await runEndpoint(endpoint, apiBase, lifecycleId)
    recordWorkbenchRun({
      lifecycleId,
      endpoint: endpoint.path,
      modelId: endpoint.gate === 'chat' ? modelId : null,
      durationMs: record.totalMs,
      headersMs: record.headersMs,
      httpStatus: record.status,
      outcome: record.error || !/^2\d\d/.test(record.status || '') ? 'error' : 'ok',
    })
    setInspection({ endpointId: endpoint.id, pending: false, ...record })
    setRunningId(null)
  }

  return (
    <section className="cxv-card cxv-panel api-workbench" aria-label="API workbench">
      <div className="cxv-section__head">
        <h2>Workbench</h2>
        <span className="cxv-section__count">{endpoints.length} routes on the live surface</span>
      </div>
      <p className="cxv-sub">Copyable examples pre-filled with this API base and the loaded model id. Generation try-its obey the same fail-closed gate as chat; nothing here runs around it.</p>

      <div className="wb-grid">
        {endpoints.map((endpoint) => (
          <EndpointCard
            key={endpoint.id}
            endpoint={endpoint}
            gateState={tryItState(endpoint, { backendOnline, chatUnlocked, tokenizerAvailable })}
            onRun={onRun}
            running={runningId === endpoint.id}
          />
        ))}
      </div>

      {inspection && (
        <div className="wb-inspector" aria-label="Request inspector">
          <div className="wb-inspector__head">
            <h3>Request inspector</h3>
            <EvidenceChip
              state="neutral"
              label="operational telemetry — not compatibility evidence"
              source={{ note: 'Timings and payloads from a single local request in this browser. They prove the wire worked once, not that any row is supported.' }}
              size="sm"
            />
          </div>
          {inspection.pending ? (
            <p className="wb-inspector__note">Running…</p>
          ) : (
            <>
              <div className="wb-inspector__cols">
                <div className="wb-inspector__col">
                  <h4>Rendered request</h4>
                  <pre><code>{`${inspection.request.method} ${inspection.request.url}`}{inspection.request.body ? `\n\n${JSON.stringify(inspection.request.body, null, 2)}` : ''}</code></pre>
                </div>
                <div className="wb-inspector__col">
                  <h4>Response</h4>
                  <p className="wb-inspector__timing">
                    <span>status <b>{inspection.status || '—'}</b></span>
                    <span>headers <b>{inspection.headersMs === null ? '—' : `${Math.round(inspection.headersMs)}ms`}</b></span>
                    <span>total <b>{inspection.totalMs === null ? '—' : `${Math.round(inspection.totalMs)}ms`}</b></span>
                  </p>
                  {inspection.error && <pre className="wb-inspector__error"><code>{inspection.error}</code></pre>}
                  {inspection.bodyText && <pre><code>{inspection.bodyText}</code></pre>}
                  {inspection.chunks.length > 0 && (
                    <ol className="wb-inspector__chunks" aria-label="SSE chunk log">
                      {inspection.chunks.map((chunk, index) => (
                        <li key={index}><span className="wb-inspector__chunk-at">{Math.round(chunk.at)}ms</span><code>{chunk.line}</code></li>
                      ))}
                    </ol>
                  )}
                  {inspection.truncated && <p className="wb-inspector__note">chunk log truncated at {CHUNK_LOG_LIMIT} lines; the request ran to completion</p>}
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </section>
  )
}

export default ApiWorkbench
