/* DetailsPanel — collapsible right-side readout of the current (or most
   recent) run, plus live errors. All values originate from telemetry. */

function fmt(value, suffix = '') {
  if (value == null || value === '' || Number.isNaN(value)) return '—'
  return `${value}${suffix}`
}

function fmtRate(value) {
  return typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(1)} tok/s` : '—'
}

function fmtBytes(bytes) {
  if (!bytes) return '—'
  if (bytes > 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
  if (bytes > 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  return `${(bytes / 1024).toFixed(0)} KB`
}

export default function DetailsPanel({ store, collapsed, onToggle }) {
  const live = store.getRun()
  const run = live.startedAtMs ? live : store.getLastRun() || live
  const workers = store.getWorkers()
  const finish = run.finish
  const decodeTps = finish?.decodeTps ?? store.liveDecodeTps()
  const prefillTps = finish?.prefillTps ?? store.livePrefillTps()
  const ttft = finish?.ttftMs
    ?? (!run.joinedMidRun && run.decode.startedAtMs && run.startedAtMs ? Math.round(run.decode.startedAtMs - run.startedAtMs) : null)
  const activeNodes = [...workers.values()].filter((w) => w.status === 'active')
  const errors = run.errors

  const statusLabel = live.active
    ? (store.isRunStale() ? 'stalled — no events' : `running · ${run.phase}`)
    : finish
      ? `${finish.status}${finish.finishReason ? ` · ${finish.finishReason}` : ''}`
      : 'idle'

  return (
    <aside className={`observatory-details ${collapsed ? 'is-collapsed' : ''}`}>
      <button type="button" className="observatory-details__toggle" onClick={onToggle} aria-expanded={!collapsed}>
        {collapsed ? '‹' : '›'}
        <span className="sr-only">{collapsed ? 'Open details panel' : 'Collapse details panel'}</span>
      </button>
      {!collapsed && (
        <div className="observatory-details__body">
          <h2>Run details</h2>
          <dl>
            <div><dt>Status</dt><dd data-status={live.active ? 'running' : finish?.status || 'idle'}>{statusLabel}</dd></div>
            <div><dt>Model</dt><dd>{fmt(run.modelId)}</dd></div>
            <div><dt>Quantization</dt><dd>{fmt(run.quantization)}</dd></div>
            <div><dt>Backend path</dt><dd>{fmt(run.backend)}</dd></div>
            <div><dt>Prompt tokens</dt><dd>{fmt(run.promptTokens || null)}</dd></div>
            <div><dt>Generated tokens</dt><dd>{fmt(finish?.completionTokens ?? (run.decode.tokens || null))}</dd></div>
            <div><dt>Prefill</dt><dd>{fmtRate(prefillTps)}</dd></div>
            <div><dt>Decode</dt><dd>{fmtRate(decodeTps)}</dd></div>
            <div><dt>TTFT</dt><dd>{ttft != null ? `${ttft} ms` : '—'}</dd></div>
            <div><dt>KV cache</dt><dd>{run.kv.position ? `${run.kv.position} pos · ${fmtBytes(run.kv.approxBytes)}` : '—'}</dd></div>
            <div><dt>Cluster nodes</dt><dd>{workers.size ? `${activeNodes.length} active of ${workers.size}` : 'local only'}</dd></div>
            <div>
              <dt>Receipt</dt>
              <dd>{run.receipt ? `✓ sealed${run.receipt.reproducible ? ' · reproducible' : ''}` : 'none'}</dd>
            </div>
          </dl>
          {errors.length > 0 && (
            <div className="observatory-details__errors">
              <h3>Errors</h3>
              <ul>
                {errors.slice(-6).map((err, i) => (
                  <li key={`${err.atMs}-${i}`}>
                    <code>{err.code}</code> {err.message}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </aside>
  )
}
