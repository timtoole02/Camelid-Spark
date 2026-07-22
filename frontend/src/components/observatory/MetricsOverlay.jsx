/* MetricsOverlay — Engineer Mode: live numbers straight from telemetry.
   Every value is either a real reported number or an em dash; nothing is
   estimated client-side beyond rates computed from real event timestamps. */

function fmtTps(value) {
  return typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(1)} tok/s` : '—'
}

function fmtMs(value) {
  return typeof value === 'number' && Number.isFinite(value) ? `${Math.round(value)} ms` : '—'
}

function fmtBytes(bytes) {
  if (!bytes) return '—'
  if (bytes > 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
  if (bytes > 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  return `${(bytes / 1024).toFixed(0)} KB`
}

export default function MetricsOverlay({ store }) {
  const run = store.getRun()
  const finish = run.finish
  const decodeTps = finish?.decodeTps ?? store.liveDecodeTps()
  const prefillTps = finish?.prefillTps ?? store.livePrefillTps()
  const ttft = finish?.ttftMs
    ?? (!run.joinedMidRun && run.decode.startedAtMs && run.startedAtMs ? run.decode.startedAtMs - run.startedAtMs : null)
  const workers = store.getWorkers()
  const activeNodes = [...workers.values()].filter((wkr) => wkr.status === 'active').length

  const cells = [
    ['decode', fmtTps(decodeTps)],
    ['prefill', fmtTps(prefillTps)],
    ['ttft', fmtMs(ttft)],
    ['tokens', run.decode.tokens ? `${run.decode.tokens} out · ${run.promptTokens} in` : run.promptTokens ? `${run.promptTokens} in` : '—'],
    ['context', run.kv.position ? `${run.kv.position} / ${run.contextLength || run.kv.capacity || '—'}` : '—'],
    ['kv cache', fmtBytes(run.kv.approxBytes)],
    ['layer', run.layerEventsSeen && run.activeLayer != null ? `${run.activeLayer + 1} / ${run.layersTotal}` : run.layersTotal ? `${run.layersTotal} (resident)` : '—'],
    ['backend', run.backend || '—'],
    ['quant', run.quantization || '—'],
    ['nodes', workers.size ? `${activeNodes} active / ${workers.size}` : 'local only'],
  ]

  return (
    <div className="observatory-metrics" role="status">
      {cells.map(([label, value]) => (
        <div key={label} className="observatory-metric">
          <span className="observatory-metric__label">{label}</span>
          <span className="observatory-metric__value">{value}</span>
        </div>
      ))}
    </div>
  )
}
