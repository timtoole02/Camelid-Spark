import { formatBytes } from '../../lib/formatters'

/* Zone 4 — one global live status area for active downloads. Progress comes ONLY
   from the /api/models/catalog/downloads poll (bytes/total) owned by
   useModelsPageData; there is no client-side download record. The panel renders
   nothing when idle, but its data source (the spine hook) stays mounted with the
   page so status is never lost on navigation. */

function pct(dl) {
  if (!dl.total_bytes) return 0
  return Math.max(0, Math.min(100, Math.round((dl.bytes_downloaded / dl.total_bytes) * 100)))
}

export function DownloadsPanel({ downloads = [], onCancel, cancelingIds = new Set() }) {
  const active = downloads.filter((d) => d.status === 'downloading')
  if (!active.length) return null

  return (
    <section className="lane-section downloads-panel" aria-label="Downloads in progress">
      <header className="lane-section-head">
        <h3>
          Downloads <span className="lane-section-count">{active.length}</span>
        </h3>
        <p className="lane-section-sub">Live from the backend download list; canceling removes the partial file.</p>
      </header>
      <div className="lane-section-body">
        {active.map((dl) => (
          <article key={dl.id} className="lane-row downloads-row" aria-label={`Downloading ${dl.filename}`}>
            <div className="lane-row-head">
              <div className="lane-row-id">
                <span className="lane-row-name">{dl.filename}</span>
                <span className="lane-row-meta">
                  {formatBytes(dl.bytes_downloaded)} / {dl.total_bytes ? formatBytes(dl.total_bytes) : 'size unknown'}
                  {dl.total_bytes ? ` · ${pct(dl)}%` : ''}
                </span>
              </div>
              <button
                type="button"
                className="lane-row-action downloads-cancel"
                onClick={() => onCancel(dl.id)}
                disabled={cancelingIds.has(dl.id)}
              >
                {cancelingIds.has(dl.id) ? 'Canceling…' : 'Cancel'}
              </button>
            </div>
            <div
              className="downloads-progress"
              role="progressbar"
              aria-valuenow={pct(dl)}
              aria-valuemin={0}
              aria-valuemax={100}
            >
              <span style={{ width: `${pct(dl)}%` }} />
            </div>
          </article>
        ))}
      </div>
    </section>
  )
}

export default DownloadsPanel
