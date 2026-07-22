import { useCallback, useEffect, useRef, useState } from 'react'
import { isCompatibilitySupportedForModel } from '../../lib/capabilities'
import { beginCatalogSettlement, catalogDownloadSettlement, completeCatalogAcquisition, reserveCatalogAcquisition } from '../../lib/catalogActivation'
import { SUPPORTED_MODELS } from '../../lib/supportedModels'
import { EvidenceChip } from '../ui/EvidenceChip'

/* Zone 5 — Get models. Curated picks first, then live Hugging Face GGUF search
   (>= 2 chars). Each row shows which lane it WOULD land in (derived: supported
   contract match, oracle-qualified runnable, or not-yet-anchored). Download is
   user-initiated and explicitly confirmed (filename + HF repo + size); no
   background/auto pulls. Live progress renders in the global Downloads zone —
   rows here only reflect their own acquisition state, read from the shared
   downloads poll + the live /api/models/local scan (never localStorage). After a
   download lands, smoke-admission runs for oracle-qualified combos and the model
   appears in its derived local section. */

const GB = 1024 * 1024 * 1024
function prettySize(bytes) {
  if (!bytes) return ''
  if (bytes >= GB) return `${(bytes / GB).toFixed(bytes >= 10 * GB ? 0 : 1)} GB`
  return `${Math.round(bytes / (1024 * 1024))} MB`
}

/* Curated download suggestions (blurbs, "Recommended") may DECORATE catalog rows,
   never place them: lane membership and outcome chips stay derived. */
const CURATED_DECORATION = new Map(SUPPORTED_MODELS.map((item) => [item.catalog_id, item]))

/* Predicted lane for a catalog entry — derived, never a hand-authored label. */
function predictedLane(item, capabilities) {
  // Experimental (live Hugging Face) rows are advisory only: their architecture/quant
  // are filename guesses, so they can never anchor a lane or imply support — even when
  // the filename happens to coincide with a supported contract row. Always not-anchored.
  if (item.group === 'experimental') return 'not_anchored'
  if (isCompatibilitySupportedForModel(capabilities, null, item)) return 'supported'
  if (item.oracle_qualified) return 'compatible'
  return 'not_anchored'
}

function laneChip(lane) {
  if (lane === 'supported') return <EvidenceChip status="supported" asText>Lands in Supported</EvidenceChip>
  if (lane === 'compatible') return <EvidenceChip state="runnable" asText>Experimental · runnable</EvidenceChip>
  return <EvidenceChip state="unsupported" asText>Experimental · unverified</EvidenceChip>
}

/* Capacity advisory for THIS host (fit axis, NOT a support claim — kept on its own
   line, never merged into the lane/support chip). `item.fit` is the backend
   FitVerdict; `unknown`/missing (e.g. unprobed host, experimental rows) shows
   nothing rather than guessing. */
function fitLabel(fit) {
  switch (fit) {
    case 'fits_resident':
      return 'Fits your machine'
    case 'fits_with_offload':
      return 'Fits (GPU + RAM offload)'
    case 'cpu_only_ok':
      return 'Fits (CPU)'
    case 'wont_fit':
      return 'Too big for this machine'
    default:
      return null
  }
}

/* A small CPU/chip glyph so the capacity chip reads as "your hardware" — distinct
   from the support/lane chips. A check (fits) or cross (too big) sits in the die. */
function FitIcon({ bad }) {
  const stroke = 'currentColor'
  return (
    <svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <rect x="4.5" y="4.5" width="7" height="7" rx="1.2" stroke={stroke} strokeWidth="1.3" />
      {bad ? (
        <path d="M6.4 6.4l3.2 3.2M9.6 6.4l-3.2 3.2" stroke={stroke} strokeWidth="1.3" strokeLinecap="round" />
      ) : (
        <path d="M6 8.2l1.4 1.4L10 6.6" stroke={stroke} strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
      )}
      <path
        d="M6.5 2.6v1.9M9.5 2.6v1.9M6.5 11.5v1.9M9.5 11.5v1.9M2.6 6.5h1.9M2.6 9.5h1.9M11.5 6.5h1.9M11.5 9.5h1.9"
        stroke={stroke}
        strokeWidth="1.1"
        strokeLinecap="round"
      />
    </svg>
  )
}

function CatalogRow({
  item,
  capabilities,
  installed,
  activeDownload,
  apiBase,
  installAvailable,
  installBlockedReason,
  onInstallStarted,
  onDownloadAcknowledged,
  onAcquired,
  canceled,
  onDownloadRetry,
  acquisitionLocked,
  onAcquisitionPending,
  onAcquisitionSettled,
  onStartModel,
  onModelStarted,
  onOperationBusy,
}) {
  // phase: idle | confirm | starting | waiting | checking | loading | failed | done
  const [phase, setPhase] = useState('idle')
  const [message, setMessage] = useState('')
  const [isError, setIsError] = useState(false)
  const [failedStage, setFailedStage] = useState('')
  const [settlementTick, setSettlementTick] = useState(0)
  const sawDownloadRef = useRef(false)
  const startedAtRef = useRef(0)
  const settledAtRef = useRef(0)
  const acquisitionModeRef = useRef('download')
  const acquisitionItemRef = useRef(item)
  const settlementInFlightRef = useRef(false)
  const lane = predictedLane(item, capabilities)
  const decoration = item.group === 'experimental' ? null : CURATED_DECORATION.get(item.catalog_id)
  const downloadAndStart = lane === 'supported' && item.fit !== 'wont_fit'
  const smokeAfterDownload = item.group !== 'experimental'
    && item.fit !== 'wont_fit'
    && !downloadAndStart
    && item.oracle_qualified
  const acquisitionMode = downloadAndStart ? 'start' : smokeAfterDownload ? 'smoke' : 'download'
  const downloading = activeDownload?.status === 'downloading'
  const rejoinableDownload = downloading || activeDownload?.status === 'completed'
  const operationBusy = phase === 'checking' || phase === 'loading'

  useEffect(() => {
    onOperationBusy?.(item.catalog_id, operationBusy)
    return () => {
      if (operationBusy) onOperationBusy?.(item.catalog_id, false)
    }
  }, [item.catalog_id, onOperationBusy, operationBusy])

  useEffect(() => {
    if (phase !== 'idle' || !rejoinableDownload || acquisitionLocked) return
    if (onAcquisitionPending?.(item) === false) return
    sawDownloadRef.current = true
    startedAtRef.current = Date.now()
    settledAtRef.current = 0
    acquisitionModeRef.current = activeDownload?.continuation_mode || acquisitionMode
    acquisitionItemRef.current = item
    settlementInFlightRef.current = false
    setMessage('Rejoined the active download.')
    setIsError(false)
    setPhase('waiting')
  }, [acquisitionLocked, acquisitionMode, activeDownload?.continuation_mode, item, onAcquisitionPending, phase, rejoinableDownload])

  const finishLanded = useCallback(async () => {
    if (!beginCatalogSettlement(settlementInFlightRef)) return
    setIsError(false)
    setFailedStage('')
    onAcquired?.()
    const result = await completeCatalogAcquisition({
      item: acquisitionItemRef.current,
      mode: acquisitionModeRef.current,
      apiBase,
      loadModelForChat: onStartModel,
      onStage: setPhase,
    })
    setMessage(result.message)
    if (!result.ok) {
      setFailedStage(result.stage)
      setIsError(true)
      setPhase('failed')
      onAcquisitionSettled?.(item.catalog_id)
      return
    }
    await onAcquired?.()
    try {
      const ack = await fetch(`${apiBase}/api/models/catalog/ack`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: item.catalog_id }),
      })
      if (!ack.ok) throw new Error(`acknowledgement failed (HTTP ${ack.status})`)
    } catch (error) {
      setFailedStage(result.started ? 'loading' : 'checking')
      setIsError(true)
      setMessage(`The model is ready, but Camelid could not finalize the download state: ${String(error?.message || error)}`)
      setPhase('failed')
      onAcquisitionSettled?.(item.catalog_id)
      return
    }
    await onDownloadAcknowledged?.()
    setPhase('done')
    onAcquisitionSettled?.(item.catalog_id)
    if (result.started) onModelStarted?.()
  }, [apiBase, item, onAcquired, onAcquisitionSettled, onDownloadAcknowledged, onModelStarted, onStartModel])

  const retryAcquisition = () => {
    if (onAcquisitionPending?.(item) === false) {
      setMessage('Wait for the current model acquisition to finish, then retry.')
      return
    }
    settlementInFlightRef.current = false
    finishLanded()
  }

  // The row watches the SHARED downloads poll + local scan instead of polling
  // itself: downloading -> (gone + on disk) = landed; (gone + not on disk after
  // having been seen) = failed or canceled.
  useEffect(() => {
    if (phase !== 'waiting') return undefined
    let refreshing = false
    const refreshSettlement = async () => {
      if (refreshing) return
      refreshing = true
      await onAcquired?.()
      setSettlementTick((value) => value + 1)
      refreshing = false
    }
    refreshSettlement()
    const timer = setInterval(refreshSettlement, 1000)
    return () => clearInterval(timer)
  }, [phase, onAcquired])

  useEffect(() => {
    if (phase !== 'waiting') return
    if (canceled) {
      settlementInFlightRef.current = false
      setPhase('idle')
      setIsError(true)
      setMessage('Download canceled. It can be retried.')
      onAcquisitionSettled?.(item.catalog_id)
      return
    }
    const settlement = catalogDownloadSettlement({
      downloading,
      installed,
      sawDownload: sawDownloadRef.current,
      settledAt: settledAtRef.current,
      startedAt: startedAtRef.current,
    })
    sawDownloadRef.current = settlement.sawDownload
    settledAtRef.current = settlement.settledAt
    if (settlement.action === 'landed') {
      finishLanded()
      return
    }
    if (settlement.action === 'failed') {
      setPhase('idle')
      setIsError(true)
      setMessage('Download did not complete (canceled or failed). It can be retried.')
      onAcquisitionSettled?.(item.catalog_id)
    }
  }, [phase, downloading, installed, canceled, settlementTick, finishLanded, item.catalog_id, onAcquisitionSettled])

  const confirmDownload = async () => {
    setPhase('starting')
    setMessage('')
    setIsError(false)
    sawDownloadRef.current = false
    startedAtRef.current = Date.now()
    settledAtRef.current = 0
    acquisitionModeRef.current = acquisitionMode
    acquisitionItemRef.current = item
    settlementInFlightRef.current = false
    onDownloadRetry?.(item.catalog_id)
    try {
      const res = await fetch(`${apiBase}/api/models/catalog/install`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          catalog_id: item.catalog_id,
          repo_id: item.repo_id,
          filename: item.filename,
          size_bytes: item.size_bytes,
          continuation_mode: acquisitionMode,
        }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        if (res.status !== 409 || body?.error?.code !== 'download_already_running') {
          throw new Error(body?.error?.message || `download failed (HTTP ${res.status})`)
        }
      }
      setPhase('waiting')
      onInstallStarted?.()
    } catch (err) {
      setPhase('idle')
      setIsError(true)
      setMessage(String(err?.message || err))
      onAcquisitionSettled?.(item.catalog_id)
    }
  }

  const openConfirmation = () => {
    if (onAcquisitionPending?.(item) === false) return
    setPhase('confirm')
  }

  const activeStage = phase === 'checking' ? 1 : phase === 'loading' ? 2 : 0
  const showProgress = ['starting', 'waiting', 'checking', 'loading'].includes(phase)

  return (
    <article className={`catalog-row${lane === 'not_anchored' ? ' catalog-row--advisory' : ''}`}>
      <div className="catalog-row-head">
        <div className="catalog-row-id">
          <span className="catalog-row-name">
            {item.name}
            {decoration?.recommended ? <span className="catalog-row-recommended">Recommended</span> : null}
          </span>
          <span className="catalog-row-meta">
            {item.repo_id} · {item.filename} · {prettySize(item.size_bytes)}
            {item.architecture ? ` · ${item.architecture}` : ''}
          </span>
        </div>
        {laneChip(lane)}
      </div>
      {fitLabel(item.fit) ? (
        <div className="catalog-fit-row">
          <span
            className={`catalog-fit-chip catalog-fit-chip--${item.fit === 'wont_fit' ? 'bad' : 'good'}${
              item.fit_confidence === 'approx' ? ' catalog-fit-chip--estimate' : ''
            }`}
            title={
              item.fit_confidence === 'exact'
                ? "Sized from the model's real dimensions (KV cache computed exactly)"
                : 'Estimate — upgrades to exact once the model header has been read'
            }
          >
            <FitIcon bad={item.fit === 'wont_fit'} />
            {item.fit_confidence === 'approx' ? '~ ' : ''}
            {fitLabel(item.fit)}
          </span>
          {Array.isArray(item.task_tags) && item.task_tags.length ? (
            <span className="catalog-fit-tags">
              <span className="catalog-fit-tags-label">best for</span>
              {item.task_tags.map((tag) => (
                <span key={tag} className="catalog-fit-tag">
                  {tag}
                </span>
              ))}
            </span>
          ) : null}
        </div>
      ) : null}
      {decoration?.blurb ? <p className="catalog-row-blurb">{decoration.blurb}</p> : null}

      {showProgress ? (
        <div className="catalog-start" role="status" aria-live="polite">
          {downloadAndStart ? (
            <ol className="catalog-start-steps" aria-label="Download and start progress">
              {['Download', 'Check', 'Load'].map((label, index) => (
                <li key={label} className={index < activeStage ? 'is-done' : index === activeStage ? 'is-active' : ''}>
                  <span>{index < activeStage ? '✓' : index + 1}</span>
                  {label}
                </li>
              ))}
            </ol>
          ) : smokeAfterDownload ? (
            <ol className="catalog-start-steps catalog-start-steps--two" aria-label="Download and check progress">
              {['Download', 'Check'].map((label, index) => (
                <li key={label} className={index < activeStage ? 'is-done' : index === activeStage ? 'is-active' : ''}>
                  <span>{index < activeStage ? '✓' : index + 1}</span>
                  {label}
                </li>
              ))}
            </ol>
          ) : null}
          <p className="catalog-row-faint">
            {phase === 'checking'
              ? 'Download complete — checking the model…'
              : phase === 'loading'
                ? 'Check passed — loading the model for Chat…'
                : downloading
                  ? 'Downloading — live progress is shown above.'
                  : 'Starting download…'}
          </p>
        </div>
      ) : phase === 'failed' ? (
        <div className="catalog-start-failure" role="alert">
          <p className="catalog-row-error">{message}</p>
          <p className="catalog-row-faint">The file is still on disk. Camelid has not opened Chat.</p>
          <button type="button" className="catalog-row-action" onClick={retryAcquisition}>
            {failedStage === 'checking' ? 'Retry check' : 'Retry start'}
          </button>
        </div>
      ) : phase === 'done' ? (
        <p className={isError ? 'catalog-row-error' : 'catalog-row-faint'}>{message}</p>
      ) : installed ? (
        <p className="catalog-row-faint">Already on disk — shown in its section above.</p>
      ) : phase === 'idle' ? (
        <>
          {item.group === 'experimental' ? (
            <p className="catalog-row-faint">
              From Hugging Face — unverified, no parity claim. Architecture/quant
              {item.architecture || item.quant ? ` (guessed ${[item.architecture, item.quant].filter(Boolean).join(' / ')})` : ''}{' '}
              are read from the filename, not the model; the real lane is only known after it loads.
            </p>
          ) : lane === 'not_anchored' ? (
            <p className="catalog-row-faint">
              Its {item.architecture}/{item.quant} combo is not yet in the runnable lane — still
              downloadable; it lands in Experimental and loads through the experimental chat path.
            </p>
          ) : null}
          {message ? <p className={isError ? 'catalog-row-error' : 'catalog-row-faint'}>{message}</p> : null}
          {installAvailable ? (
            <button
              type="button"
              className="catalog-row-action"
              onClick={openConfirmation}
              disabled={acquisitionLocked}
              title={acquisitionLocked ? 'Wait for the current model acquisition to finish' : undefined}
            >
              {downloadAndStart ? 'Download and start…' : 'Download…'}
            </button>
          ) : (
            <>
              <button type="button" className="catalog-row-action" disabled>
                Download unavailable
              </button>
              <p className="catalog-row-faint">{installBlockedReason}</p>
            </>
          )}
        </>
      ) : phase === 'confirm' ? (
        <div className="catalog-confirm">
          <p>
            Download <strong>{item.filename}</strong> from <code>{item.repo_id}</code> (
            {prettySize(item.size_bytes)})? This pulls from HuggingFace into your local models folder.
            {downloadAndStart ? ' Camelid will check it, load it, and open Chat after the download.' : ''}
          </p>
          <div className="catalog-confirm-actions">
            <button type="button" className="catalog-row-action" onClick={confirmDownload}>
              Confirm download
            </button>
            <button
              type="button"
              className="catalog-row-cancel"
              onClick={() => {
                setPhase('idle')
                onAcquisitionSettled?.(item.catalog_id)
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      ) : null}
    </article>
  )
}

/* Persistent, non-dismissible marker for the experimental group. Reuses the
   unsupported EvidenceChip so it can never read as an endorsement. */
function ExperimentalMarker() {
  return (
    <span className="catalog-experimental-marker">
      <EvidenceChip state="unsupported" asText>Experimental — unverified, no parity claim</EvidenceChip>
    </span>
  )
}

function CatalogGroup({ title, marker, items, emptyText, renderRow }) {
  return (
    <section className="catalog-group">
      <div className="catalog-group-head">
        <h3>{title}</h3>
        {marker}
      </div>
      <div className="catalog-list">
        {items.map(renderRow)}
        {items.length === 0 ? <p className="lane-empty">{emptyText}</p> : null}
      </div>
    </section>
  )
}

export function CatalogLaneBrowse({
  apiBase = '',
  capabilities,
  localFilenames = new Set(),
  downloads = [],
  installAvailable = true,
  installBlockedReason = '',
  onInstallStarted,
  onDownloadAcknowledged,
  onAcquired,
  canceledCatalogIds = new Set(),
  onDownloadRetry,
  onStartModel,
  onModelStarted,
  onOperationBusy,
}) {
  const base = (apiBase || '').replace(/\/$/, '')
  const [items, setItems] = useState(null)
  const [query, setQuery] = useState('')
  const [debouncedQuery, setDebouncedQuery] = useState('')
  const [nextCursor, setNextCursor] = useState(null)
  const [loadingMore, setLoadingMore] = useState(false)
  const [error, setError] = useState('')
  const [pendingCatalogId, setPendingCatalogId] = useState('')
  const [pendingItem, setPendingItem] = useState(null)
  const pendingCatalogIdRef = useRef('')
  const requestSequenceRef = useRef(0)

  // Debounce the query so each keystroke doesn't fire a live Hugging Face search.
  useEffect(() => {
    const t = setTimeout(() => setDebouncedQuery(query.trim()), 350)
    return () => clearTimeout(t)
  }, [query])

  const load = useCallback(async () => {
    const sequence = ++requestSequenceRef.current
    setError('')
    try {
      const params = debouncedQuery ? `?query=${encodeURIComponent(debouncedQuery)}` : ''
      const res = await fetch(`${base}/api/models/catalog${params}`)
      if (!res.ok) throw new Error(`catalog HTTP ${res.status}`)
      const body = await res.json()
      if (sequence !== requestSequenceRef.current) return
      setItems(body.items || [])
      setNextCursor(body.next_cursor || null)
    } catch (err) {
      if (sequence !== requestSequenceRef.current) return
      setError(String(err?.message || err))
    }
  }, [base, debouncedQuery])

  useEffect(() => {
    load()
  }, [load])

  const reserveAcquisition = useCallback((item) => {
    const catalogId = item.catalog_id
    const reservation = reserveCatalogAcquisition(pendingCatalogIdRef.current, catalogId)
    if (!reservation.accepted) return false
    pendingCatalogIdRef.current = reservation.catalogId
    setPendingCatalogId(catalogId)
    setPendingItem(item)
    return true
  }, [])

  const settleAcquisition = useCallback((catalogId) => {
    if (pendingCatalogIdRef.current !== catalogId) return
    pendingCatalogIdRef.current = ''
    setPendingCatalogId('')
    setPendingItem(null)
  }, [])

  // Append the next page of experimental (Hugging Face) results.
  const loadMore = useCallback(async () => {
    if (!nextCursor || !debouncedQuery) return
    const sequence = requestSequenceRef.current
    setLoadingMore(true)
    try {
      const params = `?query=${encodeURIComponent(debouncedQuery)}&cursor=${encodeURIComponent(nextCursor)}`
      const res = await fetch(`${base}/api/models/catalog${params}`)
      if (!res.ok) throw new Error(`catalog HTTP ${res.status}`)
      const body = await res.json()
      if (sequence !== requestSequenceRef.current) return
      const more = (body.items || []).filter((it) => it.group === 'experimental')
      setItems((prev) => {
        const seen = new Set((prev || []).map((it) => it.catalog_id))
        return [...(prev || []), ...more.filter((it) => !seen.has(it.catalog_id))]
      })
      setNextCursor(body.next_cursor || null)
    } catch (err) {
      if (sequence !== requestSequenceRef.current) return
      setError(String(err?.message || err))
    } finally {
      setLoadingMore(false)
    }
  }, [base, debouncedQuery, nextCursor])

  const renderRow = (item) => (
    <CatalogRow
      key={item.catalog_id}
      item={item}
      capabilities={capabilities}
      installed={localFilenames.has(item.filename)}
      activeDownload={downloads.find((download) => download.id === item.catalog_id)}
      apiBase={base}
      installAvailable={installAvailable}
      installBlockedReason={installBlockedReason}
      onInstallStarted={onInstallStarted}
      onDownloadAcknowledged={onDownloadAcknowledged}
      onAcquired={onAcquired}
      canceled={canceledCatalogIds.has(item.catalog_id)}
      onDownloadRetry={onDownloadRetry}
      acquisitionLocked={Boolean(pendingCatalogId && pendingCatalogId !== item.catalog_id)}
      onAcquisitionPending={reserveAcquisition}
      onAcquisitionSettled={settleAcquisition}
      onStartModel={onStartModel}
      onModelStarted={onModelStarted}
      onOperationBusy={onOperationBusy}
    />
  )

  const visibleItems = pendingItem && !(items || []).some((item) => item.catalog_id === pendingItem.catalog_id)
    ? [pendingItem, ...(items || [])]
    : (items || [])
  const curated = visibleItems.filter((it) => it.group !== 'experimental')
  const experimental = visibleItems.filter((it) => it.group === 'experimental')
  const searching = debouncedQuery.length >= 2

  return (
    <div className="catalog-lane-browse">
      <div className="local-lane-head">
        <h2>Get models</h2>
      </div>
      <p className="local-lane-intro">
        Curated picks are pinned and known-good. Searching also browses live Hugging Face GGUFs as an
        experimental group — those are unverified and carry no parity claim. Downloads are explicit
        and confirmed; progress appears in Downloads above, and the model joins its derived section
        when the file lands.
      </p>
      <input
        className="catalog-search"
        aria-label="Search model catalog"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        disabled={Boolean(pendingCatalogId)}
        placeholder="Search curated picks and live Hugging Face GGUFs (name, repo, filename)"
      />
      {error ? (
        <p className="lane-error">
          {items === null ? `Catalog unavailable: ${error}` : error}
        </p>
      ) : null}
      {items === null && !error ? <p className="lane-empty">Loading catalog…</p> : null}

      {items !== null || error ? (
        <CatalogGroup
          title="Curated"
          marker={null}
          items={curated}
          emptyText={debouncedQuery ? 'No curated entries match.' : 'No curated entries available.'}
          renderRow={renderRow}
        />
      ) : null}

      {searching && items !== null ? (
        <>
          <CatalogGroup
            title="Experimental (Hugging Face)"
            marker={<ExperimentalMarker />}
            items={experimental}
            emptyText="No live Hugging Face GGUFs match (or the Hub is unreachable)."
            renderRow={renderRow}
          />
          {nextCursor ? (
            <button
              type="button"
              className="catalog-row-action"
              onClick={loadMore}
              disabled={loadingMore || Boolean(pendingCatalogId)}
            >
              {loadingMore ? 'Loading…' : 'Load more from Hugging Face'}
            </button>
          ) : null}
        </>
      ) : null}
    </div>
  )
}
