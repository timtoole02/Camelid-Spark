import { useEffect, useId, useRef, useState } from 'react'
import {
  classifyEvidenceState,
  evidenceLabelFromStatus,
  EVIDENCE_STATE_COPY,
  EVIDENCE_STATE_LABELS,
} from '../../lib/evidenceStatus.js'

/* EvidenceChip — the signature component. Renders anywhere the UI makes a
   claim, with a row-scoped label and a verify-popover citing the claim's
   source (capability row id, evidence-bundle manifest path). Pure
   presentation: it displays gate/contract state, it never computes it.

   Props:
   - status: raw contract status string ('supported_exact_row_smoke', …)
   - state:  explicit state override; otherwise derived from status
   - label:  visible row-scoped label; defaults to formatted status
   - source: { rowId, manifest, note, detail } — popover citation
   - size:   'sm' | 'md' (default 'md')
   - asText: render as a non-interactive span (no popover) when there is
             nothing to cite
*/

function StateIcon({ state }) {
  const stroke = 'currentColor'
  const common = { width: 12, height: 12, viewBox: '0 0 16 16', fill: 'none', 'aria-hidden': true }
  switch (state) {
    case 'supported': /* receipt seal: circle + check */
      return (
        <svg {...common}>
          <circle cx="8" cy="8" r="6.2" stroke={stroke} strokeWidth="1.5" />
          <path d="M5.2 8.2l2 2 3.6-4" stroke={stroke} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      )
    case 'evidence': /* bounded brackets */
      return (
        <svg {...common}>
          <path d="M5.5 3H3v10h2.5M10.5 3H13v10h-2.5" stroke={stroke} strokeWidth="1.5" strokeLinecap="round" />
          <circle cx="8" cy="8" r="1.4" fill={stroke} />
        </svg>
      )
    case 'acceptance-target': /* crosshair */
      return (
        <svg {...common}>
          <circle cx="8" cy="8" r="4.6" stroke={stroke} strokeWidth="1.5" />
          <path d="M8 1.5v3M8 11.5v3M1.5 8h3M11.5 8h3" stroke={stroke} strokeWidth="1.5" strokeLinecap="round" />
        </svg>
      )
    case 'groundwork': /* layers */
      return (
        <svg {...common}>
          <path d="M2.5 10.5L8 13.5l5.5-3M2.5 7.5L8 10.5l5.5-3M8 2.5L13.5 5.5 8 8.5 2.5 5.5z" stroke={stroke} strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      )
    case 'planned': /* dashed circle */
      return (
        <svg {...common}>
          <circle cx="8" cy="8" r="5.6" stroke={stroke} strokeWidth="1.5" strokeDasharray="2.4 2.6" />
        </svg>
      )
    case 'unsupported': /* calm slash circle */
      return (
        <svg {...common}>
          <circle cx="8" cy="8" r="5.8" stroke={stroke} strokeWidth="1.4" />
          <path d="M4.5 11.5l7-7" stroke={stroke} strokeWidth="1.4" strokeLinecap="round" />
        </svg>
      )
    case 'error': /* triangle alert */
      return (
        <svg {...common}>
          <path d="M8 2.5l6 10.5H2z" stroke={stroke} strokeWidth="1.4" strokeLinejoin="round" />
          <path d="M8 6.5v3" stroke={stroke} strokeWidth="1.4" strokeLinecap="round" />
          <circle cx="8" cy="11.4" r="0.8" fill={stroke} />
        </svg>
      )
    default: /* neutral dot */
      return (
        <svg {...common}>
          <circle cx="8" cy="8" r="2.4" fill={stroke} />
        </svg>
      )
  }
}

export function EvidenceChip({
  status = '',
  state = null,
  label = '',
  source = null,
  size = 'md',
  asText = false,
  className = '',
  children,
  ...rest
}) {
  const resolvedState = state || classifyEvidenceState(status)
  const text = children || label || evidenceLabelFromStatus(status, EVIDENCE_STATE_LABELS[resolvedState])
  const [open, setOpen] = useState(false)
  const rootRef = useRef(null)
  const popId = useId()
  const hasCitation = Boolean(source && (source.rowId || source.manifest || source.note || source.detail))
  const interactive = !asText

  useEffect(() => {
    if (!open) return undefined
    const onPointer = (event) => {
      if (rootRef.current && !rootRef.current.contains(event.target)) setOpen(false)
    }
    const onKey = (event) => {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onPointer)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onPointer)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  const classes = [
    'ev-chip',
    `ev-chip--${resolvedState}`,
    size === 'sm' ? 'ev-chip--sm' : '',
    interactive ? 'ev-chip--interactive' : '',
    className,
  ].filter(Boolean).join(' ')

  const body = (
    <>
      <span className="ev-chip__icon"><StateIcon state={resolvedState} /></span>
      <span className="ev-chip__label">{text}</span>
    </>
  )

  if (!interactive) {
    return <span className={classes} data-state={resolvedState} {...rest}>{body}</span>
  }

  return (
    <span className="ev-chip-wrap" ref={rootRef}>
      <button
        type="button"
        className={classes}
        data-state={resolvedState}
        aria-label={`${EVIDENCE_STATE_LABELS[resolvedState]} claim: ${typeof text === 'string' ? text : status || 'details'} — view source`}
        aria-expanded={open}
        aria-controls={open ? popId : undefined}
        onClick={() => setOpen((v) => !v)}
        {...rest}
      >
        {body}
      </button>
      {open && (
        <div className="ev-pop" id={popId} role="dialog" aria-label="Claim source">
          <div className="ev-pop__head">
            <span className={`ev-pop__state ev-pop__state--${resolvedState}`}>
              <StateIcon state={resolvedState} />
              {EVIDENCE_STATE_LABELS[resolvedState]}
            </span>
            {status && <code className="ev-pop__status">{String(status)}</code>}
          </div>
          <p className="ev-pop__copy">{EVIDENCE_STATE_COPY[resolvedState]}</p>
          {hasCitation && (
            <dl className="ev-pop__cite">
              {source.rowId && (
                <div className="ev-pop__cite-row">
                  <dt>row</dt>
                  <dd><code>{source.rowId}</code></dd>
                </div>
              )}
              {source.manifest && (
                <div className="ev-pop__cite-row">
                  <dt>manifest</dt>
                  <dd><code>{source.manifest}</code></dd>
                </div>
              )}
              {source.detail && (
                <div className="ev-pop__cite-row">
                  <dt>scope</dt>
                  <dd>{source.detail}</dd>
                </div>
              )}
              {source.note && <p className="ev-pop__note">{source.note}</p>}
            </dl>
          )}
          {!hasCitation && (
            <p className="ev-pop__note">No citation attached — treat this as descriptive copy, not evidence.</p>
          )}
          {source?.rowId && (
            <button
              type="button"
              className="ev-pop__ledger-link"
              onClick={() => {
                setOpen(false)
                window.dispatchEvent(new CustomEvent('camelid:open-ledger', { detail: { rowId: source.rowId } }))
              }}
            >
              View in the evidence ledger →
            </button>
          )}
        </div>
      )}
    </span>
  )
}

export default EvidenceChip
