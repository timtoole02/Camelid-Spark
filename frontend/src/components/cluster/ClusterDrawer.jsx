import { useState } from 'react'
import { IconChevronDown, IconCheckCircle, IconInfo, IconWarning, IconError } from '../ui/icons'

const LEVEL = {
  ok: { tone: 'ready', Icon: IconCheckCircle },
  info: { tone: 'info', Icon: IconInfo },
  warn: { tone: 'warn', Icon: IconWarning },
  error: { tone: 'error', Icon: IconError },
}

function levelMeta(level) { return LEVEL[level] || LEVEL.info }

function timeLabel(iso) {
  if (!iso) return ''
  try { return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) } catch { return '' }
}

export function ClusterDrawer({ events, issues, summary, open, onToggle }) {
  const [tab, setTab] = useState('events')
  const problems = issues.filter((i) => i.level === 'warn' || i.level === 'error')
  const latest = events[0]

  return (
    <section className={`cluster-drawer ${open ? 'is-open' : ''}`}>
      <div className="cluster-drawer__bar">
        <div className="cluster-drawer__tabs" role="tablist">
          <button type="button" role="tab" aria-selected={tab === 'events'} className={tab === 'events' ? 'is-active' : ''} onClick={() => { setTab('events'); if (!open) onToggle() }}>
            Events <span className="cluster-drawer__count">{events.length}</span>
          </button>
          <button type="button" role="tab" aria-selected={tab === 'validation'} className={tab === 'validation' ? 'is-active' : ''} onClick={() => { setTab('validation'); if (!open) onToggle() }}>
            Validation {problems.length > 0 && <span className="cluster-drawer__count is-warn">{problems.length}</span>}
          </button>
          <button type="button" role="tab" aria-selected={tab === 'health'} className={tab === 'health' ? 'is-active' : ''} onClick={() => { setTab('health'); if (!open) onToggle() }}>
            Health
          </button>
        </div>
        {!open && latest && (
          <span className={`cluster-drawer__peek is-${levelMeta(latest.level).tone}`}>{latest.message}</span>
        )}
        <button type="button" className="cluster-drawer__toggle" onClick={onToggle} aria-label={open ? 'Collapse' : 'Expand'}>
          <IconChevronDown size={18} style={{ transform: open ? 'none' : 'rotate(180deg)' }} />
        </button>
      </div>

      {open && (
        <div className="cluster-drawer__body">
          {tab === 'events' && (
            <ul className="cluster-log">
              {events.length === 0 && <li className="cluster-log__empty">No events yet.</li>}
              {events.map((e) => {
                const m = levelMeta(e.level)
                return (
                  <li key={e.id} className={`cluster-log__row is-${m.tone}`}>
                    <m.Icon size={15} />
                    <time>{timeLabel(e.time)}</time>
                    <span>{e.message}</span>
                  </li>
                )
              })}
            </ul>
          )}
          {tab === 'validation' && (
            <ul className="cluster-log">
              {issues.map((i, idx) => {
                const m = levelMeta(i.level === 'ok' ? 'ok' : i.level)
                return (
                  <li key={idx} className={`cluster-log__row is-${m.tone}`}>
                    <m.Icon size={15} />
                    <span>{i.message}</span>
                  </li>
                )
              })}
            </ul>
          )}
          {tab === 'health' && (
            <div className="cluster-health">
              <div className="cluster-health__stats">
                <div><strong>{summary.online}/{summary.nodeCount}</strong><span>online</span></div>
                <div><strong>{summary.totalCores || '—'}</strong><span>total cores</span></div>
                <div><strong>{summary.totalRam || '—'} GB</strong><span>total RAM</span></div>
                <div><strong>{summary.gpus}</strong><span>accelerators</span></div>
              </div>
              <div className="cluster-health__roles">
                {summary.roles.filter((r) => r.count > 0).map((r) => (
                  <span key={r.value} className="cluster-health__role"><strong>{r.count}</strong> {r.label}{r.count === 1 ? '' : 's'}</span>
                ))}
                {summary.roles.every((r) => r.count === 0) && <span className="cluster-health__role">No roles assigned yet.</span>}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  )
}

export default ClusterDrawer
