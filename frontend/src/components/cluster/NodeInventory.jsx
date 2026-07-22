import { useMemo, useState } from 'react'
import { DeviceIcon } from './DeviceIcon'
import { StatusDot } from '../ui/StatusDot'
import { Button } from '../ui/Button'
import { roleLabel, statusTone } from '../../lib/clusterModel'
import { IconPlus, IconSearch, IconCpu } from '../ui/icons'

export function NodeInventory({ nodes, summary, selection, onSelect, onAddServer }) {
  const [query, setQuery] = useState('')
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    const list = q
      ? nodes.filter((n) => `${n.display_name} ${n.hostname} ${n.ip_address || ''} ${n.roles.join(' ')} ${n.node_type}`.toLowerCase().includes(q))
      : nodes
    return [...list].sort((a, b) => a.display_name.localeCompare(b.display_name))
  }, [nodes, query])

  return (
    <aside className="cluster-inventory">
      <div className="cluster-panel__head">
        <h3>Nodes</h3>
        <Button variant="tonal" size="sm" icon={<IconPlus size={15} />} onClick={onAddServer}>Add</Button>
      </div>

      <div className="cluster-inventory__stats">
        <div><strong>{summary.nodeCount}</strong><span>nodes</span></div>
        <div><strong>{summary.online}</strong><span>online</span></div>
        <div><strong>{summary.totalCores || '—'}</strong><span>cores</span></div>
        <div><strong>{summary.totalRam ? `${summary.totalRam}` : '—'}</strong><span>GB RAM</span></div>
      </div>

      <div className="cluster-inventory__search">
        <IconSearch size={15} />
        <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search nodes" aria-label="Search nodes" />
      </div>

      <div className="cluster-inventory__list">
        {filtered.length === 0 && <p className="cluster-inventory__empty">{nodes.length ? 'No matches.' : 'No nodes yet.'}</p>}
        {filtered.map((node) => {
          const tone = statusTone(node.status)
          const active = selection.kind === 'node' && selection.id === node.id
          return (
            <button
              key={node.id}
              type="button"
              className={`cluster-inv-row ${active ? 'is-active' : ''}`}
              onClick={() => onSelect('node', node.id)}
            >
              <span className="cluster-inv-row__icon"><DeviceIcon type={node.node_type} size={18} /></span>
              <span className="cluster-inv-row__body">
                <strong title={node.display_name}>{node.display_name}</strong>
                <small>{node.hostname || node.ip_address || 'no address'} · {node.roles.map(roleLabel).join(', ') || 'no role'}</small>
              </span>
              <StatusDot tone={tone} pulse={node.status === 'online'} />
            </button>
          )
        })}
      </div>

      {summary.gpus > 0 && (
        <div className="cluster-inventory__foot">
          <IconCpu size={14} /> {summary.gpus} accelerator{summary.gpus === 1 ? '' : 's'} across the fabric
        </div>
      )}
    </aside>
  )
}

export default NodeInventory
