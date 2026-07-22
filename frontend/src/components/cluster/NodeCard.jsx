import { memo } from 'react'
import { DeviceIcon } from './DeviceIcon'
import { NODE_TYPE_BY, roleLabel, statusTone, NODE_STATUS_BY } from '../../lib/clusterModel'
import { IconCpu, IconBolt } from '../ui/icons'

export const NODE_W = 248
export const NODE_H = 152

function fmtRam(gb) {
  const n = Number(gb)
  if (!Number.isFinite(n) || n <= 0) return null
  return n >= 1024 ? `${(n / 1024).toFixed(1)} TB` : `${n} GB`
}

export const NodeCard = memo(function NodeCard({ node, selected, busyLabel, onSelect }) {
  const tone = statusTone(node.status)
  const statusLabel = NODE_STATUS_BY[node.status]?.label || 'Unknown'
  const address = node.hostname || node.ip_address || 'no address'
  const ram = fmtRam(node.ram_gb)
  const roles = node.roles || []

  return (
    <div
      className={`cluster-node is-${tone} ${selected ? 'is-selected' : ''} ${busyLabel ? 'is-busy' : ''}`}
      data-node-id={node.id}
      style={{ width: NODE_W, height: NODE_H }}
      role="button"
      tabIndex={0}
      aria-label={`${node.display_name}, ${statusLabel}`}
      onKeyDown={(e) => { if (e.key === 'Enter') onSelect?.() }}
    >
      <span className={`cluster-node__ring is-${tone}`} aria-hidden="true" />
      <div className="cluster-node__head">
        <span className="cluster-node__icon"><DeviceIcon type={node.node_type} size={20} /></span>
        <div className="cluster-node__title">
          <strong title={node.display_name}>{node.display_name}</strong>
          <span title={address}>{address}{node.port ? `:${node.port}` : ''}</span>
        </div>
        <span className={`cluster-node__status is-${tone}`} title={statusLabel} />
      </div>

      <div className="cluster-node__roles">
        {roles.slice(0, 2).map((r) => <span key={r} className="cluster-node__role">{roleLabel(r)}</span>)}
        {roles.length > 2 && <span className="cluster-node__role cluster-node__role--more">+{roles.length - 2}</span>}
        {!roles.length && <span className="cluster-node__role cluster-node__role--more">No role</span>}
      </div>

      {(node.os || node.arch)
        ? <div className="cluster-node__os" title={[node.os, node.arch].filter(Boolean).join(' · ')}>{[node.os, node.arch].filter(Boolean).join(' · ')}</div>
        : <div className="cluster-node__os cluster-node__os--muted">specs not detected</div>}

      <div className="cluster-node__meta">
        <span className="cluster-node__metric" title="CPU cores"><IconCpu size={13} />{node.cpu_cores ? `${node.cpu_cores} cores` : '—'}</span>
        <span className="cluster-node__metric" title="Memory">{ram || '—'}</span>
        {node.gpu && <span className="cluster-node__metric cluster-node__metric--gpu" title={node.gpu}><IconBolt size={13} />{node.gpu.replace(/^Apple\s+/, '')}</span>}
        {node.worker_state === 'running' && <span className="cluster-node__worker" title="Worker running">live</span>}
      </div>

      {busyLabel && <span className="cluster-node__busy">{busyLabel}</span>}

      <button
        type="button"
        className="cluster-node__handle"
        data-connect-handle={node.id}
        aria-label={`Draw a link from ${node.display_name}`}
        title="Drag to link to another node"
      />
    </div>
  )
})

export default NodeCard
export const NODE_TYPE_LABELS = NODE_TYPE_BY
