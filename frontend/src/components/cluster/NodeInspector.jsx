import { useEffect, useState } from 'react'
import { DeviceIcon } from './DeviceIcon'
import { Button } from '../ui/Button'
import { Chip } from '../ui/Chip'
import { Field } from '../ui/Field'
import {
  CONNECTION_METHODS, CONNECTION_TYPES, NODE_ROLES, NODE_STATUS_BY, NODE_TYPES,
  nodeTypeLabel, roleLabel, statusTone,
} from '../../lib/clusterModel'
import {
  IconPlay, IconStop, IconRefresh, IconEdit, IconTrash, IconBolt, IconCheck, IconClose,
} from '../ui/icons'

function Row({ label, value }) {
  if (value == null || value === '') return null
  return (
    <div className="cluster-detail__row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function NodeDetail({ node, nodes, actions, onViewLogs }) {
  const [editing, setEditing] = useState(false)
  const [form, setForm] = useState(node)
  useEffect(() => { setForm(node); setEditing(false) }, [node.id]) // reset when selection changes
  const tone = statusTone(node.status)
  const statusLabel = NODE_STATUS_BY[node.status]?.label || 'Unknown'

  const set = (patch) => setForm((f) => ({ ...f, ...patch }))
  const toggleRole = (role) => set({ roles: form.roles.includes(role) ? form.roles.filter((r) => r !== role) : [...form.roles, role] })
  const save = () => {
    actions.updateNode(node.id, {
      display_name: form.display_name?.trim() || node.display_name,
      node_type: form.node_type,
      hostname: form.hostname?.trim() || '',
      ip_address: form.ip_address?.trim() || null,
      port: form.port ? Number(form.port) : null,
      connection_method: form.connection_method,
      roles: form.roles.length ? form.roles : ['worker'],
      os: form.os?.trim() || null,
      arch: form.arch?.trim() || null,
      cpu_cores: form.cpu_cores ? Number(form.cpu_cores) : null,
      ram_gb: form.ram_gb ? Number(form.ram_gb) : null,
      gpu: form.gpu?.trim() || null,
      vram: form.vram?.trim() || null,
      tags: typeof form.tags === 'string' ? form.tags.split(',').map((t) => t.trim()).filter(Boolean) : form.tags,
      model_paths: typeof form.model_paths === 'string' ? form.model_paths.split('\n').map((t) => t.trim()).filter(Boolean) : form.model_paths,
      worker_command: form.worker_command?.trim() || null,
    })
    setEditing(false)
  }

  if (editing) {
    return (
      <div className="cluster-detail">
        <div className="cluster-detail__head">
          <strong>Edit node</strong>
          <div className="cluster-detail__head-actions">
            <Button variant="ghost" size="sm" icon={<IconClose size={15} />} onClick={() => setEditing(false)}>Cancel</Button>
            <Button variant="primary" size="sm" icon={<IconCheck size={15} />} onClick={save}>Save</Button>
          </div>
        </div>
        <div className="cluster-form cluster-form--inspector">
          <Field label="Display name"><input value={form.display_name || ''} onChange={(e) => set({ display_name: e.target.value })} /></Field>
          <Field label="Device type">
            <select value={form.node_type} onChange={(e) => set({ node_type: e.target.value })}>
              {NODE_TYPES.map((t) => <option key={t.value} value={t.value}>{t.label}</option>)}
            </select>
          </Field>
          <div className="cluster-form__pair">
            <Field label="Hostname / IP"><input value={form.hostname || ''} onChange={(e) => set({ hostname: e.target.value })} /></Field>
            <Field label="Port"><input type="number" value={form.port || ''} onChange={(e) => set({ port: e.target.value })} /></Field>
          </div>
          <Field label="Connection method">
            <select value={form.connection_method} onChange={(e) => set({ connection_method: e.target.value })}>
              {CONNECTION_METHODS.map((m) => <option key={m.value} value={m.value}>{m.label}</option>)}
            </select>
          </Field>
          <div className="cluster-form__group">
            <span className="cluster-form__label">Roles</span>
            <div className="cluster-role-toggles">
              {NODE_ROLES.map((r) => (
                <button key={r.value} type="button" className={`cluster-role-toggle ${form.roles.includes(r.value) ? 'is-on' : ''}`} onClick={() => toggleRole(r.value)}>{r.label}</button>
              ))}
            </div>
          </div>
          <div className="cluster-form__pair">
            <Field label="CPU cores"><input type="number" value={form.cpu_cores || ''} onChange={(e) => set({ cpu_cores: e.target.value })} /></Field>
            <Field label="RAM (GB)"><input type="number" value={form.ram_gb || ''} onChange={(e) => set({ ram_gb: e.target.value })} /></Field>
          </div>
          <div className="cluster-form__pair">
            <Field label="OS"><input value={form.os || ''} onChange={(e) => set({ os: e.target.value })} /></Field>
            <Field label="Arch"><input value={form.arch || ''} onChange={(e) => set({ arch: e.target.value })} placeholder="arm64 / x86_64" /></Field>
          </div>
          <Field label="GPU / accelerator"><input value={form.gpu || ''} onChange={(e) => set({ gpu: e.target.value })} /></Field>
          <Field label="Tags (comma separated)"><input value={Array.isArray(form.tags) ? form.tags.join(', ') : form.tags || ''} onChange={(e) => set({ tags: e.target.value })} /></Field>
          <Field label="Worker command"><input value={form.worker_command || ''} onChange={(e) => set({ worker_command: e.target.value })} placeholder="camelid distribute-worker …" /></Field>
        </div>
      </div>
    )
  }

  return (
    <div className="cluster-detail">
      <div className="cluster-detail__title">
        <span className="cluster-detail__icon"><DeviceIcon type={node.node_type} size={22} /></span>
        <div>
          <strong>{node.display_name}</strong>
          <span>{nodeTypeLabel(node.node_type)}</span>
        </div>
        <Chip tone={tone} dot>{statusLabel}</Chip>
      </div>

      <div className="cluster-detail__rows">
        <Row label="Hostname / IP" value={`${node.hostname || node.ip_address || '—'}${node.port ? `:${node.port}` : ''}`} />
        <Row label="Connection" value={CONNECTION_METHODS.find((m) => m.value === node.connection_method)?.label} />
        <Row label="Roles" value={node.roles.map(roleLabel).join(', ') || '—'} />
        <Row label="OS" value={node.os} />
        <Row label="Architecture" value={node.arch} />
        <Row label="CPU" value={node.cpu_cores ? `${node.cpu_cores} cores` : null} />
        <Row label="RAM" value={node.ram_gb ? `${node.ram_gb} GB` : null} />
        <Row label="GPU" value={node.gpu} />
        <Row label="VRAM / unified" value={node.vram} />
        <Row label="Worker" value={node.worker_state ? node.worker_state : null} />
        <Row label="Last seen" value={node.last_seen ? new Date(node.last_seen).toLocaleString() : 'never'} />
        {node.model_paths?.length > 0 && <Row label="Model paths" value={node.model_paths.join(', ')} />}
        {node.tags?.length > 0 && <Row label="Tags" value={node.tags.join(', ')} />}
        {node.worker_command && <Row label="Worker command" value={node.worker_command} />}
      </div>

      <div className="cluster-detail__actions">
        <Button variant="tonal" size="sm" icon={<IconRefresh size={15} />} onClick={() => actions.testNode(node.id)}>Test connection</Button>
        {node.worker_state === 'running'
          ? <Button variant="ghost" size="sm" icon={<IconStop size={15} />} onClick={() => actions.stopWorker(node.id)}>Stop worker</Button>
          : <Button variant="ghost" size="sm" icon={<IconPlay size={15} />} onClick={() => actions.startWorker(node.id)}>Start worker</Button>}
        <Button variant="ghost" size="sm" icon={<IconBolt size={15} />} onClick={() => actions.restartWorker(node.id)}>Restart</Button>
        <Button variant="ghost" size="sm" icon={<IconEdit size={15} />} onClick={() => { setForm(node); setEditing(true) }}>Edit</Button>
        <Button variant="ghost" size="sm" onClick={onViewLogs}>View logs</Button>
        <Button variant="danger" size="sm" icon={<IconTrash size={15} />} onClick={() => actions.removeNode(node.id)}>Remove</Button>
      </div>
    </div>
  )
}

function ConnectionDetail({ connection, nodes, actions }) {
  const [editing, setEditing] = useState(false)
  const [form, setForm] = useState(connection)
  useEffect(() => { setForm(connection); setEditing(false) }, [connection.id])
  const source = nodes.find((n) => n.id === connection.source_node_id)
  const target = nodes.find((n) => n.id === connection.target_node_id)
  const tone = statusTone(connection.status)
  const set = (patch) => setForm((f) => ({ ...f, ...patch }))
  const save = () => {
    actions.updateConnection(connection.id, {
      connection_type: form.connection_type,
      label: form.label?.trim() || null,
      bandwidth_mbps: form.bandwidth_mbps ? Number(form.bandwidth_mbps) : null,
      notes: form.notes?.trim() || null,
    })
    setEditing(false)
  }

  return (
    <div className="cluster-detail">
      <div className="cluster-detail__title">
        <strong>Connection</strong>
        <Chip tone={tone} dot>{NODE_STATUS_BY[connection.status]?.label || 'Unknown'}</Chip>
      </div>
      {editing ? (
        <div className="cluster-form cluster-form--inspector">
          <Field label="Connection type">
            <select value={form.connection_type} onChange={(e) => set({ connection_type: e.target.value })}>
              {CONNECTION_TYPES.map((t) => <option key={t.value} value={t.value}>{t.label}</option>)}
            </select>
          </Field>
          <Field label="Label"><input value={form.label || ''} onChange={(e) => set({ label: e.target.value })} placeholder="Thunderbolt 4 / 10GbE / Wi‑Fi" /></Field>
          <Field label="Bandwidth (Mbps)"><input type="number" value={form.bandwidth_mbps || ''} onChange={(e) => set({ bandwidth_mbps: e.target.value })} /></Field>
          <Field label="Notes"><input value={form.notes || ''} onChange={(e) => set({ notes: e.target.value })} /></Field>
          <div className="cluster-detail__head-actions">
            <Button variant="ghost" size="sm" onClick={() => setEditing(false)}>Cancel</Button>
            <Button variant="primary" size="sm" onClick={save}>Save</Button>
          </div>
        </div>
      ) : (
        <>
          <div className="cluster-detail__rows">
            <Row label="Source" value={source?.display_name} />
            <Row label="Target" value={target?.display_name} />
            <Row label="Type" value={CONNECTION_TYPES.find((t) => t.value === connection.connection_type)?.label} />
            <Row label="Latency" value={connection.latency_ms != null ? `${connection.latency_ms} ms` : null} />
            <Row label="Bandwidth" value={connection.bandwidth_mbps != null ? `${connection.bandwidth_mbps} Mbps` : null} />
            <Row label="Notes" value={connection.notes} />
          </div>
          <div className="cluster-detail__actions">
            <Button variant="tonal" size="sm" icon={<IconRefresh size={15} />} onClick={() => actions.testConnection(connection.id)}>Test link</Button>
            <Button variant="ghost" size="sm" icon={<IconEdit size={15} />} onClick={() => { setForm(connection); setEditing(true) }}>Edit</Button>
            <Button variant="danger" size="sm" icon={<IconTrash size={15} />} onClick={() => actions.removeConnection(connection.id)}>Remove</Button>
          </div>
        </>
      )}
    </div>
  )
}

export function NodeInspector({ selectedNode, selectedConnection, nodes, actions, onViewLogs }) {
  return (
    <aside className="cluster-inspector">
      <div className="cluster-panel__head"><h3>Inspector</h3></div>
      <div className="cluster-inspector__body">
        {selectedNode && <NodeDetail node={selectedNode} nodes={nodes} actions={actions} onViewLogs={onViewLogs} />}
        {selectedConnection && <ConnectionDetail connection={selectedConnection} nodes={nodes} actions={actions} />}
        {!selectedNode && !selectedConnection && (
          <div className="cluster-inspector__empty">
            <p>Select a node or a link on the canvas to see its details and actions.</p>
            <p className="cluster-inspector__hint">Tip: drag the small handle on a node to draw a connection to another machine.</p>
          </div>
        )}
      </div>
    </aside>
  )
}

export default NodeInspector
