/* Cluster topology data model — mirrors the Rust schema (ClusterNode /
   ClusterConnection / ClusterTopology) as plain JS so it could map 1:1 to a
   future backend. Persisted client-side (localStorage), like conversations and
   memories. Local-only: machines sharing compute/memory/workers/model-serving. */

export const STORAGE_KEY = 'camelid.clusterTopology'

export const NODE_TYPES = [
  { value: 'mac', label: 'Mac', icon: 'mac', defaultMethod: 'ssh' },
  { value: 'windows', label: 'Windows PC', icon: 'windows', defaultMethod: 'winrm' },
  { value: 'linux', label: 'Linux Server', icon: 'linux', defaultMethod: 'ssh' },
  { value: 'raspberrypi', label: 'Raspberry Pi', icon: 'raspberrypi', defaultMethod: 'ssh' },
  { value: 'other', label: 'Other', icon: 'other', defaultMethod: 'manual' },
]

export const NODE_ROLES = [
  { value: 'coordinator', label: 'Coordinator', desc: 'Controls scheduling and cluster orchestration.' },
  { value: 'worker', label: 'Worker', desc: 'Runs inference work assigned by the coordinator.' },
  { value: 'model_host', label: 'Model Host', desc: 'Stores or serves model files locally.' },
  { value: 'storage', label: 'Storage Node', desc: 'Provides shared local storage paths.' },
  { value: 'gateway', label: 'Gateway', desc: 'Exposes API or UI access to the cluster.' },
  { value: 'observer', label: 'Observer', desc: 'Tracked in topology but not used for compute.' },
]

export const NODE_STATUS = [
  { value: 'online', label: 'Online', tone: 'ready' },
  { value: 'degraded', label: 'Degraded', tone: 'warn' },
  { value: 'offline', label: 'Offline', tone: 'error' },
  { value: 'unknown', label: 'Unknown', tone: 'neutral' },
]

export const CONNECTION_TYPES = [
  { value: 'thunderbolt', label: 'Thunderbolt 4' },
  { value: 'ethernet', label: '10GbE' },
  { value: 'lan', label: '1GbE' },
  { value: 'wifi', label: 'Wi‑Fi' },
  { value: 'manual', label: 'SSH' },
  { value: 'unknown', label: 'Link' },
]

export const CONNECTION_METHODS = [
  { value: 'ssh', label: 'SSH' },
  { value: 'winrm', label: 'WinRM' },
  { value: 'agent', label: 'Local agent' },
  { value: 'manual', label: 'Manual / offline node' },
]

const byValue = (list) => Object.fromEntries(list.map((item) => [item.value, item]))
export const NODE_TYPE_BY = byValue(NODE_TYPES)
export const NODE_ROLE_BY = byValue(NODE_ROLES)
export const NODE_STATUS_BY = byValue(NODE_STATUS)
export const CONNECTION_TYPE_BY = byValue(CONNECTION_TYPES)
export const CONNECTION_METHOD_BY = byValue(CONNECTION_METHODS)

export function statusTone(status) {
  return NODE_STATUS_BY[status]?.tone || 'neutral'
}

export function roleLabel(role) {
  return NODE_ROLE_BY[role]?.label || role
}

export function nodeTypeLabel(type) {
  return NODE_TYPE_BY[type]?.label || 'Other'
}

function makeId(prefix) {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return `${prefix}-${crypto.randomUUID().slice(0, 8)}`
  return `${prefix}-${Math.random().toString(16).slice(2, 10)}`
}

export function nowIso() {
  // Date.now is unavailable in some sandboxes; guard for SSR/tests.
  try { return new Date().toISOString() } catch { return '' }
}

export function createNode(partial = {}) {
  return {
    id: partial.id || makeId('node'),
    display_name: partial.display_name || 'New machine',
    node_type: partial.node_type || 'linux',
    hostname: partial.hostname || '',
    ip_address: partial.ip_address ?? null,
    port: partial.port ?? null,
    connection_method: partial.connection_method || NODE_TYPE_BY[partial.node_type || 'linux']?.defaultMethod || 'ssh',
    roles: partial.roles?.length ? [...partial.roles] : ['worker'],
    status: partial.status || 'unknown',
    os: partial.os ?? null,
    arch: partial.arch ?? null,
    cpu_cores: partial.cpu_cores ?? null,
    ram_gb: partial.ram_gb ?? null,
    gpu: partial.gpu ?? null,
    vram: partial.vram ?? null,
    model_paths: partial.model_paths ? [...partial.model_paths] : [],
    worker_command: partial.worker_command ?? null,
    worker_state: partial.worker_state ?? null,
    auth: partial.auth || { username: '', method: 'ssh-key', key_path: '', saved: false },
    tags: partial.tags ? [...partial.tags] : [],
    layout_x: Number.isFinite(partial.layout_x) ? partial.layout_x : 0,
    layout_y: Number.isFinite(partial.layout_y) ? partial.layout_y : 0,
    last_seen: partial.last_seen ?? null,
    notes: partial.notes ?? null,
  }
}

export function createConnection(partial = {}) {
  return {
    id: partial.id || makeId('link'),
    source_node_id: partial.source_node_id,
    target_node_id: partial.target_node_id,
    connection_type: partial.connection_type || 'lan',
    label: partial.label ?? null,
    latency_ms: partial.latency_ms ?? null,
    bandwidth_mbps: partial.bandwidth_mbps ?? null,
    status: partial.status || 'unknown',
    notes: partial.notes ?? null,
  }
}

export function emptyTopology() {
  return { nodes: [], connections: [], updated_at: nowIso() }
}

export function normalizeTopology(raw) {
  if (!raw || typeof raw !== 'object') return emptyTopology()
  const nodes = Array.isArray(raw.nodes) ? raw.nodes.map(createNode) : []
  const ids = new Set(nodes.map((n) => n.id))
  const connections = (Array.isArray(raw.connections) ? raw.connections : [])
    .map(createConnection)
    .filter((c) => ids.has(c.source_node_id) && ids.has(c.target_node_id))
  return { nodes, connections, updated_at: raw.updated_at || nowIso() }
}

export function loadTopology() {
  if (typeof window === 'undefined') return emptyTopology()
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    return raw ? normalizeTopology(JSON.parse(raw)) : emptyTopology()
  } catch {
    return emptyTopology()
  }
}

export function saveTopology(topology) {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...topology, updated_at: nowIso() }))
  } catch { /* quota / private mode — keep in-memory */ }
}

/* Deterministic, tidy layout: coordinators centered, model hosts / storage above,
   workers fanned below, gateways/observers to the sides. No physics dependency. */
export function autoLayout(nodes) {
  if (!nodes.length) return nodes
  const cx = 0
  const top = -260
  const bottom = 200
  const colGap = 260
  const primaryRole = (n) => {
    for (const r of ['coordinator', 'gateway', 'model_host', 'storage', 'worker', 'observer']) {
      if (n.roles?.includes(r)) return r
    }
    return 'worker'
  }
  const groups = { coordinator: [], gateway: [], model_host: [], storage: [], worker: [], observer: [] }
  nodes.forEach((n) => { (groups[primaryRole(n)] || groups.worker).push(n) })

  const place = (list, y, spread) => {
    const n = list.length
    list.forEach((node, i) => {
      const offset = (i - (n - 1) / 2) * spread
      node.layout_x = Math.round(cx + offset)
      node.layout_y = Math.round(y)
    })
  }
  place(groups.coordinator, -20, colGap * 1.2)
  place(groups.gateway, -20 - 200, colGap)
  place([...groups.model_host, ...groups.storage], top, colGap)
  place(groups.worker, bottom, colGap * 0.95)
  place(groups.observer, bottom + 200, colGap)
  // gateways off to the right if there are coordinators
  if (groups.coordinator.length && groups.gateway.length) {
    groups.gateway.forEach((node, i) => { node.layout_x = 360; node.layout_y = -20 + i * 150 })
  }
  return nodes
}

/* Validation — surfaced in the bottom drawer. Returns [{level,message}]. */
export function validateTopology(topology) {
  const issues = []
  const { nodes, connections } = topology
  if (!nodes.length) { issues.push({ level: 'info', message: 'No nodes in the cluster yet. Add a server to begin.' }); return issues }

  const coordinators = nodes.filter((n) => n.roles.includes('coordinator'))
  if (coordinators.length === 0) issues.push({ level: 'warn', message: 'No Coordinator node — one machine should orchestrate scheduling.' })
  if (coordinators.length > 1) issues.push({ level: 'warn', message: `${coordinators.length} Coordinators — multiple coordinators can conflict; usually one is expected.` })

  const workers = nodes.filter((n) => n.roles.includes('worker'))
  if (workers.length === 0) issues.push({ level: 'warn', message: 'No Worker nodes — nothing will run inference work.' })

  if (!nodes.some((n) => n.roles.includes('model_host'))) issues.push({ level: 'info', message: 'No Model Host — at least one node should serve model files.' })

  // duplicate hostnames / ips
  const seenHost = new Map()
  nodes.forEach((n) => {
    const key = (n.hostname || n.ip_address || '').trim().toLowerCase()
    if (!key) { issues.push({ level: 'warn', message: `“${n.display_name}” has no hostname or IP.` }); return }
    if (seenHost.has(key)) issues.push({ level: 'error', message: `Duplicate address “${key}” on “${n.display_name}” and “${seenHost.get(key)}”.` })
    else seenHost.set(key, n.display_name)
  })

  // orphan nodes (no connections), excluding observers
  const connected = new Set()
  connections.forEach((c) => { connected.add(c.source_node_id); connected.add(c.target_node_id) })
  nodes.filter((n) => !n.roles.includes('observer') && !connected.has(n.id)).forEach((n) => {
    issues.push({ level: 'info', message: `“${n.display_name}” is not linked to any other node.` })
  })

  nodes.filter((n) => n.status === 'offline').forEach((n) => issues.push({ level: 'warn', message: `“${n.display_name}” is offline.` }))
  nodes.filter((n) => n.status === 'degraded').forEach((n) => issues.push({ level: 'warn', message: `“${n.display_name}” is degraded.` }))

  if (!issues.some((i) => i.level !== 'info')) issues.unshift({ level: 'ok', message: `Cluster looks healthy — ${nodes.length} node${nodes.length === 1 ? '' : 's'}, ${connections.length} link${connections.length === 1 ? '' : 's'}.` })
  return issues
}

/* A representative example fabric so users can see a populated topology
   immediately (offered from the empty state). All local machines. */
export function sampleTopology() {
  const n = (p) => createNode(p)
  const nodes = [
    n({ id: 'node-studio', display_name: 'Mac Studio', node_type: 'mac', hostname: 'studio.local', ip_address: '192.0.2.10', port: 22, connection_method: 'ssh', roles: ['coordinator', 'model_host'], status: 'online', os: 'macOS 15', arch: 'arm64', cpu_cores: 24, ram_gb: 192, gpu: 'Apple M2 Ultra (76-core)', vram: '192 GB unified', model_paths: ['/Volumes/Untitled/models'] }),
    n({ id: 'node-mbp', display_name: 'MacBook Pro', node_type: 'mac', hostname: 'mbp.local', ip_address: '192.0.2.11', roles: ['worker'], status: 'online', os: 'macOS 15', arch: 'arm64', cpu_cores: 14, ram_gb: 64, gpu: 'Apple M3 Max', worker_state: 'running' }),
    n({ id: 'node-gpu', display_name: 'Linux GPU box', node_type: 'linux', hostname: 'gpu01', ip_address: '192.0.2.20', roles: ['worker', 'model_host'], status: 'online', os: 'Ubuntu 24.04', arch: 'x86_64', cpu_cores: 32, ram_gb: 128, gpu: 'NVIDIA RTX 4090', vram: '24 GB', worker_state: 'running' }),
    n({ id: 'node-pi', display_name: 'Raspberry Pi 5', node_type: 'raspberrypi', hostname: 'pi5.local', ip_address: '192.0.2.30', roles: ['worker'], status: 'degraded', os: 'Raspberry Pi OS', arch: 'arm64', cpu_cores: 4, ram_gb: 8 }),
    n({ id: 'node-nas', display_name: 'NAS', node_type: 'linux', hostname: 'nas.local', ip_address: '192.0.2.40', roles: ['storage'], status: 'online', os: 'Linux', arch: 'x86_64', cpu_cores: 4, ram_gb: 16, model_paths: ['/mnt/models'] }),
    n({ id: 'node-win', display_name: 'Windows PC', node_type: 'windows', hostname: 'win-rig', ip_address: '192.0.2.50', port: 5985, connection_method: 'winrm', roles: ['observer'], status: 'unknown', os: 'Windows 11', arch: 'x86_64', cpu_cores: 16, ram_gb: 32 }),
  ]
  const connections = [
    createConnection({ source_node_id: 'node-studio', target_node_id: 'node-mbp', connection_type: 'thunderbolt', label: 'Thunderbolt 4', latency_ms: 0.4, status: 'online' }),
    createConnection({ source_node_id: 'node-studio', target_node_id: 'node-gpu', connection_type: 'ethernet', label: '10GbE', latency_ms: 0.8, status: 'online' }),
    createConnection({ source_node_id: 'node-studio', target_node_id: 'node-pi', connection_type: 'wifi', label: 'Wi‑Fi', latency_ms: 9.2, status: 'degraded' }),
    createConnection({ source_node_id: 'node-studio', target_node_id: 'node-nas', connection_type: 'lan', label: '1GbE', latency_ms: 1.1, status: 'online' }),
    createConnection({ source_node_id: 'node-gpu', target_node_id: 'node-nas', connection_type: 'lan', label: '1GbE', status: 'online' }),
  ]
  return { nodes: autoLayout(nodes), connections, updated_at: nowIso() }
}

/* Merge an imported topology fragment (from discovery / a shared config) into the
   current one. Dedups nodes by id or hostname/IP so re-importing or importing a
   machine you already have won't create duplicates; remaps connection endpoints. */
export function mergeImport(topology, imp) {
  if (!imp || !Array.isArray(imp.nodes)) return topology
  const nodes = topology.nodes.map((n) => ({ ...n }))
  const hostKey = (n) => String(n.hostname || n.ip_address || '').trim().toLowerCase()
  const byHost = new Map(nodes.filter((n) => hostKey(n)).map((n) => [hostKey(n), n.id]))
  const byId = new Set(nodes.map((n) => n.id))
  const idMap = {}
  imp.nodes.forEach((impNode) => {
    const hk = hostKey(impNode)
    const existingId = (hk && byHost.has(hk)) ? byHost.get(hk) : (byId.has(impNode.id) ? impNode.id : null)
    if (existingId) {
      idMap[impNode.id] = existingId
      // Apply provided fields (e.g. a corrected address/status/specs) without clobbering with empties.
      const target = nodes.find((n) => n.id === existingId)
      if (target) {
        Object.entries(impNode).forEach(([k, v]) => {
          if (k === 'id' || v === undefined || v === null || v === '') return
          target[k] = v
        })
        const tk = hostKey(target)
        if (tk) byHost.set(tk, target.id)
      }
      return
    }
    const node = createNode(impNode)
    nodes.push(node)
    byId.add(node.id)
    if (hk) byHost.set(hk, node.id)
    idMap[impNode.id] = node.id
  })
  const connections = topology.connections.map((c) => ({ ...c }))
  ;(imp.connections || []).forEach((impConn) => {
    const s = idMap[impConn.source_node_id]
    const t = idMap[impConn.target_node_id]
    if (!s || !t || s === t) return
    const existing = connections.find((c) => (c.source_node_id === s && c.target_node_id === t) || (c.source_node_id === t && c.target_node_id === s))
    if (existing) {
      // Apply only the fields the import specifies (e.g. correcting a link's
      // connection_type/bandwidth), keeping the existing id + endpoints intact.
      const { id, source_node_id, target_node_id, ...patch } = impConn
      Object.assign(existing, patch)
      return
    }
    connections.push(createConnection({ ...impConn, source_node_id: s, target_node_id: t }))
  })
  // Optional removals: drop connections (by id or endpoint pair) and nodes (by id).
  const resolve = (id) => idMap[id] || id
  let outConnections = connections
  if (Array.isArray(imp.remove_connections)) {
    outConnections = outConnections.filter((c) => !imp.remove_connections.some((r) => {
      if (r.id) return c.id === r.id
      const s = resolve(r.source_node_id)
      const t = resolve(r.target_node_id)
      return (c.source_node_id === s && c.target_node_id === t) || (c.source_node_id === t && c.target_node_id === s)
    }))
  }
  let outNodes = nodes
  if (Array.isArray(imp.remove_nodes)) {
    const rm = new Set(imp.remove_nodes.map(resolve))
    outNodes = outNodes.filter((n) => !rm.has(n.id))
    outConnections = outConnections.filter((c) => !rm.has(c.source_node_id) && !rm.has(c.target_node_id))
  }
  return { nodes: outNodes, connections: outConnections, updated_at: nowIso() }
}

// Map live camelid telemetry specs onto a node patch, normalizing for display.
// (On Apple Silicon the chip is both CPU and GPU, so cpu_model populates `gpu`,
// which the node card renders as the chip line. RAM isn't reported, so it's left as-is.)
export function specsToNodePatch(specs) {
  if (!specs) return {}
  const osMap = { macos: 'macOS', windows: 'Windows', linux: 'Linux' }
  const archMap = { aarch64: 'arm64', arm64: 'arm64', x86_64: 'x86_64', amd64: 'x86_64' }
  const patch = {}
  if (specs.os) patch.os = osMap[String(specs.os).toLowerCase()] || specs.os
  if (specs.arch) patch.arch = archMap[String(specs.arch).toLowerCase()] || specs.arch
  if (specs.cpu_cores) patch.cpu_cores = specs.cpu_cores
  if (specs.cpu_model) patch.gpu = specs.cpu_model
  return patch
}

export function summarizeCluster(topology) {
  const { nodes } = topology
  const totalCores = nodes.reduce((sum, n) => sum + (Number(n.cpu_cores) || 0), 0)
  const totalRam = nodes.reduce((sum, n) => sum + (Number(n.ram_gb) || 0), 0)
  const online = nodes.filter((n) => n.status === 'online').length
  const gpus = nodes.filter((n) => n.gpu).length
  return {
    nodeCount: nodes.length,
    online,
    totalCores,
    totalRam,
    gpus,
    roles: NODE_ROLES.map((r) => ({ ...r, count: nodes.filter((n) => n.roles.includes(r.value)).length })),
  }
}
