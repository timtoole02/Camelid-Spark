import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  autoLayout, createConnection, createNode, loadTopology, mergeImport, nowIso, sampleTopology, saveTopology,
  specsToNodePatch, summarizeCluster, validateTopology,
} from '../lib/clusterModel'
import { discoverDevices, fetchNodeTelemetry, probeNode } from '../lib/devCluster'

const DEFAULT_PORT = { ssh: 22, winrm: 5985, agent: 8181, manual: null }

let eventSeq = 0
function makeEvent(level, message) {
  eventSeq += 1
  return { id: `evt-${eventSeq}`, time: nowIso(), level, message }
}

export function useClusterTopology({ showNotice } = {}) {
  const [topology, setTopology] = useState(loadTopology)
  const [selection, setSelection] = useState({ kind: null, id: null }) // 'node' | 'connection'
  const [events, setEvents] = useState(() => [makeEvent('info', 'Cluster topology loaded.')])
  const [busyIds, setBusyIds] = useState({}) // id -> action label
  const saveTimer = useRef(null)
  const topoRef = useRef(topology)
  topoRef.current = topology // always-latest snapshot for async actions/effects

  // Debounced persistence (covers edits + node drags uniformly).
  useEffect(() => {
    if (saveTimer.current) window.clearTimeout(saveTimer.current)
    saveTimer.current = window.setTimeout(() => saveTopology(topology), 450)
    return () => window.clearTimeout(saveTimer.current)
  }, [topology])

  const pushEvent = useCallback((level, message) => {
    setEvents((prev) => [makeEvent(level, message), ...prev].slice(0, 200))
  }, [])

  // One-time merge of a discovered/shared topology fragment (public/cluster-import.json),
  // tracked by import_id so it applies once per browser and never duplicates nodes.
  useEffect(() => {
    let cancelled = false
    fetch('/cluster-import.json', { cache: 'no-store' })
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (cancelled || !data) return
        // Accept a single import object or an array of them (each tracked by import_id).
        const imports = Array.isArray(data) ? data : (Array.isArray(data.imports) ? data.imports : [data])
        let done = []
        try { done = JSON.parse(window.localStorage.getItem('camelid.clusterImports') || '[]') } catch { /* noop */ }
        const fresh = imports.filter((imp) => imp && imp.import_id && !done.includes(imp.import_id))
        if (!fresh.length) return
        setTopology((prev) => fresh.reduce((acc, imp) => mergeImport(acc, imp), prev))
        try { window.localStorage.setItem('camelid.clusterImports', JSON.stringify([...done, ...fresh.map((i) => i.import_id)])) } catch { /* noop */ }
        const added = fresh.reduce((n, imp) => n + (imp.nodes?.length || 0), 0)
        pushEvent('ok', `Applied ${fresh.length} cluster import(s) — ${added} node entr${added === 1 ? 'y' : 'ies'}.`)
      })
      .catch(() => {})
    return () => { cancelled = true }
  }, [pushEvent])

  const setNodeBusy = useCallback((id, label) => {
    setBusyIds((prev) => {
      const next = { ...prev }
      if (label) next[id] = label
      else delete next[id]
      return next
    })
  }, [])

  const nodes = topology.nodes
  const connections = topology.connections
  const selectedNode = useMemo(() => (selection.kind === 'node' ? nodes.find((n) => n.id === selection.id) : null), [selection, nodes])
  const selectedConnection = useMemo(() => (selection.kind === 'connection' ? connections.find((c) => c.id === selection.id) : null), [selection, connections])
  const summary = useMemo(() => summarizeCluster(topology), [topology])
  const issues = useMemo(() => validateTopology(topology), [topology])

  const select = useCallback((kind, id) => setSelection(id ? { kind, id } : { kind: null, id: null }), [])

  // ---- Node CRUD ----
  const addNode = useCallback((partial) => {
    const node = createNode(partial)
    setTopology((prev) => {
      // place near center-ish if no layout supplied
      if (!partial.layout_x && !partial.layout_y) {
        const n = prev.nodes.length
        node.layout_x = (n % 4) * 240 - 360 + (Math.floor(n / 4) % 2) * 120
        node.layout_y = Math.floor(n / 4) * 200 - 120
      }
      return { ...prev, nodes: [...prev.nodes, node], updated_at: nowIso() }
    })
    pushEvent('ok', `Added “${node.display_name}” (${node.node_type}).`)
    setSelection({ kind: 'node', id: node.id })
    return node
  }, [pushEvent])

  const updateNode = useCallback((id, patch) => {
    setTopology((prev) => ({ ...prev, nodes: prev.nodes.map((n) => (n.id === id ? { ...n, ...patch } : n)), updated_at: nowIso() }))
  }, [])

  const moveNode = useCallback((id, x, y) => {
    setTopology((prev) => ({ ...prev, nodes: prev.nodes.map((n) => (n.id === id ? { ...n, layout_x: x, layout_y: y } : n)) }))
  }, [])

  const removeNode = useCallback((id) => {
    setTopology((prev) => ({
      ...prev,
      nodes: prev.nodes.filter((n) => n.id !== id),
      connections: prev.connections.filter((c) => c.source_node_id !== id && c.target_node_id !== id),
      updated_at: nowIso(),
    }))
    setSelection((s) => (s.id === id ? { kind: null, id: null } : s))
    pushEvent('warn', 'Removed a node and its links.')
  }, [pushEvent])

  // ---- Connection CRUD ----
  const addConnection = useCallback((partial) => {
    if (!partial.source_node_id || !partial.target_node_id || partial.source_node_id === partial.target_node_id) return null
    const exists = topology.connections.some(
      (c) => (c.source_node_id === partial.source_node_id && c.target_node_id === partial.target_node_id)
        || (c.source_node_id === partial.target_node_id && c.target_node_id === partial.source_node_id),
    )
    if (exists) return null
    const link = createConnection(partial)
    setTopology((prev) => ({ ...prev, connections: [...prev.connections, link], updated_at: nowIso() }))
    pushEvent('ok', 'Linked two nodes.')
    setSelection({ kind: 'connection', id: link.id })
    return link
  }, [topology.connections, pushEvent])

  const updateConnection = useCallback((id, patch) => {
    setTopology((prev) => ({ ...prev, connections: prev.connections.map((c) => (c.id === id ? { ...c, ...patch } : c)), updated_at: nowIso() }))
  }, [])

  const removeConnection = useCallback((id) => {
    setTopology((prev) => ({ ...prev, connections: prev.connections.filter((c) => c.id !== id), updated_at: nowIso() }))
    setSelection((s) => (s.id === id ? { kind: null, id: null } : s))
    pushEvent('info', 'Removed a link.')
  }, [pushEvent])

  // ---- Layout ----
  const applyAutoLayout = useCallback(() => {
    setTopology((prev) => ({ ...prev, nodes: autoLayout(prev.nodes.map((n) => ({ ...n }))), updated_at: nowIso() }))
    pushEvent('info', 'Re-arranged nodes with auto-layout.')
  }, [pushEvent])

  const resetTopology = useCallback(() => {
    setTopology({ nodes: [], connections: [], updated_at: nowIso() })
    setSelection({ kind: null, id: null })
    pushEvent('warn', 'Cleared the topology.')
  }, [pushEvent])

  const loadSample = useCallback(() => {
    setTopology(sampleTopology())
    setSelection({ kind: null, id: null })
    pushEvent('ok', 'Loaded a sample compute fabric.')
  }, [pushEvent])

  const exportTopology = useCallback(() => {
    try {
      const blob = new Blob([`${JSON.stringify(topology, null, 2)}\n`], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `camelid-cluster-topology.json`
      a.click()
      URL.revokeObjectURL(url)
      pushEvent('ok', 'Exported topology JSON.')
    } catch { /* best effort */ }
  }, [topology, pushEvent])

  const save = useCallback(() => {
    saveTopology(topology)
    pushEvent('ok', 'Topology saved.')
    showNotice?.('Cluster topology saved.', 'success')
  }, [topology, pushEvent, showNotice])

  // ---- Real-ish actions via the dev hook ----
  const portFor = (node) => node.port || DEFAULT_PORT[node.connection_method] || 22
  // camelid's HTTP API port (agent nodes may set it explicitly; else the default 8181).
  const camelidPortFor = (node) => (node.connection_method === 'agent' && node.port ? node.port : DEFAULT_PORT.agent)

  // Pull live camelid telemetry straight from the node's API (real specs + online status).
  // Tries hostname then IP; updates the node in place. Returns true if the node answered.
  const syncNode = useCallback(async (id, { quiet = false } = {}) => {
    const node = topoRef.current.nodes.find((n) => n.id === id)
    if (!node) return false
    const hosts = [node.hostname, node.ip_address].filter(Boolean)
    if (!hosts.length) return false
    // 1) Live camelid telemetry (real specs) on the camelid HTTP port.
    let tel = { online: false }
    for (const host of hosts) {
      tel = await fetchNodeTelemetry({ host, port: camelidPortFor(node) })
      if (tel.online) break
    }
    if (tel.online) {
      updateNode(id, { status: 'online', last_seen: nowIso(), ...specsToNodePatch(tel.specs) })
      if (!quiet) {
        const s = tel.specs || {}
        pushEvent('ok', `${node.display_name} online via camelid — ${s.cpu_model || tel.engine}${s.cpu_cores ? ` · ${s.cpu_cores} cores` : ''}.`)
      }
      return true
    }
    // 2) Non-HTTP services (e.g. a NanoCamelid pipeline stage on :9100): a TCP probe
    //    on the service port confirms it's listening. Needs the dev hook (npm run dev).
    //    Try each address (hostname then IP) so a cold .local/mDNS miss falls back to the IP.
    let probe = { available: false }
    for (const host of hosts) {
      probe = await probeNode({ host, port: portFor(node) })
      if (probe.available && probe.reachable) break
    }
    if (probe.available && probe.reachable) {
      updateNode(id, { status: 'online', last_seen: nowIso() })
      if (!quiet) pushEvent('ok', `${node.display_name} reachable on :${portFor(node)} — ${Math.round(probe.latencyMs)}ms.`)
      return true
    }
    return false
  }, [updateNode, pushEvent])

  // On load, quietly detect which nodes are actually running camelid (real online + specs).
  // Guard lives inside the timer (not at effect entry) so StrictMode's dev remount,
  // which clears the first timer, reschedules instead of silently skipping.
  const autoSynced = useRef(false)
  useEffect(() => {
    const timer = window.setTimeout(async () => {
      if (autoSynced.current) return
      autoSynced.current = true
      const addressed = topoRef.current.nodes.filter((n) => n.hostname || n.ip_address)
      if (!addressed.length) return
      pushEvent('info', 'Auto-detecting live camelid nodes…')
      const results = await Promise.allSettled(addressed.map((n) => syncNode(n.id, { quiet: true })))
      const online = results.filter((r) => r.status === 'fulfilled' && r.value === true).length
      pushEvent(online ? 'ok' : 'info', `Live scan: ${online} of ${addressed.length} node(s) reporting camelid telemetry.`)
    }, 600) // let the one-time import merge settle first
    return () => window.clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const testNode = useCallback(async (id) => {
    const node = topology.nodes.find((n) => n.id === id)
    if (!node) return
    const host = node.hostname || node.ip_address
    if (!host) { showNotice?.('Add a hostname or IP first.', 'error'); return }
    setNodeBusy(id, 'Testing…')
    pushEvent('info', `Testing ${node.display_name} (${host})…`)
    // Prefer live camelid telemetry — confirms online AND reads real hardware specs.
    const online = await syncNode(id)
    if (online) { setNodeBusy(id, null); return }
    // Otherwise fall back to a raw TCP reachability probe via the dev hook.
    const result = await probeNode({ host, port: portFor(node) })
    setNodeBusy(id, null)
    if (!result.available) {
      updateNode(id, { status: 'unknown', last_seen: nowIso() })
      pushEvent('warn', `${node.display_name}: camelid API not detected on :${camelidPortFor(node)}; TCP probe needs the dev server (npm run dev). Marked Unknown.`)
      return
    }
    if (result.reachable) {
      updateNode(id, { status: 'online', last_seen: nowIso() })
      pushEvent('ok', `${node.display_name} reachable on :${portFor(node)} — ${Math.round(result.latencyMs)}ms (camelid API not detected, so specs are unavailable).`)
    } else {
      updateNode(id, { status: 'offline', last_seen: nowIso() })
      pushEvent('error', `${node.display_name} offline — no camelid on :${camelidPortFor(node)} and :${portFor(node)} unreachable.`)
    }
  }, [topology.nodes, updateNode, pushEvent, setNodeBusy, showNotice, syncNode])

  const testConnection = useCallback(async (id) => {
    const link = topology.connections.find((c) => c.id === id)
    if (!link) return
    const target = topology.nodes.find((n) => n.id === link.target_node_id)
    const host = target?.hostname || target?.ip_address
    if (!host) { showNotice?.('Target node has no address to test.', 'error'); return }
    pushEvent('info', `Testing link to ${target.display_name}…`)
    const result = await probeNode({ host, port: portFor(target) })
    if (result.available && result.reachable) {
      updateConnection(id, { status: 'online', latency_ms: Math.round(result.latencyMs * 10) / 10 })
      pushEvent('ok', `Link to ${target.display_name}: ${Math.round(result.latencyMs)}ms.`)
    } else {
      updateConnection(id, { status: result.available ? 'offline' : 'unknown' })
      pushEvent(result.available ? 'error' : 'warn', `Link to ${target.display_name}: ${result.available ? 'unreachable' : 'needs the local agent'}.`)
    }
  }, [topology.connections, topology.nodes, updateConnection, pushEvent, showNotice])

  const validateCluster = useCallback(async () => {
    pushEvent('info', 'Validating cluster…')
    const reachable = topology.nodes.filter((n) => n.hostname || n.ip_address)
    await Promise.all(reachable.map((n) => testNode(n.id)))
    // Static checks (live, reactively recomputed) are surfaced into the events log too.
    validateTopology(topology).filter((i) => i.level !== 'ok').forEach((i) => pushEvent(i.level, i.message))
    pushEvent('ok', 'Validation complete.')
    showNotice?.('Cluster validation complete — see the events drawer.', 'info')
  }, [topology, testNode, pushEvent, showNotice])

  const setWorkerState = useCallback((id, state, label) => {
    const node = topology.nodes.find((n) => n.id === id)
    updateNode(id, { worker_state: state })
    pushEvent(state === 'running' ? 'ok' : 'info', `${label} on ${node?.display_name || 'node'} (${state === 'running' ? 'worker marked running' : 'worker marked stopped'}). Live control needs the local agent.`)
  }, [topology.nodes, updateNode, pushEvent])

  const startWorker = useCallback((id) => setWorkerState(id, 'running', 'Start worker'), [setWorkerState])
  const stopWorker = useCallback((id) => setWorkerState(id, 'stopped', 'Stop worker'), [setWorkerState])
  const restartWorker = useCallback((id) => setWorkerState(id, 'running', 'Restart worker'), [setWorkerState])

  const discover = useCallback(async () => {
    pushEvent('info', 'Discovering local devices…')
    const result = await discoverDevices()
    if (!result.available) {
      pushEvent('warn', 'Discovery needs the local dev server (npm run dev). You can still add machines manually.')
    } else {
      pushEvent('ok', `Discovery found ${result.devices.length} candidate device${result.devices.length === 1 ? '' : 's'}.`)
    }
    return result
  }, [pushEvent])

  return {
    topology, nodes, connections, summary, issues, events,
    selection, selectedNode, selectedConnection, select,
    busyIds,
    addNode, updateNode, moveNode, removeNode,
    addConnection, updateConnection, removeConnection,
    applyAutoLayout, resetTopology, loadSample, exportTopology, save,
    testNode, testConnection, validateCluster, syncNode,
    startWorker, stopWorker, restartWorker,
    discover, pushEvent,
  }
}
