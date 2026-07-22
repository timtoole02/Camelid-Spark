import { forwardRef, useCallback, useEffect, useImperativeHandle, useLayoutEffect, useRef, useState } from 'react'
import { NodeCard, NODE_W, NODE_H } from './NodeCard'
import { CamelidMark } from '../ui/CamelidMark'
import { Button } from '../ui/Button'
import { CONNECTION_TYPE_BY, statusTone } from '../../lib/clusterModel'
import { IconFit, IconZoomIn, IconZoomOut, IconGrid, IconNetwork, IconPlus } from '../ui/icons'

const MIN_SCALE = 0.3
const MAX_SCALE = 2.2
const GRID = 40

function linkPath(ax, ay, bx, by) {
  const dx = bx - ax
  const dy = by - ay
  if (Math.abs(dy) >= Math.abs(dx)) {
    const off = Math.max(40, Math.abs(dy) * 0.4)
    return `M ${ax} ${ay} C ${ax} ${ay + Math.sign(dy) * off} ${bx} ${by - Math.sign(dy) * off} ${bx} ${by}`
  }
  const off = Math.max(40, Math.abs(dx) * 0.4)
  return `M ${ax} ${ay} C ${ax + Math.sign(dx) * off} ${ay} ${bx - Math.sign(dx) * off} ${by} ${bx} ${by}`
}

// Point on a node card's edge in the direction of (tx,ty), so links visibly
// emanate from the card border (not from hidden centers). +margin pushes just outside.
const HALF_W = NODE_W / 2 + 4
const HALF_H = NODE_H / 2 + 4
function rectEdge(cx, cy, tx, ty) {
  const dx = tx - cx
  const dy = ty - cy
  if (!dx && !dy) return [cx, cy]
  const t = Math.min(HALF_W / (Math.abs(dx) || 1e-6), HALF_H / (Math.abs(dy) || 1e-6))
  return [cx + dx * t, cy + dy * t]
}

export const TopologyCanvas = forwardRef(function TopologyCanvas({
  nodes, connections, selection, onSelect, onMoveNode, onAddConnection, onAutoLayout, onAddServer, onLoadSample, busyIds = {},
}, ref) {
  const containerRef = useRef(null)
  const [size, setSize] = useState({ w: 1200, h: 700 })
  const [view, setView] = useState({ x: 600, y: 350, scale: 1 })
  const [snap, setSnap] = useState(true)
  const [showGrid, setShowGrid] = useState(true)
  const [pending, setPending] = useState(null) // rubber-band link
  const [hoverTarget, setHoverTarget] = useState(null)
  const drag = useRef(null)
  const viewRef = useRef(view)
  viewRef.current = view
  const interacted = useRef(false)

  const nodeById = useCallback((id) => nodes.find((n) => n.id === id), [nodes])

  // measure container
  useLayoutEffect(() => {
    const el = containerRef.current
    if (!el) return undefined
    const update = () => setSize({ w: el.clientWidth, h: el.clientHeight })
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const screenToWorld = useCallback((sx, sy) => {
    const v = viewRef.current
    return { x: (sx - v.x) / v.scale, y: (sy - v.y) / v.scale }
  }, [])

  const fit = useCallback(() => {
    if (!nodes.length) { setView({ x: size.w / 2, y: size.h / 2, scale: 1 }); return }
    const xs = nodes.map((n) => n.layout_x)
    const ys = nodes.map((n) => n.layout_y)
    const pad = 140
    const minX = Math.min(...xs) - pad, maxX = Math.max(...xs) + pad
    const minY = Math.min(...ys) - pad, maxY = Math.max(...ys) + pad
    const bw = Math.max(1, maxX - minX), bh = Math.max(1, maxY - minY)
    const scale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, Math.min(size.w / bw, size.h / bh)))
    const cxWorld = (minX + maxX) / 2
    const cyWorld = (minY + maxY) / 2
    setView({ x: size.w / 2 - cxWorld * scale, y: size.h / 2 - cyWorld * scale, scale })
  }, [nodes, size])

  useImperativeHandle(ref, () => ({
    fit,
    zoomBy: (factor) => setView((v) => zoomAround(v, factor, size.w / 2, size.h / 2)),
  }), [fit, size])

  // Auto-fit until the user first interacts — re-runs as the canvas size settles
  // (so the view stays centered even before the layout's final width is measured).
  useEffect(() => {
    if (!interacted.current && nodes.length && size.w > 1) fit()
  }, [nodes.length, size.w, size.h, fit])

  function zoomAround(v, factor, cx, cy) {
    const scale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, v.scale * factor))
    const wx = (cx - v.x) / v.scale
    const wy = (cy - v.y) / v.scale
    return { scale, x: cx - wx * scale, y: cy - wy * scale }
  }

  const onWheel = useCallback((e) => {
    e.preventDefault()
    interacted.current = true
    const rect = containerRef.current.getBoundingClientRect()
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12
    setView((v) => zoomAround(v, factor, e.clientX - rect.left, e.clientY - rect.top))
  }, [])

  const onPointerDown = useCallback((e) => {
    interacted.current = true
    const rect = containerRef.current.getBoundingClientRect()
    const sx = e.clientX - rect.left
    const sy = e.clientY - rect.top
    const handle = e.target.closest('[data-connect-handle]')
    const nodeEl = e.target.closest('[data-node-id]')
    const linkEl = e.target.closest('[data-link-id]')
    containerRef.current.setPointerCapture(e.pointerId)

    if (handle) {
      const src = nodeById(handle.getAttribute('data-connect-handle'))
      if (src) {
        const w = screenToWorld(sx, sy)
        setPending({ sourceId: src.id, fromX: src.layout_x, fromY: src.layout_y, toX: w.x, toY: w.y })
        drag.current = { mode: 'connect' }
      }
      return
    }
    if (nodeEl) {
      const id = nodeEl.getAttribute('data-node-id')
      const node = nodeById(id)
      onSelect('node', id)
      drag.current = { mode: 'node', id, startSX: sx, startSY: sy, startX: node.layout_x, startY: node.layout_y, moved: false }
      return
    }
    if (linkEl) {
      onSelect('connection', linkEl.getAttribute('data-link-id'))
      drag.current = { mode: 'link-click' }
      return
    }
    drag.current = { mode: 'pan', startSX: sx, startSY: sy, startVX: view.x, startVY: view.y, moved: false }
  }, [nodeById, onSelect, screenToWorld, view.x, view.y])

  const onPointerMove = useCallback((e) => {
    const d = drag.current
    if (!d) return
    const rect = containerRef.current.getBoundingClientRect()
    const sx = e.clientX - rect.left
    const sy = e.clientY - rect.top
    if (d.mode === 'pan') {
      d.moved = true
      setView((v) => ({ ...v, x: d.startVX + (sx - d.startSX), y: d.startVY + (sy - d.startSY) }))
    } else if (d.mode === 'node') {
      d.moved = true
      const v = viewRef.current
      const nx = d.startX + (sx - d.startSX) / v.scale
      const ny = d.startY + (sy - d.startSY) / v.scale
      onMoveNode(d.id, Math.round(nx), Math.round(ny))
    } else if (d.mode === 'connect') {
      const w = screenToWorld(sx, sy)
      setPending((p) => (p ? { ...p, toX: w.x, toY: w.y } : p))
      const overEl = document.elementFromPoint(e.clientX, e.clientY)?.closest('[data-node-id]')
      const overId = overEl?.getAttribute('data-node-id')
      setHoverTarget(overId && overId !== pending?.sourceId ? overId : null)
    }
  }, [onMoveNode, screenToWorld, pending?.sourceId])

  const onPointerUp = useCallback((e) => {
    const d = drag.current
    drag.current = null
    try { containerRef.current.releasePointerCapture(e.pointerId) } catch { /* noop */ }
    if (!d) return
    if (d.mode === 'connect') {
      if (hoverTarget) onAddConnection({ source_node_id: pending.sourceId, target_node_id: hoverTarget })
      setPending(null)
      setHoverTarget(null)
    } else if (d.mode === 'node' && d.moved && snap) {
      const node = nodeById(d.id)
      if (node) onMoveNode(d.id, Math.round(node.layout_x / GRID) * GRID, Math.round(node.layout_y / GRID) * GRID)
    } else if (d.mode === 'pan' && !d.moved) {
      onSelect(null, null) // click on empty canvas deselects
    }
  }, [hoverTarget, pending, onAddConnection, snap, nodeById, onMoveNode, onSelect])

  const worldStyle = { transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})` }

  if (!nodes.length) {
    return (
      <div className="cluster-canvas cluster-canvas--empty" ref={containerRef}>
        <div className="cluster-empty">
          <div className="cluster-empty__art" aria-hidden="true">
            <span className="cluster-empty__node"><IconNetwork size={30} /></span>
            <span className="cluster-empty__node cluster-empty__node--sm"><CamelidMark size={22} /></span>
            <span className="cluster-empty__node cluster-empty__node--sm cluster-empty__node--alt" />
          </div>
          <h3>No cluster nodes yet.</h3>
          <p>Add your first Mac, Windows PC, Linux server, or Raspberry Pi to start building your local compute fabric.</p>
          <div className="cluster-empty__actions">
            <Button variant="primary" icon={<IconPlus size={16} />} onClick={onAddServer}>Add Server</Button>
            {onLoadSample && <Button variant="ghost" onClick={onLoadSample}>Load a sample fabric</Button>}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div
      className={`cluster-canvas ${showGrid ? 'has-grid' : ''} ${pending ? 'is-linking' : ''}`}
      ref={containerRef}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onWheel={onWheel}
    >
      <svg className="cluster-canvas__links" width={size.w} height={size.h}>
        <g transform={`translate(${view.x} ${view.y}) scale(${view.scale})`}>
          {connections.map((c) => {
            const a = nodeById(c.source_node_id)
            const b = nodeById(c.target_node_id)
            if (!a || !b) return null
            const tone = statusTone(c.status)
            const selected = selection.kind === 'connection' && selection.id === c.id
            const [ax, ay] = rectEdge(a.layout_x, a.layout_y, b.layout_x, b.layout_y)
            const [bx, by] = rectEdge(b.layout_x, b.layout_y, a.layout_x, a.layout_y)
            const mx = (ax + bx) / 2
            const my = (ay + by) / 2
            const d = linkPath(ax, ay, bx, by)
            const label = c.label || CONNECTION_TYPE_BY[c.connection_type]?.label || 'Link'
            const lat = c.latency_ms != null ? ` · ${c.latency_ms} ms` : ''
            const bw = c.bandwidth_mbps != null ? ` · ${c.bandwidth_mbps >= 1000 ? `${(c.bandwidth_mbps / 1000).toFixed(0)} Gbps` : `${c.bandwidth_mbps} Mbps`}` : ''
            const text = `${label}${lat}${bw}`
            const live = tone === 'ready'
            return (
              <g key={c.id} className={`cluster-link is-${tone} cluster-link--${c.connection_type} ${selected ? 'is-selected' : ''}`}>
                <path className="cluster-link__hit" data-link-id={c.id} d={d} />
                <path className="cluster-link__line" d={d} />
                {live && <path className="cluster-link__flow" d={d} />}
                <g className="cluster-link__label" transform={`translate(${mx}, ${my})`} data-link-id={c.id}>
                  <rect x={-(text.length * 3.4 + 10)} y={-11} width={text.length * 6.8 + 20} height={22} rx={11} />
                  <text x={0} y={4} textAnchor="middle">{text}</text>
                </g>
              </g>
            )
          })}
          {pending && (() => {
            const src = nodeById(pending.sourceId)
            if (!src) return null
            return <path className="cluster-link__pending" d={linkPath(src.layout_x, src.layout_y, pending.toX, pending.toY)} />
          })()}
        </g>
      </svg>

      <div className="cluster-canvas__world" style={worldStyle}>
        {nodes.map((node) => (
          <div
            key={node.id}
            className={`cluster-node-pos ${hoverTarget === node.id ? 'is-drop-target' : ''}`}
            style={{ left: node.layout_x - NODE_W / 2, top: node.layout_y - NODE_H / 2 }}
          >
            <NodeCard
              node={node}
              selected={selection.kind === 'node' && selection.id === node.id}
              busyLabel={busyIds[node.id]}
              onSelect={() => onSelect('node', node.id)}
            />
          </div>
        ))}
      </div>

      <div className="cluster-canvas__controls">
        <Button variant="ghost" size="sm" className="cluster-ctl" icon={<IconZoomIn size={16} />} aria-label="Zoom in" onClick={() => setView((v) => zoomAround(v, 1.2, size.w / 2, size.h / 2))} />
        <Button variant="ghost" size="sm" className="cluster-ctl" icon={<IconZoomOut size={16} />} aria-label="Zoom out" onClick={() => setView((v) => zoomAround(v, 1 / 1.2, size.w / 2, size.h / 2))} />
        <span className="cluster-canvas__zoom">{Math.round(view.scale * 100)}%</span>
        <Button variant="ghost" size="sm" className="cluster-ctl" icon={<IconFit size={16} />} onClick={fit}>Fit</Button>
        <Button variant="ghost" size="sm" className="cluster-ctl" icon={<IconGrid size={16} />} onClick={() => onAutoLayout?.()}>Auto-layout</Button>
        <button type="button" className={`cluster-toggle ${snap ? 'is-on' : ''}`} onClick={() => setSnap((s) => !s)} title="Snap to grid">Snap</button>
        <button type="button" className={`cluster-toggle ${showGrid ? 'is-on' : ''}`} onClick={() => setShowGrid((g) => !g)} title="Toggle grid">Grid</button>
      </div>

      <Minimap nodes={nodes} view={view} size={size} setView={setView} />
    </div>
  )
})

function Minimap({ nodes, view, size, setView }) {
  const W = 180
  const H = 120
  if (!nodes.length) return null
  const xs = nodes.map((n) => n.layout_x)
  const ys = nodes.map((n) => n.layout_y)
  const pad = 160
  const minX = Math.min(...xs) - pad, maxX = Math.max(...xs) + pad
  const minY = Math.min(...ys) - pad, maxY = Math.max(...ys) + pad
  const bw = Math.max(1, maxX - minX), bh = Math.max(1, maxY - minY)
  const s = Math.min(W / bw, H / bh)
  const ox = (W - bw * s) / 2
  const oy = (H - bh * s) / 2
  const toMini = (wx, wy) => [ox + (wx - minX) * s, oy + (wy - minY) * s]
  // current viewport in world coords
  const vx0 = (0 - view.x) / view.scale
  const vy0 = (0 - view.y) / view.scale
  const vx1 = (size.w - view.x) / view.scale
  const vy1 = (size.h - view.y) / view.scale
  const [rx0, ry0] = toMini(vx0, vy0)
  const [rx1, ry1] = toMini(vx1, vy1)

  const onClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const mx = e.clientX - rect.left
    const my = e.clientY - rect.top
    const worldX = (mx - ox) / s + minX
    const worldY = (my - oy) / s + minY
    setView((v) => ({ ...v, x: size.w / 2 - worldX * v.scale, y: size.h / 2 - worldY * v.scale }))
  }

  return (
    <svg className="cluster-minimap" width={W} height={H} onPointerDown={(e) => e.stopPropagation()} onClick={onClick}>
      <rect x={0} y={0} width={W} height={H} rx={8} className="cluster-minimap__bg" />
      {nodes.map((n) => { const [x, y] = toMini(n.layout_x, n.layout_y); return <circle key={n.id} cx={x} cy={y} r={3} className={`is-${statusTone(n.status)}`} /> })}
      <rect className="cluster-minimap__view" x={Math.min(rx0, rx1)} y={Math.min(ry0, ry1)} width={Math.abs(rx1 - rx0)} height={Math.abs(ry1 - ry0)} />
    </svg>
  )
}

export default TopologyCanvas
