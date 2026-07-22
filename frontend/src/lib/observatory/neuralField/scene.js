/* Neural Field scene graph — the model as geometry.

   The network is built from the real model shape: one disc of nodes per
   transformer layer, receding along Z as a tunnel (front = layer 0, back =
   final layer), plus an input plane (prompt staging), an output rail to the
   sampler bloom point, and a KV column. Built lazily on the first event that
   carries a layer count; before any model reports depth, the resting
   placeholder is the same geometry at 12 discs (matching LayerVisualizer's
   `ringCount = total || 12` convention).

   18 nodes per disc is a DECLARED VISUAL ABSTRACTION — head counts are not
   in telemetry, so discs are stylized layer cross-sections, and the view
   copy says so. Geometry constants follow the design mockup approved
   2026-07-02 (S-curve axis, helix twist, strong front-to-back taper),
   remapped from the mockup's 680-wide 2D frame into camera space. */

export const NODES_PER_DISC = 18
export const PLACEHOLDER_LAYERS = 12
const HELIX_TWIST_RAD = 0.14 // per-disc angular offset; braids the edges

/* World-space extents (camera orbits at dist ~620, fov ~1.1; see projection.js).
   Front of the tunnel sits toward the camera (+Z), back recedes. */
const Z_FRONT = 160
const Z_BACK = -170
const X_START = -180 // layer 0 center (mockup x=225/680)
const X_SPAN = 340 // rightward travel to the final layer (mockup x=475/680)
const X_WOBBLE = 24 // S-curve amplitude (mockup 18/680, scaled)
const Y_START = -78 // layer 0 center (mockup y=292, low on screen)
const Y_SPAN = 158 // upward travel to the final layer (mockup y=134)

function discParams(t) {
  return {
    cx: X_START + X_SPAN * t + X_WOBBLE * Math.sin(2.2 * t),
    cy: Y_START + Y_SPAN * t,
    cz: Z_FRONT + (Z_BACK - Z_FRONT) * t,
    // Strong perspective curve from the mockup; back discs small. True
    // perspective adds a further ~0.6x on top — that is the intended look.
    rx: 32 + 128 * Math.pow(1 - t, 1.35),
  }
}

export function buildScene(layersTotal) {
  const total = Number.isInteger(layersTotal) && layersTotal > 1 ? layersTotal : PLACEHOLDER_LAYERS
  const placeholder = !(Number.isInteger(layersTotal) && layersTotal > 1)

  const discs = []
  const nodes = []
  for (let d = 0; d < total; d += 1) {
    const t = d / (total - 1)
    const { cx, cy, cz, rx } = discParams(t)
    const ry = rx * 0.3
    const disc = { index: d, t, cx, cy, cz, rx, ry, nodes: [] }
    for (let k = 0; k < NODES_PER_DISC; k += 1) {
      const angle = (k / NODES_PER_DISC) * Math.PI * 2 + d * HELIX_TWIST_RAD
      const node = {
        disc: d,
        k,
        x: cx + rx * Math.cos(angle),
        y: cy + ry * Math.sin(angle),
        z: cz,
        // Pre-projection front-half cue: lower-rim nodes render brighter.
        frontness: Math.sin(angle),
      }
      disc.nodes.push(node)
      nodes.push(node)
    }
    discs.push(disc)
  }

  // Edges: node k of disc d → node k of disc d+1. The helix twist supplies
  // the braid; no cross-wiring. Precomputed once.
  const edges = []
  for (let d = 0; d < total - 1; d += 1) {
    for (let k = 0; k < NODES_PER_DISC; k += 1) {
      edges.push({ a: discs[d].nodes[k], b: discs[d + 1].nodes[k], disc: d })
    }
  }

  const front = discParams(0)
  const back = discParams(1)

  // Input plane: prompt-token staging area floating in front of layer 0.
  const inputPlane = {
    cx: front.cx - 150,
    cy: front.cy - 10,
    cz: front.cz + 60,
    halfW: 62,
    halfH: 40,
  }

  // Output rail: from the final disc to the sampler bloom point behind it.
  const samplerPoint = { x: back.cx + 96, y: back.cy + 26, z: back.cz - 70 }
  const outputRail = {
    from: { x: back.cx, y: back.cy, z: back.cz },
    to: samplerPoint,
  }

  // KV column: a vertical bar beside the stack; filled height = kv.position /
  // kv.capacity, approx_bytes surfaces in the label layer only.
  const kvColumn = { x: 235, zBase: 20, yBottom: -100, height: 190, halfW: 9 }

  return { layersTotal: total, placeholder, discs, nodes, edges, inputPlane, outputRail, samplerPoint, kvColumn }
}

/* Point along the input-plane → layer-0 approach path, s in 0..1. Used by
   inbound prompt motes. Each mote gets a lane offset (`ox`,`oy`) so the
   stream has width without random per-frame jitter. */
export function inboundPoint(scene, s, ox = 0, oy = 0) {
  const p = scene.inputPlane
  const f = scene.discs[0]
  return {
    x: p.cx + (f.cx - p.cx) * s + ox * (1 - s),
    y: p.cy + (f.cy - p.cy) * s + oy * (1 - s),
    z: p.cz + (f.cz - p.cz) * s,
  }
}

/* Point along the output rail, s in 0..1 (final disc → sampler point). */
export function railPoint(scene, s) {
  const { from, to } = scene.outputRail
  return {
    x: from.x + (to.x - from.x) * s,
    y: from.y + (to.y - from.y) * s,
    z: from.z + (to.z - from.z) * s,
  }
}
