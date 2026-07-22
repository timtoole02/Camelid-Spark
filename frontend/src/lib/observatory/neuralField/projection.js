/* Neural Field projection core — a hand-rolled perspective projection onto
   Canvas2D. Pure functions over a small orbit-camera state; no dependency on
   the DOM, the store, or three.js (constraint: zero new runtime deps).

   Space convention: scene points live in a unit-ish space centered on the
   origin, camera orbits the origin at `dist` and looks at it, +Z toward the
   viewer at yaw = pitch = 0. `project` returns CSS-pixel coordinates plus a
   `scale` factor that drives node radius and the depth-fade alpha. */

const IDLE_DRIFT_RAD_PER_S = 0.014 // ≤ 0.02 rad/s by contract
const ACTIVE_PITCH_DELTA = -0.05
const ACTIVE_EASE_MS = 800

export function makeCamera({ fov = 1.1, dist = 620 } = {}) {
  return {
    yaw: 0,
    pitch: -0.12,
    dist,
    fov,
    // Ease state for the active-run vantage shift (orbitStep drives this).
    basePitch: -0.12,
    activeBlend: 0, // 0 = idle vantage, 1 = active vantage
  }
}

/* Perspective-project a scene point. Returns null when the point is behind
   the camera (culled). `scale` is 1 at the orbit distance, larger closer. */
export function project(point3, camera, w, h) {
  const cy = Math.cos(camera.yaw)
  const sy = Math.sin(camera.yaw)
  const cp = Math.cos(camera.pitch)
  const sp = Math.sin(camera.pitch)

  // Yaw about Y, then pitch about X, then push back by orbit distance.
  const x1 = point3.x * cy + point3.z * sy
  const z1 = -point3.x * sy + point3.z * cy
  const y2 = point3.y * cp - z1 * sp
  const z2 = point3.y * sp + z1 * cp
  const zc = camera.dist - z2
  if (zc <= 1) return null

  const focal = (0.5 * h) / Math.tan(camera.fov * 0.5)
  const scale = focal / zc
  return {
    x: w * 0.5 + x1 * scale,
    y: h * 0.5 - y2 * scale,
    depth: zc,
    scale: scale / (focal / camera.dist), // normalized: 1.0 at orbit distance
  }
}

/* Depth-fade alpha factor for a projected point: far = 0.35, near = 1. */
export function depthAlpha(projected) {
  const t = Math.min(Math.max((projected.scale - 0.6) / 0.8, 0), 1)
  return 0.35 + 0.65 * t
}

/* Advance the camera one frame.
   - reduced motion: camera fixed, no drift, no vantage ease.
   - idle: slow yaw drift only (explicitly-idle ambient treatment).
   - active run: ease toward a slightly lower vantage (pitch −0.05) over
     ~800ms; ease back when the run finishes. Drift continues while active —
     it is the same ambient camera, not an inference signal. */
export function orbitStep(camera, dtMs, active, reducedMotion = false) {
  if (reducedMotion) return camera
  const dt = Math.min(Math.max(dtMs, 0), 100)
  camera.yaw += IDLE_DRIFT_RAD_PER_S * (dt / 1000)
  if (camera.yaw > Math.PI * 2) camera.yaw -= Math.PI * 2
  const step = dt / ACTIVE_EASE_MS
  camera.activeBlend = active
    ? Math.min(1, camera.activeBlend + step)
    : Math.max(0, camera.activeBlend - step)
  // Smoothstep the blend so the vantage shift starts and ends gently.
  const b = camera.activeBlend * camera.activeBlend * (3 - 2 * camera.activeBlend)
  camera.pitch = camera.basePitch + ACTIVE_PITCH_DELTA * b
  return camera
}

/* Painter's algorithm: sort drawables far-to-near by projected depth. Items
   are `{ depth }`-bearing records; caller keeps total count ≤ ~4,000 so this
   stays cheap per frame. Sorts in place and returns the array. */
export function sortByDepth(drawables) {
  return drawables.sort((a, b) => b.depth - a.depth)
}
