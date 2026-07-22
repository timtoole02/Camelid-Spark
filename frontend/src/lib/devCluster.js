/* Client for the dev-server cluster helpers (see vite.config.js):
   real localhost-only actions, no cloud. Fails soft when the dev hook isn't
   present (static build) so the UI can fall back to config-only behavior. */
const BASE = '/__camelid/cluster'

async function parseJson(response) {
  const text = await response.text()
  return JSON.parse(text) // throws on HTML fallback
}

/** TCP reachability + latency probe for host:port. */
export async function probeNode({ host, port }) {
  try {
    const response = await fetch(`${BASE}/probe`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ host, port }),
    })
    const data = await parseJson(response)
    return { available: true, ...data }
  } catch {
    return { available: false }
  }
}

/** Live camelid telemetry for a node, fetched straight from the browser.
   camelid serves permissive CORS (access-control-allow-origin: *), so this works
   cross-origin with no dev hook — and reports the node's real hardware specs.
   Returns { online, engine, active_model_id, generation_ready, specs } or { online:false }. */
export async function fetchNodeTelemetry({ host, port = 8181, timeoutMs = 2500 }) {
  const base = `http://${host}:${port}`
  const get = (path) => {
    const ctrl = new AbortController()
    const timer = setTimeout(() => ctrl.abort(), timeoutMs)
    return fetch(`${base}${path}`, { signal: ctrl.signal, mode: 'cors' })
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null)
      .finally(() => clearTimeout(timer))
  }
  const [health, caps] = await Promise.all([get('/v1/health'), get('/api/capabilities')])
  if (!health && !caps) return { online: false }
  const plan = (caps && caps.execution_plan) || {}
  return {
    online: true,
    engine: (health && health.engine) || (caps && caps.engine) || 'camelid',
    active_model_id: (health && health.active_model_id) || null,
    generation_ready: Boolean(health && health.generation_ready),
    specs: {
      os: plan.operating_system || null,
      arch: plan.architecture || null,
      cpu_model: plan.cpu_model || null,
      cpu_cores: Number(plan.thread_count) || null,
      platform_label: plan.platform_label || null,
      backend: plan.selected_backend || null,
    },
  }
}

/** Safe local discovery: mDNS (dns-sd/avahi) or ARP table; review before adding. */
export async function discoverDevices() {
  try {
    const response = await fetch(`${BASE}/discover`, { headers: { Accept: 'application/json' } })
    const data = await parseJson(response)
    return data && data.available ? data : { available: false, devices: [] }
  } catch {
    return { available: false, devices: [] }
  }
}
