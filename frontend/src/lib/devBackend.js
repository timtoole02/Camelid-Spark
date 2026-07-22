/* Client for the dev-server backend launcher (see vite.config.js).
   All calls fail soft: when the hook isn't present (static build / vite preview),
   the routes return the SPA HTML, so JSON parsing throws and we report
   { available: false } — the Settings page then shows the copy-command fallback. */
const BASE = '/__camelid/backend'

async function parseJson(response) {
  const text = await response.text()
  const data = JSON.parse(text) // throws on HTML fallback
  return data
}

export async function backendProcStatus() {
  try {
    const response = await fetch(`${BASE}/status`, { headers: { Accept: 'application/json' } })
    const data = await parseJson(response)
    return data && data.available ? data : { available: false }
  } catch {
    return { available: false }
  }
}

export async function launchBackend(command) {
  const response = await fetch(`${BASE}/launch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ command }),
  })
  let data = {}
  try { data = await parseJson(response) } catch { /* noop */ }
  if (!response.ok) throw new Error(data?.error || `Launch failed (HTTP ${response.status})`)
  return data
}

export async function stopBackend() {
  try {
    const response = await fetch(`${BASE}/stop`, { method: 'POST' })
    return await parseJson(response)
  } catch {
    return { available: false, running: false }
  }
}
