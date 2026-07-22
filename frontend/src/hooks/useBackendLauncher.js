import { useCallback, useEffect, useRef, useState } from 'react'
import { backendProcStatus, launchBackend, stopBackend } from '../lib/devBackend'

const CMD_KEY = 'camelid.launchCommand'
// Bare values that don't resolve on most PATHs — treat as "auto-detect" instead of a real override.
const NON_OVERRIDES = new Set(['', 'camelid', 'camelid serve'])

function readCommand() {
  if (typeof window === 'undefined') return ''
  const saved = (window.localStorage.getItem(CMD_KEY) || '').trim()
  if (NON_OVERRIDES.has(saved)) {
    if (saved) window.localStorage.removeItem(CMD_KEY) // migrate the old broken default
    return ''
  }
  return saved
}

/**
 * Backend launcher state shared by the Settings page and the offline banner.
 * Polls the dev-server hook, manages an optional command override (blank =
 * auto-detect the built binary server-side), and exposes start/stop.
 */
export function useBackendLauncher({ showNotice, loadDashboard } = {}) {
  const [status, setStatus] = useState({ available: false, running: false, detected: null, logTail: '' })
  const [command, setCommandState] = useState(readCommand)
  const [starting, setStarting] = useState(false)
  const pollRef = useRef(null)

  const refresh = useCallback(async () => setStatus(await backendProcStatus()), [])

  useEffect(() => {
    refresh()
    pollRef.current = window.setInterval(refresh, 2500)
    return () => window.clearInterval(pollRef.current)
  }, [refresh])

  const setCommand = useCallback((value) => {
    setCommandState(value)
    if (typeof window === 'undefined') return
    if (value.trim()) window.localStorage.setItem(CMD_KEY, value)
    else window.localStorage.removeItem(CMD_KEY)
  }, [])

  const start = useCallback(async () => {
    setStarting(true)
    showNotice?.('Starting Camelid…', 'info')
    try {
      await launchBackend(command.trim()) // blank → the dev hook auto-detects the binary
      await refresh()
      window.setTimeout(() => loadDashboard?.({ silent: true }), 1500)
      window.setTimeout(() => loadDashboard?.({ silent: true }), 4000)
      showNotice?.('Camelid is starting — waiting for it to come online…', 'success')
    } catch (error) {
      showNotice?.(error?.message || 'Could not start Camelid.', 'error')
    } finally {
      setStarting(false)
    }
  }, [command, refresh, showNotice, loadDashboard])

  const stop = useCallback(async () => {
    showNotice?.('Stopping Camelid…', 'info')
    await stopBackend()
    await refresh()
    showNotice?.('Sent stop signal to Camelid.', 'success')
  }, [refresh, showNotice])

  const resolvedCommand = command.trim() || status.detected || ''

  return { status, command, setCommand, resolvedCommand, starting, start, stop, refresh }
}
