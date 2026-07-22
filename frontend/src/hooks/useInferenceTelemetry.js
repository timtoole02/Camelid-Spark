/* React binding for the InferenceTelemetryStore: one store instance per
   mount, connected to the active API base. Lifecycle events re-render
   immediately via the store's subscription; fast-changing live metrics are
   sampled on a coarse interval so token-rate events never thrash React. */

import { useEffect, useReducer } from 'react'
import { createInferenceTelemetryStore } from '../lib/inferenceTelemetry'

const LIVE_SAMPLE_MS = 250

/* Shared app-lifetime store (Phase 6.1 DEFECT 1 fix): the SSE stream has no
   replay, so a per-mount store loses every event emitted while the view is on
   another tab and wipes run state on navigation. One module-level store stays
   connected for the session; unmount only unsubscribes the React binding. */
const sharedStore = createInferenceTelemetryStore()
let connectedBase = null

/* Called from the app shell so the stream listens from app start — connecting
   on first view mount would still lose every run made before the user ever
   opens the Observatory. */
export function ensureInferenceTelemetryConnected(apiBase) {
  if (apiBase && apiBase !== connectedBase) {
    connectedBase = apiBase
    sharedStore.connect(apiBase)
  }
}

export function useInferenceTelemetry(apiBase) {
  const store = sharedStore
  const [, bump] = useReducer((n) => n + 1, 0)

  useEffect(() => {
    ensureInferenceTelemetryConnected(apiBase)
    const unsubscribe = store.subscribe(bump)
    return () => unsubscribe()
  }, [store, apiBase])

  useEffect(() => {
    const interval = window.setInterval(() => {
      const run = store.getRun()
      if (run.active || store.getConnection() !== 'live') bump()
    }, LIVE_SAMPLE_MS)
    return () => window.clearInterval(interval)
  }, [store])

  return store
}
