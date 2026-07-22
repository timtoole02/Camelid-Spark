/* Sampling-parameter contract lane (Phase 2).

   The chat request today is fixed: greedy temperature=0 plus the Settings
   max-tokens budget. A sampling control becomes editable ONLY when
   /api/capabilities advertises a supported feature row for that exact
   parameter; otherwise it renders as a guarded row (I3). Absence of a row is
   absence of evidence — fail closed, never default-open.

   As of 2026-06-12 the live contract advertises no sampling rows, so every
   control is guarded; see design-evidence/BACKEND_ASKS.md for the requested
   row shape. */

import { isSupportedCapabilityStatus } from './capabilities.js'

export const SAMPLING_PARAMS = [
  { key: 'temperature', label: 'temperature', current: '0 (greedy)', hint: 'Chat always sends temperature=0. Greedy decoding is what the parity evidence was produced with.' },
  { key: 'top_p', label: 'top_p', current: 'not sent', hint: 'Nucleus sampling is not part of any checked evidence lane.' },
  { key: 'top_k', label: 'top_k', current: 'not sent', hint: 'Top-k sampling is not part of any checked evidence lane.' },
  { key: 'max_tokens', label: 'max_tokens', current: 'Settings budget', hint: 'The response budget comes from Settings → Chat; the chat surface itself does not expose a cap.' },
  { key: 'stop', label: 'stop sequences', current: 'not sent', hint: 'Custom stop sequences are not part of any checked evidence lane.' },
  { key: 'seed', label: 'seed', current: 'not sent', hint: 'Seeded sampling is meaningless under greedy decoding and is not advertised by the contract.' },
]

/* A feature row counts for a parameter only on exact id match
   (sampling_<key> or <key>) — resemblance is not evidence. */
export function findSamplingFeature(apiFeatures = [], key) {
  return (apiFeatures || []).find((feature) => feature?.id === `sampling_${key}` || feature?.id === key) || null
}

export function isSamplingParamSupported(apiFeatures, key) {
  const feature = findSamplingFeature(apiFeatures, key)
  return Boolean(feature && isSupportedCapabilityStatus(feature.status))
}

const PARAM_STORAGE_PREFIX = 'camelid.samplingParams.'

export function loadSavedSamplingParams(modelId) {
  if (typeof window === 'undefined' || !modelId) return {}
  try {
    const saved = window.localStorage.getItem(`${PARAM_STORAGE_PREFIX}${modelId}`)
    return saved ? JSON.parse(saved) : {}
  } catch {
    return {}
  }
}

export function saveSamplingParams(modelId, params) {
  if (typeof window === 'undefined' || !modelId) return
  window.localStorage.setItem(`${PARAM_STORAGE_PREFIX}${modelId}`, JSON.stringify(params || {}))
}

/* Request overrides: only parameters whose exact contract row is supported
   ever reach the request body. With no supported rows this returns {}. */
export function contractSamplingOverrides(apiFeatures, modelId) {
  const saved = loadSavedSamplingParams(modelId)
  const overrides = {}
  for (const param of SAMPLING_PARAMS) {
    if (param.key === 'max_tokens') continue // owned by the Settings budget
    if (saved[param.key] !== undefined && isSamplingParamSupported(apiFeatures, param.key)) {
      overrides[param.key] = saved[param.key]
    }
  }
  return overrides
}
