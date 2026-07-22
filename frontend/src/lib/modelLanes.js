import { isCompatibilitySupportedForModel } from './capabilities.js'

/* Derived lane membership for local models — extracted verbatim from
   LocalLaneSections so every consumer computes lanes the same way. Membership is
   computed from /api/models/local lane facts + the /api/capabilities contract —
   never a hand-authored array. */

export const LANES = ['supported', 'compatible', 'eligible', 'not_anchored']

/* A model object shaped for the existing contract matcher (it reads id/name/
   model_path/quant). The supported gate stays the contract's voice — we only ask it. */
export function matchModel(entry) {
  return {
    id: entry.filename,
    name: entry.filename,
    model_path: entry.filename,
    hf_filename: entry.filename,
    quant: entry.quantization,
  }
}

export function laneOf(entry, capabilities) {
  if (isCompatibilitySupportedForModel(capabilities, matchModel(entry))) return 'supported'
  if (entry.runnable_receipt_present) return 'compatible'
  if (entry.admitted && entry.oracle_qualified) return 'eligible'
  return 'not_anchored'
}

export function bucketByLane(models = [], capabilities) {
  const buckets = { supported: [], compatible: [], eligible: [], not_anchored: [] }
  for (const m of models) buckets[laneOf(m, capabilities)].push(m)
  return buckets
}
