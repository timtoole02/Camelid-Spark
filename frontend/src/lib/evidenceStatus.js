/* Presentation-only mapping from contract status strings to Evidence Chip
   states. This module renders claims; it must NEVER participate in chat
   readiness decisions — those stay in chatGate.js/capabilities.js untouched.

   States (the design's vocabulary, one icon + one color family each):
   - supported          copper. Reserved for contract `supported`/`supported_*`
                        rows only — the receipt-stamp state.
   - runnable           amber (🟡). The generic runnable lane: a model that executes
                        deterministically and is anchored to the HF reference, but is
                        NOT a supported parity contract. Distinct from supported (never
                        copper) and from unsupported.
   - evidence           desaturated amber. Bounded/row-scoped evidence facts:
                        validated, measured, pass, partial, guarded lanes.
   - acceptance-target  cool slate. A named target the ledger tracks.
   - groundwork         cool slate. Plumbing exists, no promotion claim.
   - planned            cool slate. Honestly not built yet.
   - unsupported        muted slate. A normal, honest state — never alarming.
   - error              red. Operational failures only, never "unsupported".
   - neutral            descriptive metadata, no claim.
*/

import { formatCapabilityStatus } from './capabilities.js'

export const EVIDENCE_STATES = [
  'supported',
  'runnable',
  'evidence',
  'acceptance-target',
  'groundwork',
  'planned',
  'unsupported',
  'error',
  'neutral',
]

const EVIDENCE_VALUE_HINTS = ['validated', 'measured', 'pass', 'evidence', 'bounded', 'partial', 'guarded', 'smoke']
const PLANNED_VALUES = new Set(['planned', 'future', 'pending', 'not_started', 'roadmap'])
const UNSUPPORTED_VALUES = new Set(['unsupported', 'not_promoted', 'fail_closed', 'fail-closed', 'blocked', 'unsupported_multimodal'])

export function classifyEvidenceState(status = '') {
  const value = String(status).toLowerCase().trim()
  if (!value) return 'neutral'
  if (value === 'supported' || value.startsWith('supported_')) return 'supported'
  // Runnable is anchored/deterministic but never a supported contract — it must
  // classify to its own amber state, never into the copper supported state.
  if (value === 'runnable' || value.startsWith('runnable_')) return 'runnable'
  if (value === 'acceptance_target' || value === 'acceptance-target' || value.startsWith('acceptance_target')) return 'acceptance-target'
  if (value === 'groundwork' || value.startsWith('groundwork')) return 'groundwork'
  if (PLANNED_VALUES.has(value)) return 'planned'
  if (UNSUPPORTED_VALUES.has(value)) return 'unsupported'
  if (value === 'error' || value === 'failed' || value.startsWith('error_')) return 'error'
  if (EVIDENCE_VALUE_HINTS.some((hint) => value.includes(hint))) return 'evidence'
  return 'neutral'
}

/* Short human framing per state — claim scope, in product voice. */
export const EVIDENCE_STATE_COPY = {
  supported: 'Supported for this exact row only. Resemblance is not evidence.',
  runnable: 'Runnable lane: this model executes deterministically and is anchored to the HF reference. That is cross-checked execution, not a supported parity contract — never copper.',
  evidence: 'Row-scoped evidence exists for the bounded lane named here — not a broader support claim.',
  'acceptance-target': 'A tracked acceptance target. Listing it is not a support claim.',
  groundwork: 'Groundwork only: plumbing exists, nothing is promoted by it.',
  planned: 'Planned. It does not run today and the UI will not pretend it does.',
  unsupported: 'Not supported. This is a normal, honest state — load results and typed backend errors stay the source of truth.',
  error: 'An operational error was reported. This says nothing about row support either way.',
  neutral: 'Descriptive metadata only — not support evidence.',
}

export const EVIDENCE_STATE_LABELS = {
  supported: 'Supported',
  runnable: 'Runnable',
  evidence: 'Evidence',
  'acceptance-target': 'Target',
  groundwork: 'Groundwork',
  planned: 'Planned',
  unsupported: 'Unsupported',
  error: 'Error',
  neutral: 'Info',
}

export function evidenceLabelFromStatus(status, fallback = '') {
  const formatted = formatCapabilityStatus(status)
  return formatted || fallback
}
