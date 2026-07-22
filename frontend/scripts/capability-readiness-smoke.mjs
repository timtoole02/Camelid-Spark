/* Smoke for capabilityReadiness.js: the readiness taxonomy must come from the
   row's own contract fields, never from family-name matching, and multimodal
   input must always classify as unsupported. */

import {
  classifyCapabilityRow,
  classifyInputModality,
  isExactRowSupported,
  readinessLabel,
  READINESS,
} from '../src/lib/capabilityReadiness.js'

let failures = 0
function check(name, actual, expected) {
  const ok = actual === expected
  if (!ok) {
    failures += 1
    console.error(`FAIL ${name}: got ${JSON.stringify(actual)}, want ${JSON.stringify(expected)}`)
  } else {
    console.log(`ok ${name}`)
  }
}

check(
  'supported exact row',
  classifyCapabilityRow({ id: 'gemma4_e2b_it_q8_0', status: 'supported_exact_row_smoke' }),
  READINESS.SUPPORTED_EXACT_ROW,
)
check(
  'active validation',
  classifyCapabilityRow({ id: 'mixtral_8x7b_instruct_v0_1_q8_0', status: 'active_validation_partial_runtime' }),
  READINESS.ACTIVE_VALIDATION,
)
check(
  'planned candidate',
  classifyCapabilityRow({ id: 'gemma2_9b', status: 'planned_exact_row_candidate' }),
  READINESS.PLANNED,
)
check(
  'unsupported quantization',
  classifyCapabilityRow({ id: 'llama_spm_q4_0_q5_0', status: 'planned_phase_10', tensors_load: 'unsupported_typed_error' }),
  READINESS.UNSUPPORTED_QUANTIZATION,
)
check(
  'gpu experimental stays distinct from green',
  classifyCapabilityRow({ id: 'x', status: 'supported_exact_row_smoke', performance_measured: 'gpu_experimental_parity_pending' }),
  READINESS.GPU_EXPERIMENTAL,
)
check('image input fails closed', classifyInputModality('image'), READINESS.UNSUPPORTED_MULTIMODAL)
check('audio input fails closed', classifyInputModality('audio'), READINESS.UNSUPPORTED_MULTIMODAL)
check('video input fails closed', classifyInputModality('video'), READINESS.UNSUPPORTED_MULTIMODAL)
check('text input is allowed', classifyInputModality('text'), READINESS.SUPPORTED_EXACT_ROW)

const capabilities = {
  model_compatibility: [
    { id: 'gemma4_e4b_it_q8_0', status: 'supported_exact_row_smoke' },
    { id: 'gemma4_e2b_it_q8_0', status: 'supported_exact_row_smoke' },
  ],
}
check('exact row id supported', isExactRowSupported(capabilities, 'gemma4_e2b_it_q8_0'), true)
check(
  'family-name prefix must NOT count as supported',
  isExactRowSupported(capabilities, 'gemma4_12b_it_q8_0'),
  false,
)
check('family string never matches', isExactRowSupported(capabilities, 'gemma4'), false)
check('label text', readinessLabel(READINESS.UNSUPPORTED_MULTIMODAL), 'Unsupported: multimodal input')

if (failures > 0) {
  console.error(`${failures} capability-readiness checks failed`)
  process.exit(1)
}
console.log('capability-readiness smoke passed')
