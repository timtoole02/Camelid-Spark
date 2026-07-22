function displayPlanValue(value = '') {
  return String(value).trim().replace(/_/g, ' ')
}

const CPU_BACKENDS = new Set([
  'cpu_reference',
  'cpu_q8_runtime_repack',
  'cpu_kquant_block_dot',
])

const CUDA_BACKENDS = new Set([
  'cuda_resident_q8_runtime',
  'cuda_resident_q8_runtime_runnable_unvalidated',
  'cuda_resident_kquant_runtime',
])

const METAL_BACKENDS = new Set([
  'metal_resident_q8_runtime',
])

const SPECIALIZED_BACKENDS = new Set([
  'gemma4-runtime',
  'runnable-runtime',
  'diffusion-gemma-runtime',
])

export function executionRuntimeFields(health) {
  return {
    execution_plan: health?.execution_plan || null,
    backend: health?.backend || 'none',
  }
}

export function describeExecutionPlan(runtime) {
  if (runtime?.status === 'offline') {
    return {
      state: 'offline',
      device: 'Unavailable',
      backend: 'Backend offline',
      summary: 'Execution details are unavailable while the Camelid backend is offline.',
    }
  }

  if (!runtime?.loaded_now) {
    return {
      state: 'idle',
      device: 'No model loaded',
      backend: 'No active plan',
      summary: 'No model is loaded, so Camelid has no active execution plan.',
    }
  }

  if (!runtime?.generation_ready) {
    return {
      state: 'pending',
      device: 'Not active',
      backend: 'Model not generation-ready',
      summary: 'A model is loaded, but Camelid is not generation-ready, so no execution claim is shown.',
    }
  }

  if (SPECIALIZED_BACKENDS.has(runtime?.backend)) {
    const backend = displayPlanValue(runtime.backend)
    return {
      state: 'specialized',
      device: 'Runtime-specific',
      backend,
      summary: `Camelid reports the active model is served by ${backend}; the generic load-time execution plan is not used for a device claim.`,
    }
  }

  if (runtime?.backend !== 'llama') {
    return {
      state: 'unknown',
      device: 'Not reported',
      backend: displayPlanValue(runtime?.backend) || 'Backend unavailable',
      summary: 'Camelid did not report a recognized serving backend for the loaded model.',
    }
  }

  const plan = runtime?.execution_plan || null
  if (!plan) {
    return {
      state: 'unknown',
      device: 'Not reported',
      backend: 'Plan unavailable',
      summary: 'A model is loaded, but Camelid did not return an active execution plan.',
    }
  }

  const selectedBackend = String(plan.selected_backend || '')
  if (!selectedBackend) {
    return {
      state: 'unknown',
      device: 'Not reported',
      backend: 'Plan unavailable',
      summary: 'Camelid returned an execution plan without a selected backend.',
    }
  }

  const cudaSelected = CUDA_BACKENDS.has(selectedBackend)
  const metalActive = METAL_BACKENDS.has(selectedBackend)
  const cpuSelected = CPU_BACKENDS.has(selectedBackend)
  const cudaConsistent = cudaSelected && plan.cuda_resident_active === true
  const contradictoryCuda = cudaSelected !== (plan.cuda_resident_active === true)
  if ((!cpuSelected && !cudaSelected && !metalActive) || contradictoryCuda) {
    return {
      state: 'unknown',
      device: 'Not reported',
      backend: displayPlanValue(selectedBackend),
      summary: 'Camelid returned a load-time execution plan that this UI cannot classify safely.',
    }
  }

  const device = cudaConsistent ? 'CUDA GPU' : metalActive ? 'Metal GPU' : 'CPU'
  const backend = displayPlanValue(selectedBackend)

  return {
    state: cudaConsistent ? 'cuda' : metalActive ? 'metal' : 'cpu',
    device,
    backend,
    summary: `At model load, Camelid selected ${device} using ${backend}. Runtime controls may change the effective path afterward.`,
  }
}