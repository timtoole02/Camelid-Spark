import { compatibilityHintCopy, compatibilityHintLabel, findCompatibilityHint, isCompatibilitySupportedForModel } from './capabilities.js'
import { isRunnableInCurrentRuntime, modelRuntimeIdMatches } from './modelState.js'

export function getChatGateState(capabilities, model, runtime) {
  const runtimeLoaded = Boolean(runtime?.loaded_now && modelRuntimeIdMatches(model, runtime))
  const runtimeGenerationReady = Boolean(runtime?.generation_ready && modelRuntimeIdMatches(model, runtime))
  const runtimeReady = Boolean(isRunnableInCurrentRuntime(model, runtime) && runtimeLoaded && runtimeGenerationReady)
  const hint = findCompatibilityHint(capabilities, model)
  const contractSupported = isCompatibilitySupportedForModel(capabilities, model)
  const chatUnlocked = Boolean(runtimeReady && contractSupported)
  // Experimental lane: the model loaded and is generation-ready (so its architecture
  // is implemented — generation_ready is false for unimplemented archs) but it is NOT
  // a supported contract row. A separate, weaker affordance from the supported gate:
  // chat is allowed but every turn is marked unverified with no parity claim.
  const experimentalUnlocked = Boolean(runtimeReady && !contractSupported)
  const chatMode = contractSupported ? 'supported' : experimentalUnlocked ? 'experimental' : 'blocked'

  return {
    hint,
    runtimeReady,
    runtimeLoaded,
    runtimeGenerationReady,
    contractSupported,
    chatUnlocked,
    experimentalUnlocked,
    chatMode,
    label: compatibilityHintLabel(hint, 'No matching COMPATIBILITY.md row'),
    copy: compatibilityHintCopy(hint),
  }
}
