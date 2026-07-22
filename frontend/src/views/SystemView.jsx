import { displayCapabilityCopy, displayCapabilityId, exactRowSupportLanes, findCompatibilityHint, formatCapabilityStatus, frontendSupportContractCopy, guardedCapabilityCopy, isExactCompatibilityHint, isGuardedCapabilityStatus, isSupportedCapabilityStatus, rowSupportNextStepCopy } from '../lib/capabilities'
import { getChatGateState } from '../lib/chatGate'
import { describeModelState, getRuntimeRequestModelId } from '../lib/modelState'
import { describeExecutionPlan } from '../lib/executionPlan'
import { StatusDot } from '../components/ui/StatusDot'
import { EvidenceChip } from '../components/ui/EvidenceChip'
import { CanonicalStatement } from '../components/ui/CanonicalStatement'
import { ExactRowEvidenceSummary } from '../components/ui/ExactRowEvidenceSummary'
import { EmptyState } from '../components/ui/EmptyState'
import { IconSystem } from '../components/ui/icons'

function runtimeReadinessLabel(runtime) {
  if (runtime?.generation_ready) return 'Loaded for local generation'
  if (runtime?.loaded_now) return 'Loaded, checking generation readiness'
  return 'Waiting for a generation-ready model'
}

function supportLaneTitle(lane) {
  if (lane.key === 'template') return 'Template/Jinja readiness'
  if (lane.key === 'context') return 'Checked context readiness'
  return 'Throughput readiness'
}

export default function SystemView({ runtime, selectedModel, capabilities }) {
  const runtimePill = runtimeReadinessLabel(runtime)
  const selectedModelName = selectedModel?.name || 'No next-chat model selected'
  const apiBase = runtime?.api_base || 'Local API unavailable'
  const modelId = getRuntimeRequestModelId(selectedModel, runtime, '<loaded-model-id>') || '<loaded-model-id>'
  const supportContract = capabilities?.support_contract
  const supportContractCurrentGate = frontendSupportContractCopy(capabilities)
  const compatibilityTargets = capabilities?.model_compatibility || []
  const execution = describeExecutionPlan(runtime)
  const supportedCompatibilityCount = compatibilityTargets.filter((target) => isSupportedCapabilityStatus(target.status)).length
  const apiFeatures = capabilities?.api_features || []
  const supportedFeatures = apiFeatures.filter((feature) => isSupportedCapabilityStatus(feature.status))
  const unsupportedFeatures = apiFeatures.filter((feature) => isGuardedCapabilityStatus(feature.status))
  const selectedChatGate = getChatGateState(capabilities, selectedModel, runtime)
  const selectedCompatibilityHint = selectedChatGate.hint || findCompatibilityHint(capabilities, selectedModel)
  const selectedCompatibilityTarget = isExactCompatibilityHint(selectedCompatibilityHint) ? selectedCompatibilityHint.target : null
  const selectedSupportLanes = exactRowSupportLanes(selectedCompatibilityTarget, apiFeatures)
  const selectedExactRowReady = selectedChatGate.chatUnlocked
  const endpointReadinessLabel = selectedExactRowReady
    ? 'Selected exact-row local /v1 ready'
    : selectedChatGate.runtimeReady
      ? 'Runtime ready, support gated'
      : runtime?.generation_ready
        ? 'Different loaded model or exact row required'
        : runtime?.loaded_now
          ? 'Loaded, not generation-ready'
          : 'Load a supported exact row'
  const chatCompletionsCopy = selectedExactRowReady
    ? 'Runs now for this selected GGUF because loaded_now=true, generation_ready=true, active_model_id matches, and the exact /api/capabilities row is supported.'
    : selectedCompatibilityTarget
      ? 'Blocked for UX chat until loaded_now=true, generation_ready=true, active_model_id matches, and this exact row is supported.'
      : 'Blocked for UX chat until a selected model matches an exact supported COMPATIBILITY.md row and runtime readiness is green.'
  const curlExample = selectedExactRowReady
    ? `# Selected exact row is runtime-ready now\ncurl ${runtime?.api_base}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "${modelId}",
    "messages": [{"role": "user", "content": "Hello from Camelid"}],
    "temperature": 0
  }'`
    : `# Blocked for UX chat until selected exact row evidence and runtime readiness both match
# loaded_now=${runtime?.loaded_now ? 'true' : 'false'} generation_ready=${runtime?.generation_ready ? 'true' : 'false'} active_model_id=${runtime?.active_model_id || 'none'}
# selected_exact_row=${selectedCompatibilityTarget?.id || 'none'} support_gate=${selectedChatGate.label}`
  const q8Runtime = runtime?.q8_runtime
  const q8RuntimeLabel = q8Runtime?.retain_q8_blocks
    ? 'Retained Q8 blocks'
    : q8Runtime?.lazy_q8_linear
      ? 'Lazy Q8 policy'
      : q8Runtime
        ? 'Eager CPU materialization'
        : 'Q8 policy unavailable'
  const q8RuntimeDetail = q8Runtime
    ? `${q8Runtime.policy}${Number.isFinite(q8Runtime.file_cache_bytes) ? ` · cache ${q8Runtime.file_cache_bytes} bytes` : ''}`
    : 'Start the local runtime to inspect Q8 storage policy.'

  const runtimeState = runtime?.generation_ready ? 'Ready' : runtime?.loaded_now ? 'Loaded' : runtime?.status === 'offline' ? 'Offline' : 'Idle'
  const runtimeTone = runtime?.generation_ready ? 'ready' : runtime?.loaded_now ? 'warn' : 'neutral'
  const gateStat = selectedExactRowReady ? 'Ready' : selectedChatGate.runtimeReady ? 'Gated' : 'Blocked'
  const gateStatSub = selectedExactRowReady ? 'chat + API unlocked' : selectedChatGate.runtimeReady ? 'runtime ready, support gated' : 'exact row required'

  return (
    <section className="system-view cxv">
      <header className="cxv-head">
        <div className="cxv-head__copy">
          <p className="cxv-kicker"><IconSystem size={14} /> System</p>
          <h1>Runtime &amp; API</h1>
          <p className="cxv-sub">The operational view — runtime health, model readiness, and developer connection details in one calm place. Support always comes from /api/capabilities and COMPATIBILITY.md, never inferred from runtime health alone.</p>
        </div>
        <div className="cxv-head__actions">
          <StatusDot tone={runtimeTone} pulse={runtime?.generation_ready} label={runtimePill} />
        </div>
      </header>

      {runtime?.status === 'offline' && (
        <EmptyState
          className="cx-empty--inline"
          icon={<IconSystem size={22} />}
          title="Backend unreachable"
          description={`Nothing answered at ${runtime?.api_base || 'the configured API base'}. Start the local runtime (cargo run -- serve) or fix the API base in Settings; runtime health and the support gate below stay unknown until /v1/health responds.`}
        />
      )}

      <div className="cxv-stat-grid">
        <div className="cxv-stat"><span>Runtime</span><strong>{runtimeState}</strong><small>{runtime?.engine || 'engine unknown'}</small></div>
        <div className="cxv-stat"><span>Generation</span><strong>{runtime?.generation_ready ? 'Yes' : 'No'}</strong><small>{runtimePill}</small></div>
        <div className="cxv-stat"><span>Loaded model</span><strong>{runtime?.loaded_now ? 'Active' : 'None'}</strong><small title={runtime?.loaded_now ? runtime?.active_model_id : 'Nothing loaded'}>{runtime?.loaded_now ? runtime?.active_model_id : 'Nothing loaded'}</small></div>
        <div className="cxv-stat"><span>Selected gate</span><strong>{gateStat}</strong><small>{gateStatSub}</small></div>
        <div className="cxv-stat"><span>Local API</span><strong>{runtime?.api_base ? 'Online' : 'Offline'}</strong><small>{apiBase}</small></div>
      </div>

      <div className="cxv-grid cxv-grid--two">
        <section className="cxv-card cxv-panel">
          <div className="cxv-section__head"><h2>Runtime</h2><span className="cxv-section__count">local engine</span></div>
          <div className="sys-defs">
            <div><span>Runtime state</span><strong>{runtime?.generation_ready ? 'Generation-ready' : runtime?.loaded_now ? 'Loaded, not generation-ready' : runtime?.status === 'offline' ? 'Backend offline' : 'Online, no model loaded'}</strong></div>
            <div><span>Local engine</span><strong>{runtime?.engine || 'Unknown'}</strong></div>
            <div><span>Loaded model</span><strong>{runtime?.loaded_now ? runtime?.active_model_id : 'Nothing loaded'}</strong></div>
            <div><span>Generation ready</span><strong>{runtime?.generation_ready ? 'Yes' : 'No'}</strong></div>
            <div><span>Selected device at load</span><strong>{execution.device}</strong></div>
            <div><span>Selected backend at load</span><strong>{execution.backend}</strong></div>
            <div><span>Selected exact-row gate</span><strong>{selectedExactRowReady ? 'Ready for chat/API' : selectedChatGate.runtimeReady ? 'Runtime ready; support gated' : selectedChatGate.label}</strong></div>
            <div><span>Q8 storage</span><strong>{q8RuntimeLabel}</strong></div>
            <div><span>Next chat selection</span><strong>{selectedModelName}</strong></div>
            <div><span>API base</span><strong>{apiBase}</strong></div>
          </div>
        </section>

        <section className="cxv-card cxv-panel">
          <div className="cxv-section__head"><h2>Handling locally</h2><span className="cxv-section__count">on-device</span></div>
          <ul className="sys-feed">
            <li>Persistent conversations are already available from local storage.</li>
            <li>Saved memory remains on-device and can be recalled in later chats.</li>
            <li>{execution.summary}</li>
            <li>Q8 runtime policy: {q8RuntimeDetail}. {q8Runtime?.note || ''}</li>
            <li>Current next-chat model state: {describeModelState(selectedModel)}</li>
            <li>Chat stays blocked until loaded_now=true, generation_ready=true, active_model_id matches the selected local GGUF, and COMPATIBILITY.md plus /api/capabilities expose an exact supported row.</li>
            <li>The standard /v1-compatible local API is exposed at {apiBase}.</li>
          </ul>
        </section>
      </div>

      <section className="cxv-card cxv-panel">
        <div className="cxv-section__head">
          <h2>Local API access</h2>
          <StatusDot tone={selectedExactRowReady ? 'ready' : 'warn'} label={endpointReadinessLabel} />
        </div>
        <p className="cxv-sub">Use the same local runtime through standard /v1-compatible endpoints for apps, scripts, and quick terminal checks.</p>
        <div className="sys-endpoints">
          <div className="sys-endpoint">
            <div className="sys-endpoint__head"><strong>Chat completions</strong><span className="cxv-tag">POST</span></div>
            <code>{runtime?.api_base ? `${runtime.api_base}/v1/chat/completions` : 'Unavailable until the local API is running'}</code>
            <p>{chatCompletionsCopy}</p>
          </div>
          <div className="sys-endpoint">
            <div className="sys-endpoint__head"><strong>Models</strong><span className="cxv-tag">GET</span></div>
            <code>{runtime?.api_base ? `${runtime.api_base}/v1/models` : 'Unavailable until the local API is running'}</code>
            <p>Lists the currently loaded runtime model; this is not a broad model catalog.</p>
          </div>
          <div className="sys-endpoint">
            <div className="sys-endpoint__head"><strong>Health</strong><span className="cxv-tag">GET</span></div>
            <code>{runtime?.api_base ? `${runtime.api_base}/v1/health` : 'Unavailable until the local API is running'}</code>
            <p>Source of truth for active_model_id, generation_ready, and the selected load-time execution plan.</p>
          </div>
          <div className="sys-endpoint">
            <div className="sys-endpoint__head"><strong>Capabilities</strong><span className="cxv-tag">GET</span></div>
            <code>{runtime?.api_base ? `${runtime.api_base}/api/capabilities` : 'Unavailable until the local API is running'}</code>
            <p>Support contract for model families, quantization, API features, and compatibility rows.</p>
          </div>
        </div>
        <div className="sys-curl">
          <div className="sys-curl__head"><strong>Readiness-gated request</strong><span className="cxv-tag">curl</span></div>
          <pre>{runtime?.api_base ? curlExample : 'Start the local runtime to see an exact-row readiness check.'}</pre>
        </div>
      </section>

      <section className="cxv-card cxv-panel">
        <div className="cxv-section__head"><h2>Support contract</h2><span className="cxv-section__count">evidence-based</span></div>
        <p className="cxv-sub">This mirrors /api/capabilities so the UI never implies unvalidated model families, quantization formats, or API features.</p>

        <div className="cxv-grid cxv-grid--two">
          <div className="cxv-card cxv-card--flat sys-evidence">
            <strong>Current contract</strong>
            {supportContract ? (
              <>
                <div className="sys-contract-overview">
                  <span><b>{supportedCompatibilityCount}</b> supported exact rows</span>
                  <span><b>{compatibilityTargets.length - supportedCompatibilityCount}</b> guarded or unclaimed rows</span>
                </div>
                <details className="sys-evidence-details sys-evidence-details--canonical">
                  <summary>Read the complete current-gate statement</summary>
                  <CanonicalStatement text={supportContractCurrentGate} />
                </details>
                <dl className="sys-policy-list">
                  <div><dt>Support policy</dt><dd>{supportContract.support_policy}</dd></div>
                  <div><dt>Unsupported policy</dt><dd>{supportContract.unsupported_policy}</dd></div>
                </dl>
              </>
            ) : (
              <p>/api/capabilities is unavailable, so Camelid falls back to health/model readiness only and does not infer broader support.</p>
            )}
          </div>

          <div className="cxv-card cxv-card--flat sys-evidence">
            <strong>Selected exact-row evidence</strong>
            {selectedCompatibilityTarget ? (
              <>
                <code className="a-code">{selectedCompatibilityTarget.id}</code>
                <p>{formatCapabilityStatus(selectedCompatibilityTarget.status)} · {selectedCompatibilityTarget.family} · {selectedCompatibilityTarget.quantization}</p>
                <p><b>Readiness gate:</b> {displayCapabilityCopy(selectedCompatibilityTarget.frontend_readiness_gate || 'not advertised')}</p>
                <p><b>Endpoint/chat gate:</b> {selectedExactRowReady ? 'Ready: runtime readiness and exact-row support both match.' : `${selectedChatGate.label}; loaded_now=${selectedChatGate.runtimeLoaded ? 'true' : 'false'}, generation_ready=${selectedChatGate.runtimeGenerationReady ? 'true' : 'false'}, exact row supported=${selectedChatGate.contractSupported ? 'true' : 'false'}.`}</p>
                {selectedSupportLanes.map((lane) => (
                  <p key={lane.key}><b>{supportLaneTitle(lane)}:</b> {lane.label}. {displayCapabilityCopy(lane.copy)}</p>
                ))}
                <p>{displayCapabilityCopy(selectedCompatibilityTarget.evidence || selectedCompatibilityTarget.next_step || 'No row evidence advertised.')}</p>
              </>
            ) : (
              <p>No selected model exact row matched, so System will not promote runtime health, families, or quant lists into support.</p>
            )}
          </div>
        </div>

        <div className="cxv-grid cxv-grid--two">
          <div className="cxv-card cxv-card--flat sys-evidence">
            <strong>Exact-row quant evidence</strong>
            <ExactRowEvidenceSummary targets={compatibilityTargets} field="quantization" />
            <p>Quant labels here come only from compatibility rows and do not unlock chat without a matching loaded model.</p>
          </div>
          <div className="cxv-card cxv-card--flat sys-evidence">
            <strong>Exact-row family evidence</strong>
            <ExactRowEvidenceSummary targets={compatibilityTargets} field="family" />
            <p>Family labels remain row-scoped evidence boundaries, not broad support for neighboring files.</p>
          </div>
        </div>

        <div className="cxv-card cxv-card--flat sys-evidence">
          <strong>Validated API features</strong>
          {supportedFeatures.length ? (
            <div className="sys-rows">
              {supportedFeatures.map((feature) => (
                <div key={feature.id} className="sys-row">
                  <div className="sys-row__head">
                    <span>{displayCapabilityId(feature.id)}</span>
                    <EvidenceChip status={feature.status} source={{ rowId: feature.id }} size="sm" />
                  </div>
                  <small>{displayCapabilityCopy(feature.notes)}</small>
                </div>
              ))}
            </div>
          ) : (
            <p>No supported API feature rows advertised yet.</p>
          )}
        </div>
      </section>

      <details className="cxv-disclosure">
        <summary>Full compatibility evidence — every row and guarded feature from /api/capabilities</summary>
        <div className="cxv-disclosure__body">
          <div className="sys-rows-block">
            <strong>COMPATIBILITY.md rows from /api/capabilities</strong>
            {compatibilityTargets.length ? (
              <div className="sys-rows">
                {compatibilityTargets.map((target) => (
                  <div key={target.id} className="sys-row">
                    <div className="sys-row__head">
                      <span>{target.family} · {target.quantization}</span>
                      <span className="sys-row__claims">
                        <span className="sys-row__meta">{target.id}</span>
                        <EvidenceChip status={target.status} source={{ rowId: target.id, detail: `${target.family} · ${target.quantization}` }} size="sm" />
                      </span>
                    </div>
                    <small>Metadata: {formatCapabilityStatus(target.metadata_parses)} · tokenizer: {formatCapabilityStatus(target.tokenizer_works)} · tensors: {formatCapabilityStatus(target.tensors_load)} · generation: {formatCapabilityStatus(target.generation_runs)}</small>
                    <small>{exactRowSupportLanes(target, apiFeatures).map((lane) => `${supportLaneTitle(lane).replace(' readiness', '')}: ${lane.label}`).join(' · ')}</small>
                    <small>{displayCapabilityCopy(rowSupportNextStepCopy(target, apiFeatures))}</small>
                  </div>
                ))}
              </div>
            ) : (
              <p className="cxv-sub">No compatibility rows advertised yet.</p>
            )}
          </div>

          <div className="sys-rows-block">
            <strong>Unsupported / partial API features</strong>
            {unsupportedFeatures.length ? (
              <div className="sys-rows">
                {unsupportedFeatures.map((feature) => (
                  <div key={feature.id} className="sys-row">
                    <div className="sys-row__head">
                      <span>{displayCapabilityId(feature.id)}</span>
                      <EvidenceChip status={feature.status} source={{ rowId: feature.id }} size="sm" />
                    </div>
                    <small>{displayCapabilityCopy(guardedCapabilityCopy(feature, 'System/API controls'))}</small>
                  </div>
                ))}
              </div>
            ) : (
              <p className="cxv-sub">No unsupported or partial API rows advertised.</p>
            )}
          </div>
        </div>
      </details>
    </section>
  )
}
