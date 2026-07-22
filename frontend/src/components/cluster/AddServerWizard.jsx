import { useState } from 'react'
import { Modal } from '../ui/Modal'
import { Button } from '../ui/Button'
import { Field } from '../ui/Field'
import { Chip } from '../ui/Chip'
import { DeviceIcon } from './DeviceIcon'
import { CONNECTION_METHODS, NODE_ROLES, NODE_TYPES, NODE_TYPE_BY, createNode } from '../../lib/clusterModel'
import { probeNode } from '../../lib/devCluster'
import { IconCheck, IconClose, IconChevronRight, IconWarning } from '../ui/icons'

const STEPS = ['Device', 'Connection', 'Role', 'Resources', 'Authentication', 'Test & add']
const DEFAULT_PORT = { ssh: 22, winrm: 5985, agent: 8181, manual: '' }

function emptyDraft() {
  return {
    node_type: 'mac',
    display_name: '',
    hostname: '',
    ip_address: '',
    port: 22,
    connection_method: 'ssh',
    roles: ['worker'],
    cpu_cores: '',
    ram_gb: '',
    gpu: '',
    vram: '',
    os: '',
    arch: '',
    tags: '',
    auth: { username: '', method: 'ssh-key', key_path: '', password: '' },
  }
}

export function AddServerWizard({ open, onClose, onAdd, initial }) {
  const [step, setStep] = useState(0)
  const [draft, setDraft] = useState(() => ({ ...emptyDraft(), ...initial }))
  const [test, setTest] = useState(null) // { running, checks:[], verdict }
  const set = (patch) => setDraft((d) => ({ ...d, ...patch }))

  const pickType = (type) => {
    const method = NODE_TYPE_BY[type]?.defaultMethod || 'ssh'
    set({ node_type: type, connection_method: method, port: DEFAULT_PORT[method] || draft.port, arch: type === 'raspberrypi' ? 'arm64' : draft.arch })
    setStep(1)
  }
  const toggleRole = (role) => set({ roles: draft.roles.includes(role) ? draft.roles.filter((r) => r !== role) : [...draft.roles, role] })

  const canNext = () => {
    if (step === 1) return Boolean(draft.display_name.trim() && (draft.hostname.trim() || draft.ip_address.trim()))
    if (step === 2) return draft.roles.length > 0
    return true
  }

  const runChecks = async () => {
    const host = draft.hostname.trim() || draft.ip_address.trim()
    setTest({ running: true, checks: [], verdict: null })
    const checks = []
    const probe = await probeNode({ host, port: Number(draft.port) || 22 })
    if (!probe.available) {
      checks.push({ label: 'Reachability', state: 'skip', note: 'needs the local dev server (npm run dev)' })
    } else if (probe.reachable) {
      checks.push({ label: 'Reachability', state: 'ok', note: `${Math.round(probe.latencyMs)}ms` })
    } else {
      checks.push({ label: 'Reachability', state: 'fail', note: `${host}:${draft.port} unreachable` })
    }
    checks.push({ label: 'OS / CPU / RAM', state: draft.os || draft.cpu_cores ? 'ok' : 'skip', note: draft.os || draft.cpu_cores ? 'from your input' : 'auto-detect needs the agent' })
    checks.push({ label: 'Worker binary', state: 'skip', note: 'verified once the agent connects' })
    checks.push({ label: 'Network latency', state: probe.reachable ? 'ok' : 'skip', note: probe.reachable ? `${Math.round(probe.latencyMs)}ms` : 'not measured' })
    const verdict = checks.some((c) => c.state === 'fail') ? 'failed' : checks.some((c) => c.state === 'skip') ? 'warnings' : 'ready'
    setTest({ running: false, checks, verdict })
  }

  const finish = (forceManual) => {
    const node = createNode({
      display_name: draft.display_name.trim() || 'New machine',
      node_type: draft.node_type,
      hostname: draft.hostname.trim(),
      ip_address: draft.ip_address.trim() || null,
      port: draft.port ? Number(draft.port) : null,
      connection_method: forceManual ? 'manual' : draft.connection_method,
      roles: draft.roles.length ? draft.roles : ['worker'],
      status: test?.verdict === 'ready' ? 'online' : 'unknown',
      os: draft.os.trim() || null,
      arch: draft.arch.trim() || null,
      cpu_cores: draft.cpu_cores ? Number(draft.cpu_cores) : null,
      ram_gb: draft.ram_gb ? Number(draft.ram_gb) : null,
      gpu: draft.gpu.trim() || null,
      vram: draft.vram.trim() || null,
      tags: draft.tags.split(',').map((t) => t.trim()).filter(Boolean),
      auth: { username: draft.auth.username.trim(), method: draft.auth.method, key_path: draft.auth.key_path.trim(), saved: false },
    })
    onAdd(node)
    reset()
  }

  const reset = () => { setStep(0); setDraft(emptyDraft()); setTest(null); onClose() }

  return (
    <Modal open={open} onClose={reset} title="Add a server" size="lg" labelledById="cluster-wizard-title">
      <div className="cluster-wizard">
        <ol className="cluster-wizard__steps">
          {STEPS.map((label, i) => (
            <li key={label} className={`${i === step ? 'is-active' : ''} ${i < step ? 'is-done' : ''}`}>
              <span>{i < step ? <IconCheck size={13} /> : i + 1}</span>{label}
            </li>
          ))}
        </ol>

        <div className="cluster-wizard__body">
          {step === 0 && (
            <>
              <p className="cluster-wizard__lead">What kind of machine are you adding?</p>
              <div className="cluster-type-grid">
                {NODE_TYPES.map((t) => (
                  <button key={t.value} type="button" className={`cluster-type-card ${draft.node_type === t.value ? 'is-active' : ''}`} onClick={() => pickType(t.value)}>
                    <span className="cluster-type-card__icon"><DeviceIcon type={t.value} size={28} /></span>
                    <strong>{t.label}</strong>
                  </button>
                ))}
              </div>
            </>
          )}

          {step === 1 && (
            <div className="cluster-form">
              <Field label="Display name"><input autoFocus value={draft.display_name} onChange={(e) => set({ display_name: e.target.value })} placeholder="e.g. Studio Mac, Pi worker 1" /></Field>
              <div className="cluster-form__pair">
                <Field label="Hostname or IP address"><input value={draft.hostname} onChange={(e) => set({ hostname: e.target.value })} placeholder="192.0.2.20 or mac-studio.local" /></Field>
                <Field label="Port"><input type="number" value={draft.port} onChange={(e) => set({ port: e.target.value })} /></Field>
              </div>
              <div className="cluster-form__group">
                <span className="cluster-form__label">Connection method</span>
                <div className="cluster-segmented">
                  {CONNECTION_METHODS.map((m) => (
                    <button key={m.value} type="button" className={draft.connection_method === m.value ? 'is-on' : ''} onClick={() => set({ connection_method: m.value, port: DEFAULT_PORT[m.value] || draft.port })}>{m.label}</button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {step === 2 && (
            <div className="cluster-roles">
              <p className="cluster-wizard__lead">Pick one or more roles for this machine.</p>
              {NODE_ROLES.map((r) => (
                <button key={r.value} type="button" className={`cluster-role-card ${draft.roles.includes(r.value) ? 'is-on' : ''}`} onClick={() => toggleRole(r.value)}>
                  <span className="cluster-role-card__check">{draft.roles.includes(r.value) && <IconCheck size={14} />}</span>
                  <div><strong>{r.label}</strong><small>{r.desc}</small></div>
                </button>
              ))}
            </div>
          )}

          {step === 3 && (
            <div className="cluster-form">
              <div className="cluster-form__pair">
                <Field label="CPU cores"><input type="number" value={draft.cpu_cores} onChange={(e) => set({ cpu_cores: e.target.value })} placeholder="e.g. 10" /></Field>
                <Field label="RAM (GB)"><input type="number" value={draft.ram_gb} onChange={(e) => set({ ram_gb: e.target.value })} placeholder="e.g. 32" /></Field>
              </div>
              <div className="cluster-form__pair">
                <Field label="GPU / accelerator"><input value={draft.gpu} onChange={(e) => set({ gpu: e.target.value })} placeholder="Apple M-series, RTX 4090…" /></Field>
                <Field label="VRAM / unified memory"><input value={draft.vram} onChange={(e) => set({ vram: e.target.value })} placeholder="e.g. 24 GB" /></Field>
              </div>
              <div className="cluster-form__pair">
                <Field label="OS"><input value={draft.os} onChange={(e) => set({ os: e.target.value })} placeholder="macOS 15, Ubuntu 24.04…" /></Field>
                <Field label="Architecture">
                  <select value={draft.arch} onChange={(e) => set({ arch: e.target.value })}>
                    <option value="">Unknown</option>
                    <option value="arm64">arm64</option>
                    <option value="x86_64">x86_64</option>
                  </select>
                </Field>
              </div>
              <Field label="Tags (comma separated)"><input value={draft.tags} onChange={(e) => set({ tags: e.target.value })} placeholder="studio, fast, gpu" /></Field>
              <p className="cluster-wizard__hint">You can fill these in now or let the agent auto-detect them later.</p>
            </div>
          )}

          {step === 4 && (
            <div className="cluster-form">
              {draft.connection_method === 'winrm' ? (
                <>
                  <Field label="Username"><input value={draft.auth.username} onChange={(e) => set({ auth: { ...draft.auth, username: e.target.value } })} /></Field>
                  <Field label="Auth method">
                    <select value={draft.auth.method} onChange={(e) => set({ auth: { ...draft.auth, method: e.target.value } })}>
                      <option value="winrm-ntlm">WinRM (NTLM)</option>
                      <option value="winrm-kerberos">WinRM (Kerberos)</option>
                      <option value="local-agent">Local agent</option>
                    </select>
                  </Field>
                </>
              ) : (
                <>
                  <Field label="Username"><input value={draft.auth.username} onChange={(e) => set({ auth: { ...draft.auth, username: e.target.value } })} placeholder="e.g. tim" /></Field>
                  <Field label="SSH key path"><input value={draft.auth.key_path} onChange={(e) => set({ auth: { ...draft.auth, key_path: e.target.value } })} placeholder="~/.ssh/id_ed25519" /></Field>
                  <Field label="Password (optional, not stored)"><input type="password" value={draft.auth.password} onChange={(e) => set({ auth: { ...draft.auth, password: e.target.value } })} /></Field>
                </>
              )}
              <p className="cluster-wizard__hint cluster-wizard__hint--secure"><IconWarning size={14} /> Secrets are never written to config — only connection details are saved, and auth is marked “not saved” until secure OS keychain storage is wired.</p>
            </div>
          )}

          {step === 5 && (
            <div className="cluster-test">
              {!test && <p className="cluster-wizard__lead">Run a quick reachability check, or add the node now (you can test later).</p>}
              {test && (
                <ul className="cluster-test__checks">
                  {test.checks.map((c) => (
                    <li key={c.label} className={`is-${c.state}`}>
                      <span className="cluster-test__icon">{c.state === 'ok' ? <IconCheck size={14} /> : c.state === 'fail' ? <IconClose size={14} /> : <IconWarning size={13} />}</span>
                      <strong>{c.label}</strong>
                      <small>{c.note}</small>
                    </li>
                  ))}
                </ul>
              )}
              {test?.verdict && (
                <div className={`cluster-test__verdict is-${test.verdict}`}>
                  {test.verdict === 'ready' ? 'Ready to add.' : test.verdict === 'warnings' ? 'Added with warnings — some checks need the local agent.' : 'Connection failed — you can still add it as a manual/offline node.'}
                </div>
              )}
            </div>
          )}
        </div>

        <div className="cluster-wizard__footer">
          <Button variant="ghost" onClick={() => (step === 0 ? reset() : setStep((s) => s - 1))}>{step === 0 ? 'Cancel' : 'Back'}</Button>
          <div className="cluster-wizard__footer-right">
            {step === 5 ? (
              <>
                <Button variant="tonal" loading={test?.running} onClick={runChecks}>Run checks</Button>
                {test?.verdict === 'failed'
                  ? <Button variant="primary" onClick={() => finish(true)}>Add as manual node</Button>
                  : <Button variant="primary" onClick={() => finish(false)}>Add node</Button>}
              </>
            ) : step === 0 ? null : (
              <Button variant="primary" iconRight={<IconChevronRight size={15} />} disabled={!canNext()} onClick={() => setStep((s) => s + 1)}>Next</Button>
            )}
          </div>
        </div>
      </div>
    </Modal>
  )
}

export default AddServerWizard
