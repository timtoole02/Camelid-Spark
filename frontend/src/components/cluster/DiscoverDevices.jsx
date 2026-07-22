import { useEffect, useState } from 'react'
import { Modal } from '../ui/Modal'
import { Button } from '../ui/Button'
import { Chip } from '../ui/Chip'
import { DeviceIcon } from './DeviceIcon'
import { IconCheck, IconRefresh, IconPlus } from '../ui/icons'

export function DiscoverDevices({ open, onClose, onDiscover, onAdd }) {
  const [state, setState] = useState({ loading: true, devices: [], available: false, method: '' })
  const [added, setAdded] = useState({})

  const run = async () => {
    setState((s) => ({ ...s, loading: true }))
    const result = await onDiscover()
    setState({ loading: false, devices: result.devices || [], available: result.available, method: result.method || '' })
  }

  useEffect(() => { if (open) { setAdded({}); run() } /* eslint-disable-next-line */ }, [open])

  const add = (device) => {
    onAdd({
      display_name: device.hostname || device.ip || 'Discovered device',
      node_type: device.node_type || 'other',
      hostname: device.hostname || '',
      ip_address: device.ip || null,
      os: device.os || null,
      status: 'unknown',
      roles: ['worker'],
    })
    setAdded((a) => ({ ...a, [device.ip || device.hostname]: true }))
  }

  return (
    <Modal open={open} onClose={onClose} title="Discover devices" size="md" labelledById="cluster-discover-title"
      footer={<><Button variant="ghost" onClick={onClose}>Done</Button><Button variant="tonal" icon={<IconRefresh size={15} />} loading={state.loading} onClick={run}>Rescan</Button></>}>
      <div className="cluster-discover">
        <p className="cluster-discover__lead">
          Safe local discovery only — this machine plus LAN neighbors from the ARP table. Nothing is added without your approval.
        </p>
        {!state.loading && !state.available && (
          <p className="cluster-discover__note">Live discovery needs the local dev server (npm run dev). You can still add machines manually with “Add Server”.</p>
        )}
        {state.loading ? (
          <div className="cluster-discover__loading">Scanning the local network…</div>
        ) : (
          <ul className="cluster-discover__list">
            {state.devices.length === 0 && <li className="cluster-discover__empty">No devices found. Try Rescan or add one manually.</li>}
            {state.devices.map((d) => {
              const key = d.ip || d.hostname
              const isAdded = added[key]
              return (
                <li key={key} className="cluster-discover__row">
                  <span className="cluster-discover__icon"><DeviceIcon type={d.node_type || 'other'} size={20} /></span>
                  <div className="cluster-discover__body">
                    <strong>{d.hostname || d.ip}</strong>
                    <small>{[d.ip, d.os, d.service].filter(Boolean).join(' · ')}</small>
                  </div>
                  <Chip tone={d.confidence === 'high' ? 'ready' : 'neutral'}>{d.confidence}</Chip>
                  {isAdded
                    ? <Button variant="ghost" size="sm" icon={<IconCheck size={15} />} disabled>Added</Button>
                    : <Button variant="tonal" size="sm" icon={<IconPlus size={15} />} onClick={() => add(d)}>Add</Button>}
                </li>
              )
            })}
          </ul>
        )}
      </div>
    </Modal>
  )
}

export default DiscoverDevices
