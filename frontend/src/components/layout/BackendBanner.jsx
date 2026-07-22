import { Button } from '../ui/Button'
import { StatusDot } from '../ui/StatusDot'
import { IconPlay, IconSettings } from '../ui/icons'

/* Shown across the app whenever the backend is offline. One-click Start when the
   dev launch hook is available; otherwise points the user at Settings. */
export function BackendBanner({ backend, onOpenSettings }) {
  const canStart = backend.status.available
  const running = backend.status.running
  return (
    <div className="backend-banner" role="status" aria-live="polite">
      <StatusDot tone="offline" />
      <div className="backend-banner__copy">
        <strong>Camelid backend isn’t running</strong>
        <span>
          {canStart
            ? 'Start it to load a model and chat — no setup needed.'
            : 'Start it from a terminal, then this clears automatically.'}
        </span>
      </div>
      <div className="backend-banner__actions">
        {canStart && (
          <Button
            variant="primary"
            size="sm"
            icon={<IconPlay size={16} />}
            onClick={backend.start}
            loading={backend.starting}
            disabled={backend.starting || running}
          >
            {running ? 'Starting…' : 'Start Camelid'}
          </Button>
        )}
        <Button variant="ghost" size="sm" icon={<IconSettings size={16} />} onClick={onOpenSettings}>
          Settings
        </Button>
      </div>
    </div>
  )
}

export default BackendBanner
