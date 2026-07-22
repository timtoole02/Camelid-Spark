import { IconMac, IconWindows, IconLinux, IconRaspberryPi, IconServer } from '../ui/icons'

const MAP = { mac: IconMac, windows: IconWindows, linux: IconLinux, raspberrypi: IconRaspberryPi, other: IconServer }

export function DeviceIcon({ type, size = 20 }) {
  const Icon = MAP[type] || IconServer
  return <Icon size={size} />
}

export default DeviceIcon
