import { displayCapabilityCopy, formatCapabilityStatus, isSupportedCapabilityStatus } from '../../lib/capabilities'

function groupExactRows(targets, field) {
  const groups = new Map()
  targets
    .filter((target) => target?.id && target?.[field])
    .forEach((target) => {
      const label = displayCapabilityCopy(target[field])
      const rows = groups.get(label) || []
      rows.push(target)
      groups.set(label, rows)
    })
  return Array.from(groups, ([label, rows]) => ({ label, rows }))
}

export function ExactRowEvidenceSummary({ targets = [], field, fallback = 'No exact compatibility rows advertised by this backend.' }) {
  const groups = groupExactRows(targets, field)
  if (!groups.length) return <p>{fallback}</p>

  const rowCount = groups.reduce((count, group) => count + group.rows.length, 0)
  return (
    <>
      <ul className="sys-evidence-groups">
        {groups.map((group) => (
          <li key={group.label}>
            <strong>{group.label}</strong>
            <span>
              {group.rows.filter((row) => isSupportedCapabilityStatus(row.status)).length} supported
              {' · '}
              {group.rows.filter((row) => !isSupportedCapabilityStatus(row.status)).length} guarded
            </span>
          </li>
        ))}
      </ul>
      <details className="sys-evidence-details">
        <summary>View all {rowCount} row mappings</summary>
        <div className="sys-evidence-mappings">
          {groups.map((group) => (
            <section key={group.label}>
              <h3>{group.label}</h3>
              <ul>
                {group.rows.map((target) => (
                  <li key={target.id}>
                    <code>{target.id}</code>
                    <span>{formatCapabilityStatus(target.status)}</span>
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </div>
      </details>
    </>
  )
}