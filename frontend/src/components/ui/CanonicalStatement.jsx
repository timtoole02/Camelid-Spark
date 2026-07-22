import { canonicalStatementLabel, splitCanonicalStatement } from '../../lib/canonicalStatement'

function StatementCopy({ text }) {
  const colonIndex = text.indexOf(':')
  if (colonIndex < 1 || colonIndex > 90) return <p>{text}</p>

  return (
    <p>
      <strong>{text.slice(0, colonIndex + 1)}</strong>
      <span>{text.slice(colonIndex + 1)}</span>
    </p>
  )
}

export function CanonicalStatement({ text }) {
  const statements = splitCanonicalStatement(text)
  if (!statements.length) return null

  return (
    <div className="canonical-statement">
      <ol className="canonical-statement__claims">
        {statements.map((statement, index) => (
          <li key={`${index}-${statement.slice(0, 24)}`}>
            <span className="canonical-statement__label">{canonicalStatementLabel(statement, index)}</span>
            <StatementCopy text={statement} />
          </li>
        ))}
      </ol>
      <details className="canonical-statement__raw">
        <summary>View as one canonical paragraph</summary>
        <p>{text}</p>
      </details>
    </div>
  )
}