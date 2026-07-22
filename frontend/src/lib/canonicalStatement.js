function pushSegment(segments, text, start, end) {
  const segment = text.slice(start, end).trim()
  if (segment) segments.push(segment)
}

export function splitCanonicalStatement(value = '') {
  const text = String(value).trim()
  if (!text) return []

  const segments = []
  let depth = 0
  let start = 0

  for (let index = 0; index < text.length; index += 1) {
    const character = text[index]
    if (character === '(') depth += 1
    if (character === ')' && depth > 0) depth -= 1
    if (depth !== 0) continue

    const nextCharacter = text[index + 1]
    const afterWhitespace = text.slice(index + 1).match(/^\s+([A-Z])/)
    const isTopLevelClause = character === ';' && /\s/.test(nextCharacter || '')
    const isSentenceEnd = /[.!?]/.test(character) && Boolean(afterWhitespace)
    if (!isTopLevelClause && !isSentenceEnd) continue

    pushSegment(segments, text, start, index + 1)
    start = index + 1
  }

  pushSegment(segments, text, start, text.length)
  return segments
}

export function canonicalStatementLabel(segment, index) {
  if (index === 0 && /^Current exact-row support:/i.test(segment)) return 'Current gate'
  return 'Contract statement'
}