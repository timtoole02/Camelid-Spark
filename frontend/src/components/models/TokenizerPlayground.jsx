import { useState } from 'react'
import { EvidenceChip } from '../ui/EvidenceChip'

/* Tokenizer playground (Phase 3): drives the live encode/decode endpoints
   (api feature row `tokenizer_encode_decode`). Type text → token ids + pieces
   + a byte-exact round-trip check against the decoder. Works whenever the
   runtime has a tokenizer available — chat support is NOT required and using
   this proves nothing about generation support (the chip says so).

   Pieces come from decoding each id individually: BPE decode is a fixed
   id→bytes mapping, so per-id decode is faithful; the round-trip line uses the
   full sequence so any normalization differences still surface there. */

const PIECE_LIMIT = 200

export function TokenizerPlayground({ apiBase }) {
  const [text, setText] = useState('')
  const [addSpecial, setAddSpecial] = useState(true)
  const [parseSpecial, setParseSpecial] = useState(false)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const base = (apiBase || '').replace(/\/$/, '')

  const run = async () => {
    if (!text.trim() || busy) return
    setBusy(true)
    setError(null)
    try {
      const encodeRes = await fetch(`${base}/api/models/tokenizer/encode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, add_special: addSpecial, parse_special: parseSpecial }),
      })
      const encoded = await encodeRes.json()
      if (!encodeRes.ok) {
        setResult(null)
        setError(encoded?.error?.message || `encode failed (${encodeRes.status})`)
        return
      }
      const tokens = encoded.tokens || []

      const roundTripRes = await fetch(`${base}/api/models/tokenizer/decode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tokens }),
      })
      const roundTrip = roundTripRes.ok ? await roundTripRes.json() : null

      const pieceIds = tokens.slice(0, PIECE_LIMIT)
      const pieces = []
      const CHUNK = 16
      for (let i = 0; i < pieceIds.length; i += CHUNK) {
        const chunk = pieceIds.slice(i, i + CHUNK)
        const decoded = await Promise.all(chunk.map(async (id) => {
          const res = await fetch(`${base}/api/models/tokenizer/decode`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tokens: [id] }),
          })
          if (!res.ok) return '␀'
          const data = await res.json()
          return data.text ?? '␀'
        }))
        pieces.push(...decoded)
      }

      setResult({
        tokens,
        tokenCount: encoded.token_count ?? tokens.length,
        pieces,
        piecesTruncated: tokens.length > PIECE_LIMIT,
        roundTripText: roundTrip?.text ?? null,
        /* Byte-exact comparison vs the typed text. add_special tokens decode
           into the output, so exact match is only expected when they are off —
           the verdict copy explains rather than pretends. */
        roundTripExact: roundTrip?.text === text,
      })
    } catch (err) {
      setResult(null)
      setError(`Backend did not answer: ${err.message}`)
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="cxv-card cxv-panel tokenizer-playground" aria-label="Tokenizer playground">
      <div className="cxv-section__head">
        <h2>Tokenizer playground</h2>
        <EvidenceChip
          status="supported_current_gate"
          label="tokenizer_encode_decode"
          source={{ rowId: 'tokenizer_encode_decode', detail: 'Live encode/decode against the loaded tokenizer. Token output is descriptive — it does not widen generation support for any row.' }}
          size="sm"
        />
      </div>
      <p className="cxv-sub">Round-trip the loaded model&apos;s tokenizer: text → ids → pieces → text. Byte-exact round-trips are the same property the parity evidence leans on.</p>

      <div className="tokenizer-playground__controls">
        <textarea
          className="tokenizer-playground__input"
          rows={3}
          placeholder="Type text to tokenize against the loaded model…"
          value={text}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
              event.preventDefault()
              run()
            }
          }}
          aria-label="Text to tokenize"
        />
        <div className="tokenizer-playground__options">
          <label><input type="checkbox" checked={addSpecial} onChange={(e) => setAddSpecial(e.target.checked)} /> add_special</label>
          <label><input type="checkbox" checked={parseSpecial} onChange={(e) => setParseSpecial(e.target.checked)} /> parse_special</label>
          <button type="button" className="cxv-button" onClick={run} disabled={busy || !text.trim()}>
            {busy ? 'Tokenizing…' : 'Tokenize'}
          </button>
        </div>
      </div>

      {error && <p className="tokenizer-playground__error" role="status">{error} — load a model with an available tokenizer first.</p>}

      {result && (
        <div className="tokenizer-playground__result">
          <div className="tokenizer-playground__stat-row">
            <span className="tokenizer-playground__stat"><b>{result.tokenCount}</b> tokens</span>
            <span className={`tokenizer-playground__verdict ${result.roundTripExact ? 'is-exact' : ''}`}>
              round-trip {result.roundTripExact ? 'byte-exact ✓' : 'differs from input'}
            </span>
            {!result.roundTripExact && addSpecial && (
              <span className="tokenizer-playground__hint">expected with add_special on — special tokens decode into the output</span>
            )}
          </div>
          <ol className="tokenizer-playground__tokens" aria-label="Tokens">
            {result.tokens.slice(0, PIECE_LIMIT).map((id, index) => (
              <li key={`${index}-${id}`} className="tokenizer-playground__token" title={`token[${index}] id=${id}`}>
                <span className="tokenizer-playground__piece">{(result.pieces[index] ?? '').replace(/ /g, '␣') || '∅'}</span>
                <span className="tokenizer-playground__id">{id}</span>
              </li>
            ))}
          </ol>
          {result.piecesTruncated && (
            <p className="tokenizer-playground__hint">showing the first {PIECE_LIMIT} of {result.tokens.length} tokens; the round-trip check above used the full sequence</p>
          )}
          {result.roundTripText !== null && !result.roundTripExact && (
            <p className="tokenizer-playground__roundtrip"><b>decoded:</b> <code>{result.roundTripText.length > 400 ? `${result.roundTripText.slice(0, 400)}…` : result.roundTripText}</code></p>
          )}
        </div>
      )}
    </section>
  )
}

export default TokenizerPlayground
