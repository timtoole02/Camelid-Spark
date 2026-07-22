/* ProofOverlay — Proof Mode: the verifiable identity of what just ran.
   Hashes and token ids come from telemetry (lane identity captured at model
   load, receipt sealed server-side). Receipt status is shown only when a
   receipt was really written. */

function shortHash(hash) {
  return hash ? `${hash.slice(0, 12)}…${hash.slice(-6)}` : '—'
}

export default function ProofOverlay({ store }) {
  const run = store.getRun()
  const shown = run.startedAtMs ? run : store.getLastRun() || run
  const receipt = shown.receipt
  const ids = shown.generatedTokenIds

  return (
    <div className="observatory-proof">
      <div className="observatory-proof__row">
        <span>model</span>
        <code>{shown.modelId || '—'}</code>
      </div>
      <div className="observatory-proof__row">
        <span>model hash</span>
        <code title={receipt?.ggufSha256 || ''}>{shortHash(receipt?.ggufSha256)}</code>
      </div>
      <div className="observatory-proof__row">
        <span>quantization</span>
        <code>{shown.quantization || '—'}</code>
      </div>
      <div className="observatory-proof__row">
        <span>architecture</span>
        <code>{shown.architecture || '—'}</code>
      </div>
      <div className="observatory-proof__row">
        <span>prompt tokens</span>
        <code>{shown.promptTokens || '—'}</code>
      </div>
      <div className="observatory-proof__row observatory-proof__row--ids">
        <span>generated ids</span>
        <code>{ids.length ? `[${ids.slice(-24).join(', ')}${ids.length > 24 ? ' …' : ''}]` : '—'}</code>
      </div>
      <div className={`observatory-proof__row observatory-proof__seal ${receipt ? 'is-sealed' : ''}`}>
        <span>receipt</span>
        {receipt ? (
          <code title={receipt.receiptId}>
            ✓ sealed · {receipt.reproducible ? 'reproducible' : 'recorded'} · {shortHash(receipt.receiptId)}
          </code>
        ) : (
          <code>none for this run (request with camelid_receipt)</code>
        )}
      </div>
    </div>
  )
}
