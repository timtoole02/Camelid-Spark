import { useEffect, useState } from 'react'
import { EvidenceChip } from '../ui/EvidenceChip'
import { UnsupportedBlocker } from './UnsupportedBlocker'
import { quantLabelFromGgufFileType } from '../../lib/capabilities'

/* Model inspector (Phase 3): a metadata drawer for the LOADED model, fed by
   GET /api/models/current + /api/models/tokenizer at open time.

   Everything here is descriptive metadata — explicitly NOT support evidence
   (I2/I4); the banner chip says so. The local file path renders here because
   this is a local-only operator surface; exports never include it (I7 covers
   shareable surfaces, which this is not). */

const summarizeValue = (value) => {
  if (Array.isArray(value)) {
    const preview = value.slice(0, 4).map((item) => String(item)).join(', ')
    return value.length > 4 ? `[${preview}, … ${value.length.toLocaleString()} items]` : `[${preview}]`
  }
  const text = String(value)
  return text.length > 160 ? `${text.slice(0, 160)}… (${text.length.toLocaleString()} chars)` : text
}

const metadataRows = (metadata = {}) =>
  Object.entries(metadata)
    .map(([key, value]) => [key, summarizeValue(value)])
    .sort(([a], [b]) => a.localeCompare(b))

const contextLengthFrom = (metadata = {}) => {
  const arch = metadata['general.architecture']
  return metadata[`${arch}.context_length`] ?? null
}

export function ModelInspector({ apiBase, onClose }) {
  const [state, setState] = useState({ loading: true, error: null, current: null, tokenizer: null })

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const base = (apiBase || '').replace(/\/$/, '')
        const [currentRes, tokenizerRes] = await Promise.all([
          fetch(`${base}/api/models/current`),
          fetch(`${base}/api/models/tokenizer`),
        ])
        const current = currentRes.ok ? await currentRes.json() : null
        const tokenizer = tokenizerRes.ok ? await tokenizerRes.json() : null
        if (!cancelled) {
          setState({
            loading: false,
            error: current ? null : 'No loaded model to inspect — load a local GGUF first.',
            current,
            tokenizer,
          })
        }
      } catch (error) {
        if (!cancelled) setState({ loading: false, error: `Backend did not answer: ${error.message}`, current: null, tokenizer: null })
      }
    }
    load()
    return () => { cancelled = true }
  }, [apiBase])

  const { loading, error, current, tokenizer } = state
  const metadata = current?.gguf?.metadata || {}
  const fileType = metadata['general.file_type']
  const quant = fileType !== undefined ? (quantLabelFromGgufFileType(fileType) || `file_type ${fileType}`) : null
  const contextLength = contextLengthFrom(metadata)
  const rows = metadataRows(metadata)

  return (
    <div className="model-inspector-overlay" role="dialog" aria-modal="true" aria-label="Model inspector" onClick={(event) => { if (event.target === event.currentTarget) onClose() }}>
      <aside className="model-inspector">
        <header className="model-inspector__head">
          <div>
            <h2>Model inspector</h2>
            {current && <code className="model-inspector__id">{current.id}</code>}
          </div>
          <button type="button" className="cxturn__action" onClick={onClose}>Close</button>
        </header>

        <EvidenceChip
          state="neutral"
          label="descriptive metadata — not support evidence"
          source={{ note: 'GGUF key/value metadata and tokenizer details describe the file. Support comes only from the exact /api/capabilities row plus runtime readiness; nothing in this drawer widens it.' }}
          size="sm"
          className="model-inspector__banner"
        />

        {loading && <p className="model-inspector__note">Reading /api/models/current…</p>}
        {error && <p className="model-inspector__note">{error}</p>}

        {current?.unsupported_runtime && (
          <UnsupportedBlocker blocker={current.unsupported_runtime} />
        )}

        {current && (
          <>
            <section className="model-inspector__section">
              <h3>File</h3>
              <dl className="model-inspector__grid">
                <div><dt>path</dt><dd><code>{current.path}</code> <span className="model-inspector__hint">local-only display; never exported</span></dd></div>
                <div><dt>gguf version</dt><dd>{current.gguf?.version}</dd></div>
                <div><dt>quant</dt><dd>{quant || 'unknown'}</dd></div>
                <div><dt>tensors</dt><dd>{current.gguf?.tensor_count?.toLocaleString()} tensors · data offset {current.gguf?.data_start_offset?.toLocaleString()} · align {current.gguf?.alignment}</dd></div>
                <div><dt>context length</dt><dd>{contextLength ? `${Number(contextLength).toLocaleString()} (model-native metadata; checked context support comes only from bounded packs on the contract row)` : 'not present in metadata'}</dd></div>
                <div><dt>runtime</dt><dd>config {current.llama_config ? 'parsed' : '—'} · tensors {current.llama_tensors ? 'bound' : '—'}{current.lane ? ` · lane ${current.lane}` : ''}</dd></div>
              </dl>
            </section>

            {tokenizer && (
              <section className="model-inspector__section">
                <h3>Tokenizer</h3>
                <dl className="model-inspector__grid">
                  <div><dt>model</dt><dd>{tokenizer.model}</dd></div>
                  <div><dt>vocab</dt><dd>{tokenizer.token_count?.toLocaleString()} tokens{tokenizer.byte_token_count ? ` · ${tokenizer.byte_token_count} byte tokens` : ''}</dd></div>
                  <div><dt>specials</dt><dd><code>{Object.entries(tokenizer.special || {}).filter(([, v]) => v !== null && !Array.isArray(v)).map(([k, v]) => `${k}=${v}`).join(' ')}</code></dd></div>
                  <div><dt>config</dt><dd><code>{Object.entries(tokenizer.config || {}).map(([k, v]) => `${k}=${v}`).join(' ')}</code></dd></div>
                </dl>
              </section>
            )}

            <section className="model-inspector__section">
              <h3>GGUF metadata <span className="model-inspector__hint">({rows.length} keys; long values summarized)</span></h3>
              <dl className="model-inspector__grid model-inspector__grid--kv">
                {rows.map(([key, value]) => (
                  <div key={key}><dt>{key}</dt><dd>{value}</dd></div>
                ))}
              </dl>
            </section>
          </>
        )}
      </aside>
    </div>
  )
}

export default ModelInspector
