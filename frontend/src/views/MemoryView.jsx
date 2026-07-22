import { useMemo, useState } from 'react'
import { clampText, formatDate } from '../lib/formatters'
import { Button } from '../components/ui/Button'
import { EmptyState } from '../components/ui/EmptyState'
import { IconMemory, IconPin, IconCopy, IconEdit, IconTrash, IconChat, IconPlus, IconSearch } from '../components/ui/icons'

export default function MemoryView({
  memories,
  memorySearch,
  setMemorySearch,
  selectedConversation,
  latestAssistantMessage,
  saveToMemory,
  createMemory,
  updateMemory,
  deleteMemory,
  setTab,
}) {
  const [scopeFilter, setScopeFilter] = useState('all')
  const [showPinnedOnly, setShowPinnedOnly] = useState(false)
  const [newMemory, setNewMemory] = useState({ title: '', scope: 'General', body: '' })
  const [editingId, setEditingId] = useState(null)
  const [editDraft, setEditDraft] = useState({ title: '', scope: '', body: '' })
  const [pendingDeleteId, setPendingDeleteId] = useState(null)
  const [busyAction, setBusyAction] = useState('')

  const searchableMemories = useMemo(() => {
    if (!memorySearch.trim()) return memories
    const q = memorySearch.toLowerCase()
    return memories.filter((memory) =>
      memory.title.toLowerCase().includes(q)
      || memory.body.toLowerCase().includes(q)
      || memory.scope.toLowerCase().includes(q),
    )
  }, [memories, memorySearch])

  const availableScopes = useMemo(
    () => ['all', ...new Set(memories.map((memory) => memory.scope).filter(Boolean))],
    [memories],
  )

  const visibleMemories = useMemo(() => searchableMemories.filter((memory) => {
    if (showPinnedOnly && !memory.pinned) return false
    if (scopeFilter !== 'all' && memory.scope !== scopeFilter) return false
    return true
  }), [scopeFilter, searchableMemories, showPinnedOnly])

  const pinnedCount = memories.filter((memory) => memory.pinned).length
  const latestChatLabel = clampText(selectedConversation?.title?.trim() || 'Current chat', 40)
  const canSaveLatestReply = Boolean(selectedConversation && latestAssistantMessage?.content)

  const handleCreateMemory = async () => {
    if (busyAction) return
    setBusyAction('create')
    const saved = await createMemory(newMemory)
    if (saved) setNewMemory({ title: '', scope: newMemory.scope || 'General', body: '' })
    setBusyAction('')
  }

  const startEditing = (memory) => {
    setEditingId(memory.id)
    setEditDraft({ title: memory.title, scope: memory.scope, body: memory.body })
    setPendingDeleteId(null)
  }

  const cancelEditing = () => {
    if (busyAction) return
    setEditingId(null)
    setEditDraft({ title: '', scope: '', body: '' })
  }

  const handleSaveEdit = async (memoryId) => {
    if (busyAction) return
    setBusyAction(`edit:${memoryId}`)
    const saved = await updateMemory(memoryId, editDraft)
    if (saved) cancelEditing()
    setBusyAction('')
  }

  const handleTogglePin = async (memory) => {
    if (busyAction) return
    setBusyAction(`pin:${memory.id}`)
    await updateMemory(memory.id, { pinned: !memory.pinned }, { successMessage: memory.pinned ? 'Memory unpinned.' : 'Memory pinned.' })
    setBusyAction('')
  }

  const handleCopy = async (memory) => {
    try { await navigator.clipboard.writeText(`${memory.title}\n\n${memory.body}`) } catch { /* notices handled elsewhere */ }
  }

  const handleDelete = async (memoryId) => {
    if (busyAction) return
    if (pendingDeleteId !== memoryId) { setPendingDeleteId(memoryId); return }
    setBusyAction(`delete:${memoryId}`)
    const deleted = await deleteMemory(memoryId, { successMessage: 'Memory removed from local memory.' })
    if (deleted) {
      if (editingId === memoryId) cancelEditing()
      setPendingDeleteId(null)
    }
    setBusyAction('')
  }

  return (
    <section className="memory-view cxv">
      <header className="cxv-head">
        <div className="cxv-head__copy">
          <p className="cxv-kicker"><IconMemory size={14} /> Memory</p>
          <h1>Memory</h1>
          <p className="cxv-sub">Capture durable notes, keep the important ones pinned, and clean up stale context — all without leaving the app.</p>
        </div>
        <div className="cxv-stats">
          <div className="cxv-stat">
            <span>Memories</span>
            <strong>{memories.length}</strong>
            <small>{visibleMemories.length} visible now</small>
          </div>
          <div className="cxv-stat">
            <span>Pinned</span>
            <strong>{pinnedCount}</strong>
            <small>{Math.max(availableScopes.length - 1, 0)} scopes</small>
          </div>
        </div>
      </header>

      <div className="cxv-grid cxv-grid--two">
        <div className="cxv-card cxv-form">
          <div className="cxv-card__titles">
            <strong>Quick capture</strong>
            <span className="cxv-card__sub-plain">Save a fact, decision, or preference</span>
          </div>
          <input value={newMemory.title} onChange={(e) => setNewMemory((c) => ({ ...c, title: e.target.value }))} placeholder="Short title" />
          <input value={newMemory.scope} onChange={(e) => setNewMemory((c) => ({ ...c, scope: e.target.value }))} placeholder="Scope (e.g. General, Project)" />
          <textarea value={newMemory.body} onChange={(e) => setNewMemory((c) => ({ ...c, body: e.target.value }))} placeholder="What should Camelid remember?" rows={4} />
          <div className="cxv-form__actions">
            <Button variant="primary" icon={<IconPlus size={16} />} onClick={handleCreateMemory} loading={busyAction === 'create'}>Save memory</Button>
          </div>
        </div>

        <div className="cxv-card cxv-form">
          <div className="cxv-card__titles">
            <strong>From this chat</strong>
            <span className="cxv-card__sub-plain">Pull a useful reply straight into memory</span>
          </div>
          <div className="cxv-mem__context">
            <strong title={selectedConversation?.title || 'No chat selected'}>{selectedConversation ? latestChatLabel : 'No chat selected'}</strong>
            <span>{canSaveLatestReply ? 'Latest assistant reply is ready to save.' : 'Open a conversation with an assistant reply to save it here.'}</span>
          </div>
          <div className="cxv-form__actions">
            <Button variant="ghost" icon={<IconChat size={16} />} onClick={() => setTab('chat')}>Open chat</Button>
            <Button variant="primary" onClick={saveToMemory} disabled={!canSaveLatestReply}>Save latest reply</Button>
          </div>
        </div>
      </div>

      <div className="cxv-toolbar">
        <label className="cxv-search">
          <IconSearch size={16} />
          <input value={memorySearch} onChange={(e) => setMemorySearch(e.target.value)} placeholder="Search memories" />
        </label>
        <select value={scopeFilter} onChange={(e) => setScopeFilter(e.target.value)}>
          {availableScopes.map((scope) => (
            <option key={scope} value={scope}>{scope === 'all' ? 'All scopes' : scope}</option>
          ))}
        </select>
        <Button variant="ghost" icon={<IconPin size={15} />} className={showPinnedOnly ? 'is-active' : ''} onClick={() => setShowPinnedOnly((c) => !c)}>
          Pinned only
        </Button>
      </div>

      {visibleMemories.length === 0 ? (
        <EmptyState
          icon={<IconMemory size={26} />}
          title={memories.length === 0 ? 'No memories yet' : 'Nothing matched'}
          description={memories.length === 0
            ? 'Add one above, or save a useful assistant reply from chat — the Memory page becomes your working set, not a dead archive.'
            : 'No memories matched those filters. Try a broader search or clear the pinned / scope filter.'}
        />
      ) : (
        <div className="cxv-grid">
          {visibleMemories.map((memory) => {
            const isEditing = editingId === memory.id
            const isDeleting = busyAction === `delete:${memory.id}`
            const isPinning = busyAction === `pin:${memory.id}`
            const isSavingEdit = busyAction === `edit:${memory.id}`
            const confirming = pendingDeleteId === memory.id

            return (
              <article key={memory.id} className={`cxv-card cxv-mem ${memory.pinned ? 'is-pinned' : ''}`}>
                <div className="cxv-card__head">
                  <div className="cxv-card__titles">
                    <strong>{memory.title}</strong>
                    <div className="cxv-mem__meta">
                      <span className="cxv-tag">{memory.scope}</span>
                      {memory.pinned && <span className="cxv-tag cxv-tag--accent"><IconPin size={11} /> Pinned</span>}
                    </div>
                  </div>
                  <div className="cxv-card__actions">
                    <Button variant="ghost" size="sm" icon={<IconPin size={15} />} aria-label={memory.pinned ? 'Unpin' : 'Pin'} className={memory.pinned ? 'is-active' : ''} loading={isPinning} onClick={() => handleTogglePin(memory)} disabled={Boolean(busyAction)} />
                    <Button variant="ghost" size="sm" icon={<IconCopy size={15} />} aria-label="Copy" onClick={() => handleCopy(memory)} />
                  </div>
                </div>

                {isEditing ? (
                  <div className="cxv-edit">
                    <input value={editDraft.title} onChange={(e) => setEditDraft((c) => ({ ...c, title: e.target.value }))} placeholder="Title" />
                    <input value={editDraft.scope} onChange={(e) => setEditDraft((c) => ({ ...c, scope: e.target.value }))} placeholder="Scope" />
                    <textarea value={editDraft.body} onChange={(e) => setEditDraft((c) => ({ ...c, body: e.target.value }))} rows={5} placeholder="Memory body" />
                    <div className="cxv-form__actions">
                      <Button variant="primary" onClick={() => handleSaveEdit(memory.id)} loading={isSavingEdit}>Save changes</Button>
                      <Button variant="ghost" onClick={cancelEditing} disabled={isSavingEdit}>Cancel</Button>
                    </div>
                  </div>
                ) : (
                  <>
                    <p className="cxv-mem__body">{memory.body}</p>
                    <footer className="cxv-card__foot">
                      <span className="cxv-card__meta">Updated {formatDate(memory.updated_at) || 'Unknown'}</span>
                      <div className="cxv-card__actions">
                        <Button variant="ghost" size="sm" icon={<IconEdit size={15} />} onClick={() => startEditing(memory)} disabled={Boolean(busyAction)}>Edit</Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className={confirming ? 'cxv-danger is-armed' : 'cxv-danger'}
                          icon={<IconTrash size={15} />}
                          loading={isDeleting}
                          onClick={() => handleDelete(memory.id)}
                          disabled={isDeleting || (Boolean(busyAction) && !confirming)}
                        >
                          {confirming ? 'Confirm' : ''}
                        </Button>
                      </div>
                    </footer>
                  </>
                )}
              </article>
            )
          })}
        </div>
      )}
    </section>
  )
}
