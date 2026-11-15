import { useEffect, useRef, useState } from 'react'
import { ArrowLeft, Check, File, Folder, HardDrive, X } from 'lucide-react'
import { api } from './api'

interface Directory {
  path: string
  parent: string
  entries: { name: string; path: string; directory: boolean }[]
}

interface Props {
  initial: string
  mode: 'directory' | 'file' | 'all' | 'config'
  title?: string
  onSelect: (path: string) => void
  onClose: () => void
}

export default function FileBrowser({ initial, mode, title, onSelect, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null)
  const [directory, setDirectory] = useState<Directory | null>(null)
  const [path, setPath] = useState(initial)
  const [filter, setFilter] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  async function load(value: string) {
    setLoading(true)
    setError('')
    try {
      const result = await api<Directory>(`/browse?mode=${mode}&path=${encodeURIComponent(value)}`)
      setDirectory(result)
      setPath(result.path)
      setFilter('')
    } catch (reason) {
      setDirectory(null)
      const message = reason instanceof Error ? reason.message : '目录读取失败'
      setError(message === 'Directory does not exist' ? '目录不存在，请修改路径或重新选择。'
        : message === 'This directory cannot be read' ? '没有权限读取此目录。' : message)
    } finally { setLoading(false) }
  }

  useEffect(() => {
    dialog.current?.showModal()
    void load(initial)
  }, [])

  const entries = directory?.entries.filter(entry => entry.name.toLocaleLowerCase().includes(filter.toLocaleLowerCase())) || []
  return <dialog ref={dialog} className="file-dialog" onCancel={onClose} aria-labelledby="file-browser-title">
    <div className="dialog-heading">
      <h2 id="file-browser-title">{title || (mode === 'directory' ? '选择数据集目录' : '选择模型文件')}</h2>
      <button className="icon-button" onClick={onClose} aria-label="关闭目录选择"><X size={19} /></button>
    </div>
    <form className="browser-path" onSubmit={event => { event.preventDefault(); void load(path) }}>
      <button type="button" className="icon-button" disabled={loading} onClick={() => void load(directory?.parent || '')} aria-label="上一级目录"><ArrowLeft size={17} /></button>
      <button type="button" className="icon-button" disabled={loading} onClick={() => void load('')} aria-label="显示磁盘"><HardDrive size={17} /></button>
      <input aria-label="当前目录" disabled={loading} value={path} onChange={event => setPath(event.target.value)} spellCheck={false} />
      <button type="submit" className="button secondary" disabled={loading}>前往</button>
    </form>
    <input className="browser-filter" aria-label="筛选文件名" disabled={loading} placeholder="筛选名称" value={filter} onChange={event => setFilter(event.target.value)} />
    {error && <p className="inline-error" role="alert">{error}</p>}
    <div className="browser-list" aria-busy={loading}>
      {loading ? <p className="empty-message">正在读取目录…</p> : error ? <p className="empty-message">请修改路径，或点击磁盘按钮重新选择。</p> : entries.length ? entries.map(entry =>
        <button key={entry.path} className="browser-entry" onClick={() => entry.directory ? void load(entry.path) : onSelect(entry.path)}>
          {entry.directory ? <Folder size={18} /> : <File size={18} />}
          <span>{entry.name}</span>
          {!entry.directory && <span className="entry-action">选择</span>}
        </button>) : <p className="empty-message">{directory?.path ? '此目录下没有匹配的项目' : '没有可读取的磁盘'}</p>}
    </div>
    <footer className="dialog-footer">
      <span className="path-text" title={directory?.path}>{directory?.path || (error ? '目录不可用' : '本机磁盘')}</span>
      <button className="button secondary" onClick={onClose}>取消</button>
      {(mode === 'directory' || mode === 'all') && <button className="button primary" disabled={!directory?.path || loading} onClick={() => directory && onSelect(directory.path)}><Check size={16} />选择当前目录</button>}
    </footer>
  </dialog>
}
