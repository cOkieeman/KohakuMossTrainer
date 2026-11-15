import { useEffect, useRef, useState } from 'react'
import { Check, Copy, Database, Download, LoaderCircle, Moon, Play, Plus, RotateCcw, Save, Sun, Trash2, Upload, X } from 'lucide-react'
import { api } from './api'
import Configuration from './Configuration'
import FileBrowser from './FileBrowser'
import Monitor from './Monitor'
import Tools from './Tools'
import { ConfigPreview, EnvironmentPage, History, RootPath } from './Pages'
import { activeStatuses, type Environment, type Job, type JobKind, type Project, type TrainSettings } from './types'

interface Draft { name: string; config: TrainSettings }
interface Bootstrap { projects: Project[]; jobs: Job[]; environment: Environment; bindings: Record<string, string[]> }
type Page = 'workspace' | 'history' | 'environment' | 'tools'
type PathKey = 'dataset_path' | 'reg_dataset_path' | 'transformer_path' | 'text_encoder_path' | 'vae_path'

function restoreDraft(project: Project): Draft {
  try {
    const saved = JSON.parse(localStorage.getItem(`kohaku-draft-${project.id}`) || 'null')
    if (saved?.version === 1 && saved.base === project.updated_at) return {
      name: saved.draft.name,
      config: { ...project.config, ...saved.draft.config },
    }
  } catch { /* An unreadable browser draft falls back to the saved project. */ }
  return { name: project.name, config: project.config }
}

export default function App() {
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    try { return localStorage.getItem('kohaku-theme') === 'light' ? 'light' : 'dark' }
    catch { return 'dark' }
  })
  const [projects, setProjects] = useState<Project[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [environment, setEnvironment] = useState<Environment | null>(null)
  const [selectedId, setSelectedId] = useState('')
  const [selectedJobId, setSelectedJobId] = useState<string>()
  const [draft, setDraft] = useState<Draft | null>(null)
  const [page, setPage] = useState<Page>('workspace')
  const [stage, setStage] = useState(0)
  const [busy, setBusy] = useState('')
  const [connection, setConnection] = useState(true)
  const [loadError, setLoadError] = useState('')
  const [message, setMessage] = useState<{ text: string; error: boolean } | null>(null)
  const [browserKey, setBrowserKey] = useState<PathKey | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [bindings, setBindings] = useState<Record<string, string[]>>({})
  const [importOpen, setImportOpen] = useState(false)
  const [showArchived, setShowArchived] = useState(false)
  const messageTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try { localStorage.setItem('kohaku-theme', theme) } catch { /* Theme remains available for this session. */ }
  }, [theme])

  const project = projects.find(item => item.id === selectedId)
  const dirty = !!draft && !!project && JSON.stringify(draft) !== JSON.stringify({ name: project.name, config: project.config })
  const projectJobs = jobs.filter(job => job.project_id === selectedId)
  const currentJob = projectJobs.find(job => job.id === selectedJobId) || projectJobs.find(job => activeStatuses.has(job.status)) || projectJobs[0]
  const gpuTask = jobs.find(job => job.kind !== 'check' && ['starting', 'running', 'stopping'].includes(job.status))
  const nativeDatasets = draft?.config.native_overrides?.DATASETS
  const dataReady = Array.isArray(nativeDatasets) ? nativeDatasets.length > 0 : draft ? Boolean(draft.config.dataset_path.trim() && (!draft.config.reg_enabled || draft.config.reg_dataset_path.trim())) : false

  function notify(text: string, error = false) {
    clearTimeout(messageTimer.current)
    setMessage({ text, error })
    if (!error) messageTimer.current = setTimeout(() => setMessage(null), 4500)
  }

  useEffect(() => {
    const controller = new AbortController()
    api<Bootstrap>('/bootstrap', { signal: controller.signal }).then(value => {
      setProjects(value.projects)
      setJobs(value.jobs)
      setEnvironment(value.environment)
      setBindings(value.bindings || {})
      const selected = value.projects.find(item => item.id === localStorage.getItem('kohaku-project')) || value.projects.find(item => !item.archived) || value.projects[0]
      if (selected) { setSelectedId(selected.id); setDraft(restoreDraft(selected)) }
    }).catch(reason => { if (!controller.signal.aborted) setLoadError(String(reason.message || reason)) })
    return () => controller.abort()
  }, [])

  useEffect(() => {
    if (!environment) return
    let alive = true
    let timer: ReturnType<typeof setTimeout>
    const controller = new AbortController()
    async function refresh() {
      try {
        const [nextProjects, nextJobs, nextEnvironment] = await Promise.all([
          api<Project[]>('/projects', { signal: controller.signal }),
          api<Job[]>('/jobs', { signal: controller.signal }),
          api<Environment>('/environment', { signal: controller.signal }),
        ])
        if (!alive) return
        setProjects(nextProjects)
        setJobs(nextJobs)
        setEnvironment(nextEnvironment)
        setConnection(true)
      } catch { if (alive) setConnection(false) }
      if (alive) timer = setTimeout(refresh, document.hidden ? 8000 : 2600)
    }
    timer = setTimeout(refresh, 1500)
    return () => { alive = false; clearTimeout(timer); controller.abort() }
  }, [!!environment])

  useEffect(() => {
    if (!draft || !project) return
    try {
      localStorage.setItem('kohaku-project', project.id)
      localStorage.setItem(`kohaku-draft-${project.id}`, JSON.stringify({ version: 1, base: project.updated_at, draft }))
    } catch { /* Explicit saves remain available when browser storage is full. */ }
  }, [draft, project?.id, project?.updated_at])

  useEffect(() => {
    const warn = (event: BeforeUnloadEvent) => { if (dirty) event.preventDefault() }
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [dirty])

  function selectProject(value: Project) {
    setSelectedId(value.id)
    setDraft(restoreDraft(value))
    setSelectedJobId(undefined)
    setStage(0)
    setPage('workspace')
  }

  function change<K extends keyof TrainSettings>(key: K, value: TrainSettings[K]) {
    setDraft(previous => {
      if (!previous) return previous
      const config = { ...previous.config, [key]: value }
      if (key !== 'native_overrides') {
        config.native_overrides = { ...previous.config.native_overrides }
        for (const name of bindings[key] || []) delete config.native_overrides[name]
      }
      return { ...previous, config }
    })
  }

  function validForm() {
    return ['training-form', 'native-tool-form'].every(id => {
      const form = document.getElementById(id) as HTMLFormElement | null
      return !form || form.reportValidity()
    })
  }

  async function save(quiet = false): Promise<Project | null> {
    if (!draft || !project || !validForm()) return null
    const value = draft
    try {
      const saved = await api<Project>(`/projects/${project.id}`, { method: 'PUT', body: JSON.stringify(value) })
      setProjects(previous => previous.map(item => item.id === saved.id ? saved : item))
      if (!quiet) notify('配置已保存')
      return saved
    } catch (reason) { notify(reason instanceof Error ? reason.message : '保存失败', true); return null }
  }

  async function saveFromButton() {
    setBusy('save')
    await save()
    setBusy('')
  }

  async function newProject() {
    setBusy('new')
    try {
      const created = await api<Project>('/projects', { method: 'POST', body: JSON.stringify({ name: `新训练项目 ${projects.length + 1}` }) })
      setProjects(previous => [created, ...previous])
      selectProject(created)
    } catch (reason) { notify(reason instanceof Error ? reason.message : '创建失败', true) }
    finally { setBusy('') }
  }

  async function start(kind: JobKind) {
    if (busy || !project) return
    setBusy(kind)
    const saved = await save(true)
    if (!saved) { setBusy(''); return }
    try {
      const job = await api<Job>(`/projects/${saved.id}/jobs`, { method: 'POST', body: JSON.stringify({ kind }) })
      setJobs(previous => [job, ...previous])
      setSelectedJobId(job.id)
      if (['train', 'cache', 'check', 'doctor'].includes(kind)) {
        setPage('workspace')
        setStage(kind === 'train' ? 2 : kind === 'cache' ? 1 : 0)
      } else setPage('tools')
      notify('任务已提交，运行顺序可在运行记录中查看')
    } catch (reason) { notify(reason instanceof Error ? reason.message : '任务提交失败', true) }
    finally { setBusy('') }
  }

  async function stop(job: Job) {
    try {
      const next = await api<Job>(`/jobs/${job.id}/cancel`, { method: 'POST' })
      setJobs(previous => previous.map(item => item.id === next.id ? next : item))
      notify('正在停止任务')
    } catch (reason) { notify(reason instanceof Error ? reason.message : '停止失败', true) }
  }

  async function showConfig() {
    const saved = await save(true)
    if (!saved) return
    try {
      const response = await fetch(`/api/projects/${saved.id}/config`)
      if (!response.ok) throw new Error('配置生成失败')
      setPreview(await response.text())
    } catch (reason) { notify(reason instanceof Error ? reason.message : '配置生成失败', true) }
  }

  function goToStage(value: number) {
    setStage(value)
    const target = ['data-settings', 'cache-settings', 'train-settings', 'checkpoints'][value]
    requestAnimationFrame(() => document.getElementById(target)?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }))
  }

  async function copyProject() {
    const saved = await save(true)
    if (!saved) return
    try {
      const copied = await api<Project>(`/projects/${saved.id}/copy`, { method: 'POST' })
      setProjects(previous => [copied, ...previous]); selectProject(copied)
    } catch (reason) { notify(String(reason), true) }
  }

  async function archiveProject(restore = false) {
    if (!project) return
    try {
      const updated = await api<Project>(`/projects/${project.id}${restore ? '/restore' : ''}`, { method: restore ? 'POST' : 'DELETE' })
      setProjects(previous => previous.map(item => item.id === updated.id ? updated : item))
      const next = projects.find(item => item.id !== project.id && !item.archived)
      if (!restore && next) selectProject(next)
      else selectProject(updated)
      notify(restore ? '项目已恢复' : '项目已移入已删除项目')
    } catch (reason) { notify(String(reason), true) }
  }

  async function importProject(path: string) {
    setImportOpen(false)
    try {
      const imported = await api<Project>('/projects/import', { method: 'POST', body: JSON.stringify({ path }) })
      setProjects(previous => [imported, ...previous]); selectProject(imported)
    } catch (reason) { notify(String(reason), true) }
  }

  async function retry(job: Job) {
    try {
      const next = await api<Job>(`/jobs/${job.id}/retry`, { method: 'POST' })
      setJobs(previous => [next, ...previous]); setSelectedJobId(next.id)
      notify('已按原任务快照重新提交')
    } catch (reason) { notify(String(reason), true) }
  }

  async function resumeFrom(path: string, fresh: boolean) {
    const saved = await save(true)
    if (!saved) return
    try {
      const result = await api<{ project: Project }>(`/projects/${saved.id}/resume`, { method: 'POST', body: JSON.stringify({ checkpoint: path, fresh }) })
      setProjects(previous => previous.map(item => item.id === result.project.id ? result.project : item))
      setDraft({ name: result.project.name, config: result.project.config }); setStage(4)
      notify(fresh ? '已设置初始训练权重，可调整参数后开始训练' : '已设置恢复检查点，可确认参数后开始训练')
    } catch (reason) { notify(String(reason), true) }
  }

  return <div className="app-shell">
    <header className="app-nav"><a className="app-brand" href="#" onClick={event => { event.preventDefault(); setPage('workspace') }}>KohakuMoss Studio</a>
      <nav aria-label="主导航">{([['workspace', '训练工作台'], ['tools', '原生工具'], ['history', '运行记录'], ['environment', '模型与环境']] as const).map(([value, label]) =>
        <button key={value} aria-current={page === value ? 'page' : undefined} onClick={() => setPage(value)}>{label}</button>)}</nav>
      <button className="theme-toggle" type="button" aria-label={theme === 'dark' ? '切换为浅色主题' : '切换为暗色主题'}
        onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>{theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}<span>{theme === 'dark' ? '浅色' : '暗色'}</span></button>
    </header>
    {!connection && <div className="connection-error" role="alert">本机服务连接中断，正在重试…</div>}
    {loadError ? <div className="boot-error" role="alert"><h1>工作台加载失败</h1><p>{loadError}</p><button className="button primary" onClick={() => location.reload()}>重新连接</button></div>
      : !environment || !draft || !project ? <div className="loading-shell" aria-label="正在加载工作台"><div /><div /><div /></div>
      : <div className="app-body">
        <aside className="project-rail"><header><h2>项目</h2><button className="icon-button" onClick={newProject} disabled={!!busy} aria-label="新建训练项目"><Plus size={18} /></button></header>
          <div className="project-list">{projects.filter(item => showArchived || !item.archived).map(item => <button key={item.id} className={item.id === selectedId ? 'selected' : ''} onClick={() => selectProject(item)}>
            <span>{item.name}</span>{jobs.some(job => job.project_id === item.id && activeStatuses.has(job.status)) && <i className="running-dot" />}</button>)}</div>
          <div className="rail-footer"><div className="project-actions"><button onClick={() => setImportOpen(true)}><Upload size={14} />导入配置</button><button onClick={copyProject}><Copy size={14} />复制项目</button>
            <a href={`/api/projects/${project.id}/studio.json`}><Download size={14} />导出已保存项目</a>
            {project.archived ? <button onClick={() => archiveProject(true)}><RotateCcw size={14} />恢复项目</button> : <button onClick={() => archiveProject()}><Trash2 size={14} />删除项目</button>}
            <label><input type="checkbox" checked={showArchived} onChange={event => setShowArchived(event.target.checked)} />显示已删除项目</label></div><span>KohakuMoss 0.1.0</span><span>本机训练工作台</span></div>
        </aside>
        {page === 'workspace' ? <main className="workspace-main">
          <header className="workspace-heading"><div className="project-name"><label htmlFor="project-name" className="visually-hidden">项目名称</label><input id="project-name" aria-label="项目名称" value={draft.name} maxLength={120} onChange={event => setDraft({ ...draft, name: event.target.value })} /><span className="save-state">{dirty ? '未保存' : <><Check size={12} />已保存</>}</span></div>
            {Object.keys(draft.config.native_overrides || {}).length > 0 && <button className="text-button" onClick={() => goToStage(4)}>原生覆盖 {Object.keys(draft.config.native_overrides).length} 项</button>}
            <button className="button primary save-button" onClick={saveFromButton} disabled={!!busy}>{busy === 'save' ? <LoaderCircle className="spin" size={15} /> : <Save size={15} />}保存配置</button></header>
          <div className="workspace-columns"><div className="configuration-pane"><nav className="workflow-nav" aria-label="训练流程">{['数据检查', '建立缓存', '训练', '检查点', '完整参数'].map((label, index) =>
            <button key={label} aria-current={stage === index ? 'step' : undefined} onClick={() => goToStage(index)}>{label}</button>)}</nav>
            <Configuration project={project} value={draft.config} onChange={change} onBrowse={setBrowserKey} onSave={saveFromButton} onConfig={showConfig} stage={stage} />
            <footer className="training-actions"><div className="action-buttons"><button className="button secondary" disabled={!!busy || !dataReady} onClick={() => start('check')}>{busy === 'check' ? <LoaderCircle className="spin" size={16} /> : <Check size={16} />}检查数据</button>
              <button className="button secondary" disabled={!!busy || !dataReady} onClick={() => start('cache')}>{busy === 'cache' ? <LoaderCircle className="spin" size={16} /> : <Database size={16} />}建立缓存</button>
              <button className="button primary start-button" disabled={!!busy || !dataReady} onClick={() => start('train')}>{busy === 'train' ? <LoaderCircle className="spin" size={16} /> : <Play size={16} />}开始训练</button></div>
              {!dataReady && !draft.config.dataset_path.trim() && <p>选择数据集目录后即可检查数据和开始训练。</p>}
              {!dataReady && draft.config.dataset_path.trim() && draft.config.reg_enabled && !draft.config.reg_dataset_path.trim() && <p>选择正则数据集目录后即可检查数据和开始训练。</p>}
              {gpuTask && <p>GPU 任务运行中 · {gpuTask.project_name} <button className="text-button" onClick={() => {
                const target = projects.find(item => item.id === gpuTask.project_id)
                if (target) { selectProject(target); setSelectedJobId(gpuTask.id) }
              }}>查看任务</button></p>}
            </footer>
          </div><Monitor job={currentJob} environment={environment} onStop={stop} onMessage={notify} onResume={resumeFrom} /></div>
        </main> : page === 'tools' ? <Tools value={draft.config} environment={environment} job={currentJob} projectId={project.id} busy={!!busy} onChange={config => setDraft({ ...draft, config })} onRun={start} onSave={() => { void save() }} onStop={stop} onMessage={notify} />
          : page === 'history' ? <main className="page-main"><History jobs={jobs} onRetry={retry} onCancel={stop} onSelect={job => { const target = projects.find(item => item.id === job.project_id); if (target) { selectProject(target); setSelectedJobId(job.id); if (job.origin !== 'external' && !['train', 'cache', 'check', 'doctor'].includes(job.kind)) setPage('tools') } }} /></main>
          : <main className="page-main"><EnvironmentPage environment={environment} busy={!!busy} onCheck={() => start('doctor')} /></main>}
      </div>}
    <footer className="app-statusbar"><RootPath value={environment?.root || '正在连接本机服务'} /><span>{connection ? '本机服务已连接' : '连接中断'}</span></footer>
    {browserKey && draft && <FileBrowser mode={browserKey === 'dataset_path' || browserKey === 'reg_dataset_path' ? 'directory' : 'file'} title={browserKey === 'reg_dataset_path' ? '选择正则数据集目录' : undefined} initial={draft.config[browserKey] || environment?.root || ''} onSelect={path => { change(browserKey, path); setBrowserKey(null) }} onClose={() => setBrowserKey(null)} />}
    {importOpen && <FileBrowser mode="config" title="导入项目 JSON 或原生 Python 配置" initial={environment?.root || ''} onSelect={importProject} onClose={() => setImportOpen(false)} />}
    {preview !== null && project && <ConfigPreview text={preview} projectId={project.id} onClose={() => setPreview(null)} />}
    {message && <div className={`toast${message.error ? ' error' : ''}`} role={message.error ? 'alert' : 'status'}><span>{message.text}</span><button className="icon-button" onClick={() => setMessage(null)} aria-label="关闭提示"><X size={15} /></button></div>}
  </div>
}
