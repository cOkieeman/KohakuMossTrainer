import { useEffect, useRef, useState } from 'react'
import { Copy, Download, FolderOpen, Square } from 'lucide-react'
import { api, bytes, dateTime, elapsed } from './api'
import { activeStatuses, kindLabels, phaseLabels, statusLabels, type Environment, type Job } from './types'
import { useTask } from './useTask'

interface Props {
  job?: Job
  environment: Environment
  onStop: (job: Job) => void
  onMessage: (message: string, error?: boolean) => void
  onResume?: (path: string, fresh: boolean) => void
}

export default function Monitor({ job, environment, onStop, onMessage, onResume }: Props) {
  const { detail, log, error } = useTask(job?.id)
  const [follow, setFollow] = useState(true)
  const logArea = useRef<HTMLPreElement>(null)
  const current = detail?.job || job
  const running = current && activeStatuses.has(current.status)
  const metrics = detail?.metrics || []
  const last = metrics.at(-1)
  const progress = last ? Math.min(100, 100 * last.step / (current?.config.steps || 1)) : 0
  const weightFiles = detail?.artifacts.weights || []
  const samples = detail?.artifacts.samples || []
  const files = detail?.artifacts.files || []
  const checkpoints = detail?.artifacts.checkpoints || []
  const failure = current?.error || error
  const memoryFailure = /1455|页面文件|paging file|out of memory|cuda.*memory/i.test(`${failure || ''}\n${log}`)
  useEffect(() => {
    if (follow && logArea.current) logArea.current.scrollTop = logArea.current.scrollHeight
  }, [log, follow])

  async function copyLog() {
    try { await navigator.clipboard.writeText(log); onMessage('日志已复制') }
    catch { onMessage('复制失败，可使用下载日志', true) }
  }

  async function openFolder() {
    if (!current) return
    try { await api(`/jobs/${current.id}/open-folder`, { method: 'POST' }) }
    catch (reason) { onMessage(reason instanceof Error ? reason.message : '目录打开失败', true) }
  }

  const artifactUrl = (path: string) => `/api/jobs/${current?.id}/artifacts/${path.split('/').map(encodeURIComponent).join('/')}`
  return <aside className="monitor" aria-label="运行监视">
    <header className="monitor-heading"><h2>运行监视</h2><span className={`ready-indicator ${environment.ready ? '' : 'unready'}`}>
      <i />{environment.ready ? '环境就绪' : '待检查'}</span></header>
    <div className="gpu-summary"><span>{environment.gpu?.name.replace('NVIDIA GeForce ', '') || 'GPU 未识别'}</span>
      <span>{environment.gpu ? `${(environment.gpu.used_mib / 1024).toFixed(1)} / ${(environment.gpu.total_mib / 1024).toFixed(1)} GB` : '—'}</span></div>
    <div className={`task-status status-${current?.status || 'idle'}`}>
      <span>状态</span><strong>{current ? statusLabels[current.status] || current.status : '等待任务'}</strong>
      {running && <button className="stop-button" onClick={() => current && onStop(current)} disabled={current.status === 'stopping'}><Square size={12} />停止任务</button>}
    </div>
    {current && <div className="task-caption"><span>{current.origin === 'external' ? '外部启动 · ' : ''}{kindLabels[current.kind] || current.kind}{current.phase ? ` / ${phaseLabels[current.phase] || current.phase}` : ''}</span><span>{dateTime(current.started_at)}</span></div>}
    {current?.run_label && <p className="path-text">{current.run_label}</p>}
    <div className="metric-strip">
      <div><span>步数</span><strong>{last?.step ?? '—'}</strong></div>
      <div><span>Loss</span><strong>{last?.loss !== undefined ? last.loss.toFixed(4) : '—'}</strong></div>
      <div><span>耗时</span><strong>{elapsed(current?.started_at, current?.finished_at)}</strong></div>
    </div>
    {last && <div className="training-progress"><progress max="100" value={progress} /><span>{last.step} / {current?.config.steps}</span></div>}
    {metrics.filter(row => row.loss !== undefined).length > 1 && <LossChart rows={metrics} />}
    {failure && <div className="task-error" role="alert">
      <p>{current?.error ? memoryFailure ? '可用内存不足。请释放空闲模型缓存，核对内存与显存占用后重试。' : current.can_retry === 0 ? '任务执行失败。请查看下方日志和原始错误详情。' : '任务执行失败。请查看下方日志，修正配置后在运行记录中重试。' : '任务信息读取失败。请检查本机服务连接。'}</p>
      {current && <a href={`/api/jobs/${current.id}/console.log`}>下载完整日志</a>}
      <details><summary>原始错误详情</summary><pre>{failure}</pre></details>
    </div>}
    <section className="log-section">
      <div className="section-heading"><h2>运行日志</h2><div className="log-actions">
        <label><input type="checkbox" checked={follow} onChange={event => setFollow(event.target.checked)} />跟随</label>
        <button onClick={copyLog} disabled={!log} title="复制日志"><Copy size={14} /><span>复制</span></button>
        {current && <a href={`/api/jobs/${current.id}/console.log`} title="下载日志" aria-label="下载日志"><Download size={15} /></a>}
      </div></div>
      <pre className="log-output" ref={logArea} tabIndex={0} aria-label="实时运行日志">{log || '运行任务后，这里显示实时日志。'}</pre>
    </section>
    <section className="checkpoint-section" id="checkpoints" tabIndex={-1}>
      <div className="section-heading"><h2>检查点</h2><button className="text-button" disabled={!current} onClick={openFolder}><FolderOpen size={14} />打开目录</button></div>
      {checkpoints.length > 0 && onResume && <div className="resume-list">{checkpoints.map(checkpoint => <div key={checkpoint.path}><span>{checkpoint.name}</span><button type="button" className="text-button" onClick={() => onResume(checkpoint.path, false)}>设置续训</button><button type="button" className="text-button" onClick={() => onResume(checkpoint.path, true)}>加载权重另训</button></div>)}</div>}
      {weightFiles.length ? <div className="artifact-list">{weightFiles.map(file =>
        <a className="artifact-row" href={artifactUrl(file.path)} key={file.path} title={file.path}>
          <div><strong>{file.name}</strong><span>{file.path}</span></div><span>{bytes(file.size || 0)}</span><Download size={15} />
        </a>)}</div> : <div className="checkpoint-empty">训练保存的权重将在这里显示。</div>}
    </section>
    {samples.length > 0 && <section className="sample-section"><h2>训练采样</h2><div className="sample-grid">{samples.map(sample =>
      <a href={artifactUrl(sample.path)} key={sample.path} title={sample.path}><img src={artifactUrl(sample.path)} alt={sample.name} loading="lazy" /><span>{sample.name}</span></a>
    )}</div></section>}
    {files.length > 0 && <section className="sample-section"><h2>报告与其他文件</h2><div className="artifact-list">{files.map(file => <a className="artifact-row" key={file.path} href={artifactUrl(file.path)}><div><strong>{file.name}</strong><span>{file.path}</span></div><Download size={15} /></a>)}</div></section>}
  </aside>
}

function LossChart({ rows }: { rows: Record<string, number>[] }) {
  const points = rows.filter(row => row.loss !== undefined)
  const minimum = Math.min(...points.map(row => row.loss))
  const maximum = Math.max(...points.map(row => row.loss))
  const firstStep = points[0].step
  const lastStep = points.at(-1)!.step
  const coordinates = points.map(row => `${10 + (row.step - firstStep) / Math.max(1, lastStep - firstStep) * 340},${80 - (row.loss - minimum) / Math.max(0.00001, maximum - minimum) * 65}`).join(' ')
  return <div className="loss-chart"><div><span>Loss 曲线</span><span>{firstStep}–{lastStep} step</span></div>
    <svg viewBox="0 0 360 90" role="img" aria-label={`训练 Loss，当前 ${points.at(-1)!.loss.toFixed(4)}`}><line x1="10" y1="81" x2="350" y2="81" className="chart-baseline" /><polyline points={coordinates} /></svg></div>
}
