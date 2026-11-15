import { useEffect, useRef } from 'react'
import { Check, Download, FolderOpen, RefreshCw, X } from 'lucide-react'
import { bytes, dateTime } from './api'
import { activeStatuses, kindLabels, statusLabels, type Environment, type Job } from './types'

export function History({ jobs, onSelect, onRetry, onCancel }: { jobs: Job[]; onSelect: (job: Job) => void; onRetry: (job: Job) => void; onCancel: (job: Job) => void }) {
  return <section className="standalone-page"><header className="page-heading"><div><h1>运行记录</h1><p>自动同步工作台与外部窗口启动的任务</p></div><span>{jobs.length} 个任务</span></header>
    {jobs.length ? <div className="history-table-wrap"><table className="history-table"><thead><tr><th>项目与任务</th><th>类型</th><th>状态</th><th>开始时间</th><th /></tr></thead><tbody>{jobs.map(job =>
      <tr key={job.id}><td><strong>{job.project_name}</strong>{job.run_label && <span className="path-text">{job.run_label}</span>}<span className="path-text">{job.id}</span></td><td>{kindLabels[job.kind] || job.kind}<span className="path-text">{job.origin === 'external' ? '外部启动' : '工作台启动'}</span></td><td><span className={`history-status status-${job.status}`}>{statusLabels[job.status] || job.status}</span></td><td>{dateTime(job.started_at)}</td><td><div className="history-actions"><button className="button secondary" onClick={() => onSelect(job)}>查看</button>
        {activeStatuses.has(job.status) ? <button className="text-button" disabled={job.status === 'stopping'} onClick={() => onCancel(job)}>停止</button> : job.can_retry !== 0 && <button className="text-button" onClick={() => onRetry(job)}>重新提交</button>}</div></td></tr>)}</tbody></table></div>
      : <div className="large-empty"><h2>还没有运行任务</h2><p>在工作台或本机命令行启动训练后，记录会自动出现在这里。</p></div>}
  </section>
}

export function EnvironmentPage({ environment, busy, onCheck }: { environment: Environment; busy: boolean; onCheck: () => void }) {
  return <section className="standalone-page environment-page"><header className="page-heading"><div><h1>模型与环境</h1><p>{environment.root}</p></div><button className="button primary" disabled={busy} onClick={onCheck}><RefreshCw size={16} />重新检测环境</button></header>
    <div className="environment-columns"><section><h2>运行环境</h2><dl className="environment-list"><div><dt>GPU</dt><dd>{environment.gpu?.name || '未识别'}</dd></div>
      <div><dt>显存</dt><dd>{environment.gpu ? `${(environment.gpu.used_mib / 1024).toFixed(1)} / ${(environment.gpu.total_mib / 1024).toFixed(1)} GB 已用` : '—'}</dd></div>
      <div><dt>Python</dt><dd>{environment.python}</dd></div>{Object.entries(environment.packages).map(([name, version]) => <div key={name}><dt>{name}</dt><dd>{version || '未安装'}</dd></div>)}
      <div><dt>上次检测</dt><dd>{dateTime(environment.checked_at)}</dd></div></dl>
      {environment.doctor.missing?.map(item => <div className="inline-error" key={item.what}><strong>{item.what}</strong><code>{item.fix}</code></div>)}
    </section><section><h2>本机权重</h2><div className="model-assets">{environment.assets.map(asset => <article key={asset.path}>
      <div className="asset-heading"><span>{asset.key === 'transformer_path' ? 'MOSS-Image-heal' : asset.key === 'vae_path' ? 'FLUX.2 VAE' : 'Qwen3.5-9B'}</span>{asset.exists ? <Check size={17} /> : <X size={17} />}</div>
      <strong>{asset.name}</strong><p className="path-text">{asset.path}</p><small>{asset.exists ? bytes(asset.size) : '文件缺失'}</small>
    </article>)}</div></section></div>
  </section>
}

export function ConfigPreview({ text, projectId, onClose }: { text: string; projectId: string; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null)
  useEffect(() => { dialog.current?.showModal() }, [])
  return <dialog ref={dialog} className="config-preview" onCancel={onClose} aria-labelledby="config-title"><header className="dialog-heading"><h2 id="config-title">原生训练配置</h2><button className="icon-button" onClick={onClose} aria-label="关闭配置预览"><X size={19} /></button></header><pre tabIndex={0}>{text}</pre><footer className="dialog-footer"><span>由当前保存的配置生成</span><a className="button primary" href={`/api/projects/${projectId}/config`}><Download size={16} />下载 .py 配置</a></footer></dialog>
}

export function RootPath({ value }: { value: string }) {
  return <span className="root-path" title={value}><FolderOpen size={12} />{value}</span>
}
