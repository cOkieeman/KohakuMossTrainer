import { useEffect, useRef, useState } from 'react'
import { ArrowLeft, ArrowRight, Download, Expand, X } from 'lucide-react'
import { api, dateTime } from './api'
import {
  activeStatuses,
  statusLabels,
  type Artifact,
  type GalleryPage,
  type Job,
  type SamplePoint,
} from './types'
import './samples.css'

interface Props {
  job?: Job
  jobs: Job[]
  onSelectJob: (job: Job) => void
  onBack: () => void
}

const pageSize = 40

function pointLabel(point: SamplePoint) {
  return `${point.epoch !== null ? `EP ${point.epoch.toLocaleString('zh-CN', { maximumFractionDigits: 2 })} · ` : ''}Step ${point.step}`
}

function imageUrl(image: Artifact, thumbnail = false) {
  return `/api/jobs/${image.job_id}/image/${image.path.split('/').map(encodeURIComponent).join('/')}?v=${image.modified}${thumbnail ? '&thumbnail=true' : ''}`
}

function downloadUrl(image: Artifact) {
  return `/api/jobs/${image.job_id}/artifacts/${image.path.split('/').map(encodeURIComponent).join('/')}`
}

export default function SamplePreview({
  job,
  jobs,
  onSelectJob,
  onBack,
}: Props) {
  const trainingJobs = jobs.filter((item) => item.kind === 'train')
  const current = job?.kind === 'train' ? job : trainingJobs[0]
  return (
    <main className="sample-page">
      <header className="sample-page-heading">
        <div>
          <h1>采样预览</h1>
          <p>
            {current
              ? `${current.project_name} · ${statusLabels[current.status] || current.status}`
              : '选择训练记录查看采样图'}
          </p>
        </div>
        <div className="sample-page-actions">
          <label>
            <span>训练记录</span>
            <select
              aria-label="选择采样训练记录"
              value={current?.id || ''}
              onChange={(event) => {
                const target = trainingJobs.find(
                  (item) => item.id === event.target.value,
                )
                if (target) onSelectJob(target)
              }}
            >
              {!current && <option value="">暂无训练记录</option>}
              {trainingJobs.map((item) => (
                <option key={item.id} value={item.id}>
                  {dateTime(item.started_at || item.created_at)} ·{' '}
                  {statusLabels[item.status]}
                </option>
              ))}
            </select>
          </label>
          <button className="button secondary" onClick={onBack}>
            <ArrowLeft size={15} />
            返回训练
          </button>
        </div>
      </header>
      {current ? (
        <Gallery key={current.id} job={current} />
      ) : (
        <div className="large-empty">
          <h2>还没有训练记录</h2>
          <p>训练生成的采样图会按采样点保存在这里。</p>
        </div>
      )}
    </main>
  )
}

function Gallery({ job }: { job: Job }) {
  const [page, setPage] = useState(0)
  const [data, setData] = useState<GalleryPage | null>(null)
  const [known, setKnown] = useState<Record<string, SamplePoint>>({})
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [compareId, setCompareId] = useState('')
  const [lightboxId, setLightboxId] = useState<string | null>(null)
  const [lightboxIndex, setLightboxIndex] = useState(0)
  const [error, setError] = useState('')
  const [newPoint, setNewPoint] = useState<SamplePoint | null>(null)
  const latestId = useRef<string | null>(null)
  const historyRevision = useRef('')
  const active = activeStatuses.has(job.status)
  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout>
    let alive = true
    async function refresh() {
      try {
        const result = await api<GalleryPage>(
          `/jobs/${job.id}/gallery?offset=${page * pageSize}&limit=${pageSize}&history_revision=${historyRevision.current}`,
          { signal: controller.signal },
        )
        if (!alive) return
        setData(result)
        setError('')
        historyRevision.current = result.history_revision
        setKnown((previous) =>
          result.history
            ? Object.fromEntries(result.history.map((point) => [point.id, point]))
            : {
                ...previous,
                ...Object.fromEntries(
                  result.points.map((point) => [point.id, point]),
                ),
              },
        )
        if (result.points.length) {
          setSelectedId((previous) => previous || result.points[0].id)
          if (page === 0) {
            if (latestId.current && latestId.current !== result.points[0].id)
              setNewPoint(result.points[0])
            latestId.current = result.points[0].id
          }
        }
      } catch (reason) {
        if (alive && !controller.signal.aborted)
          setError(
            reason instanceof Error ? reason.message : '采样记录读取失败',
          )
      }
      if (alive) timer = setTimeout(refresh, active ? 3500 : 12000)
    }
    void refresh()
    return () => {
      alive = false
      controller.abort()
      clearTimeout(timer)
    }
  }, [job.id, page, active])
  const points = Object.values(known).sort(
    (a, b) => b.step - a.step || b.source_started_at - a.source_started_at,
  )
  const selected = points.find((point) => point.id === selectedId)
  const comparison = points.find(
    (point) => point.id === compareId && point.id !== selectedId,
  )
  const lightbox = points.find((point) => point.id === lightboxId)
  function select(point: SamplePoint) {
    setSelectedId(point.id)
    if (point.id === compareId) setCompareId('')
    if (point.id === newPoint?.id) setNewPoint(null)
  }
  function neighbour(point: SamplePoint, direction: number) {
    return points[points.findIndex((item) => item.id === point.id) + direction]
  }
  function expand(point: SamplePoint, index: number) {
    setLightboxId(point.id)
    setLightboxIndex(index)
  }
  return (
    <div className="sample-browser">
      <aside className="sample-timeline" aria-label="采样点列表">
        <div className="sample-timeline-heading">
          <h2>采样点</h2>
          <span>
            {data?.total ?? '—'} 个 · {data?.runs.length ?? '—'} 段训练
          </span>
        </div>
        <div className="sample-point-list">
          {data?.points.map((point) => (
            <button
              type="button"
              className={point.id === selectedId ? 'selected' : ''}
              key={point.id}
              onClick={() => select(point)}
              aria-pressed={point.id === selectedId}
            >
              <img
                src={imageUrl(point.images[0], true)}
                alt=""
                loading="lazy"
              />
              <span>
                <strong>{pointLabel(point)}</strong>
                <small>{dateTime(point.modified)}</small>
                <small>
                  {point.inherited ? '续训来源' : '本次运行'} ·{' '}
                  {dateTime(point.source_started_at)}
                </small>
              </span>
            </button>
          ))}
        </div>
        {data && data.total > pageSize && (
          <div className="sample-pagination">
            <button
              className="text-button"
              disabled={page === 0}
              onClick={() => setPage((value) => value - 1)}
            >
              上一页
            </button>
            <span>
              {page + 1} / {Math.ceil(data.total / pageSize)}
            </span>
            <button
              className="text-button"
              disabled={data.next_offset === null}
              onClick={() => setPage((value) => value + 1)}
            >
              下一页
            </button>
          </div>
        )}
      </aside>
      <section className="sample-view" aria-label="采样图查看">
        {error && (
          <div className="inline-error" role="alert">
            {error}
          </div>
        )}
        {newPoint && (
          <button
            className="sample-new-image"
            onClick={() => {
              select(newPoint)
              setPage(0)
            }}
          >
            {pointLabel(newPoint)} 已生成，点击查看
          </button>
        )}
        {!data ? (
          <p className="empty-message">正在读取采样记录…</p>
        ) : !data.total ? (
          <div className="large-empty">
            <h2>等待采样图</h2>
            <p>本次运行及关联续训记录中的采样图会自动加入。</p>
            <p>
              本任务每{' '}
              {Number(
                job.config.native_overrides.SAMPLE_EVERY ??
                  job.config.sample_every,
              )}{' '}
              步采样一次。
            </p>
          </div>
        ) : (
          selected && (
            <>
              <div className="sample-view-toolbar">
                <div className="sample-step-controls">
                  <button
                    className="button secondary"
                    disabled={!neighbour(selected, 1)}
                    onClick={() => {
                      const point = neighbour(selected, 1)
                      if (point) select(point)
                    }}
                  >
                    <ArrowLeft size={15} />
                    上一采样点
                  </button>
                  <button
                    className="button secondary"
                    disabled={!neighbour(selected, -1)}
                    onClick={() => {
                      const point = neighbour(selected, -1)
                      if (point) select(point)
                    }}
                  >
                    下一采样点
                    <ArrowRight size={15} />
                  </button>
                </div>
                <label>
                  并排对比
                  <select
                    aria-label="选择对比采样点"
                    value={compareId}
                    onChange={(event) => setCompareId(event.target.value)}
                  >
                    <option value="">选择采样点</option>
                    {points
                      .filter((point) => point.id !== selectedId)
                      .map((point) => (
                        <option key={point.id} value={point.id}>
                          {pointLabel(point)}
                        </option>
                      ))}
                  </select>
                </label>
              </div>
              <div
                className={`sample-comparison${comparison ? ' comparing' : ''}`}
              >
                <ImagePanel
                  key={selected.id}
                  point={selected}
                  onExpand={(index) => expand(selected, index)}
                />
                {comparison && (
                  <ImagePanel
                    key={comparison.id}
                    point={comparison}
                    onExpand={(index) => expand(comparison, index)}
                  />
                )}
              </div>
              <details className="sample-settings">
                <summary>采样设置与来源</summary>
                <dl>
                  <div>
                    <dt>训练记录</dt>
                    <dd>{selected.job_id}</dd>
                  </div>
                  <div>
                    <dt>种子</dt>
                    <dd>{selected.settings.seeds?.join(', ') || '未记录'}</dd>
                  </div>
                  <div>
                    <dt>采样步数 / CFG</dt>
                    <dd>
                      {selected.settings.steps ?? '—'} /{' '}
                      {selected.settings.cfg ?? '—'}
                    </dd>
                  </div>
                  <div>
                    <dt>尺寸（高 × 宽）</dt>
                    <dd>{selected.settings.sizes?.join(' × ') || '未记录'}</dd>
                  </div>
                  <div>
                    <dt>提示词</dt>
                    <dd>
                      {selected.settings.prompts?.join('\n\n') || '未记录'}
                    </dd>
                  </div>
                </dl>
              </details>
            </>
          )
        )}
      </section>
      {lightbox && (
        <Lightbox
          point={lightbox}
          initialIndex={lightboxIndex}
          onClose={() => setLightboxId(null)}
          previous={neighbour(lightbox, 1)}
          next={neighbour(lightbox, -1)}
          onNavigate={(point) => expand(point, 0)}
        />
      )}
    </div>
  )
}

function ImagePanel({
  point,
  onExpand,
}: {
  point: SamplePoint
  onExpand: (index: number) => void
}) {
  const [index, setIndex] = useState(0)
  const image = point.images[index] || point.images[0]
  return (
    <article className="sample-image-panel">
      <header>
        <h2>{pointLabel(point)}</h2>
        <span>{dateTime(point.modified)}</span>
      </header>
      <button
        className="sample-image-surface"
        onClick={() => onExpand(index)}
        aria-label={`放大查看 ${pointLabel(point)}`}
      >
        <img
          src={imageUrl(image)}
          alt={`${pointLabel(point)} 采样图 ${index + 1}`}
        />
      </button>
      <footer>
        {point.images.length > 1 && (
          <label>
            图片
            <select
              aria-label={`${pointLabel(point)} 图片`}
              value={index}
              onChange={(event) => setIndex(Number(event.target.value))}
            >
              {point.images.map((item, i) => (
                <option key={item.path} value={i}>
                  {i + 1} / {point.images.length}
                </option>
              ))}
            </select>
          </label>
        )}
        <button className="text-button" onClick={() => onExpand(index)}>
          <Expand size={15} />
          放大查看
        </button>
        <a className="text-button" href={downloadUrl(image)}>
          <Download size={15} />
          下载原图
        </a>
      </footer>
    </article>
  )
}

function Lightbox({
  point,
  initialIndex,
  onClose,
  previous,
  next,
  onNavigate,
}: {
  point: SamplePoint
  initialIndex: number
  onClose: () => void
  previous?: SamplePoint
  next?: SamplePoint
  onNavigate: (point: SamplePoint) => void
}) {
  const dialog = useRef<HTMLDialogElement>(null)
  const [original, setOriginal] = useState(false)
  const [index, setIndex] = useState(initialIndex)
  useEffect(() => {
    dialog.current?.showModal()
  }, [])
  useEffect(() => {
    setIndex(initialIndex)
  }, [point.id, initialIndex])
  const image = point.images[index] || point.images[0]
  return (
    <dialog
      ref={dialog}
      className="sample-lightbox"
      onCancel={onClose}
      aria-label="采样原图"
      onKeyDown={(event) => {
        if (event.target instanceof HTMLSelectElement) return
        if (event.key === 'ArrowLeft' && previous) {
          event.preventDefault()
          onNavigate(previous)
        }
        if (event.key === 'ArrowRight' && next) {
          event.preventDefault()
          onNavigate(next)
        }
      }}
    >
      <header>
        <h2>{pointLabel(point)}</h2>
        <div>
          <button
            className="button secondary"
            onClick={() => setOriginal((value) => !value)}
          >
            {original ? '适应窗口' : '100% 查看'}
          </button>
          <a className="button secondary" href={downloadUrl(image)}>
            <Download size={15} />
            下载原图
          </a>
          <button
            className="icon-button"
            onClick={onClose}
            aria-label="关闭采样原图"
          >
            <X size={20} />
          </button>
        </div>
      </header>
      <div
        className={`sample-lightbox-image${original ? ' original-size' : ''}`}
      >
        <img src={imageUrl(image)} alt={pointLabel(point)} />
      </div>
      <footer>
        <button
          className="button secondary"
          disabled={!previous}
          onClick={() => previous && onNavigate(previous)}
        >
          <ArrowLeft size={15} />
          上一采样点
        </button>
        {point.images.length > 1 && (
          <label>
            图片
            <select
              aria-label="原图编号"
              value={index}
              onChange={(event) => setIndex(Number(event.target.value))}
            >
              {point.images.map((item, i) => (
                <option key={item.path} value={i}>
                  {i + 1} / {point.images.length}
                </option>
              ))}
            </select>
          </label>
        )}
        <span>{dateTime(point.modified)}</span>
        <button
          className="button secondary"
          disabled={!next}
          onClick={() => next && onNavigate(next)}
        >
          下一采样点
          <ArrowRight size={15} />
        </button>
      </footer>
    </dialog>
  )
}
