import { useEffect, useState } from 'react'
import { api } from './api'
import { activeStatuses, type JobDetail } from './types'

export function useTask(jobId?: string) {
  const [detail, setDetail] = useState<JobDetail | null>(null)
  const [log, setLog] = useState('')
  const [error, setError] = useState('')
  useEffect(() => {
    setDetail(null)
    setLog('')
    setError('')
    if (!jobId) return
    let alive = true
    let offset = -1
    let timer: ReturnType<typeof setTimeout>
    const controller = new AbortController()
    async function poll() {
      try {
        const next = await api<JobDetail>(`/jobs/${jobId}`, { signal: controller.signal })
        const chunk = await api<{ text: string; offset: number; reset: boolean; more: boolean }>(`/jobs/${jobId}/log?offset=${offset}`, { signal: controller.signal })
        if (!alive) return
        offset = chunk.offset
        setDetail(next)
        setError('')
        if (chunk.text || chunk.reset) {
          setLog(previous => `${chunk.reset ? '' : previous}${chunk.text}`.slice(-220_000))
        }
        if (chunk.more || activeStatuses.has(next.job.status)) timer = setTimeout(poll, chunk.more ? 100 : 1800)
      } catch (reason) {
        if (!alive) return
        setError(reason instanceof Error ? reason.message : '读取任务失败')
        timer = setTimeout(poll, 3000)
      }
    }
    void poll()
    return () => { alive = false; controller.abort(); clearTimeout(timer) }
  }, [jobId])
  return { detail, log, error }
}
