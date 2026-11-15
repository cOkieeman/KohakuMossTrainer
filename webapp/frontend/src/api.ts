export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...options?.headers },
  })
  if (!response.ok) {
    const value = await response.json().catch(() => ({ detail: response.statusText }))
    const detail = Array.isArray(value.detail)
      ? value.detail.map((item: { loc: string[]; msg: string }) => `${item.loc.slice(1).join('.')} ${item.msg}`).join('\n')
      : value.detail
    throw new Error(detail || `请求失败 (${response.status})`)
  }
  return response.json() as Promise<T>
}

export function dateTime(value?: number | null) {
  return value ? new Date(value * 1000).toLocaleString('zh-CN', { hour12: false }) : '—'
}

export function bytes(value: number) {
  return value > 1e9 ? `${(value / 1e9).toFixed(2)} GB` : `${(value / 1e6).toFixed(1)} MB`
}

export function elapsed(start?: number | null, end?: number | null) {
  if (!start) return '—'
  const seconds = Math.max(0, Math.floor((end || Date.now() / 1000) - start))
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  return hours ? `${hours}h ${minutes}m` : `${minutes}m ${seconds % 60}s`
}

export function fileName(path: string) {
  return path.split(/[\\/]/).pop() || path
}
