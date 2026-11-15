import { useEffect, useState } from 'react'
import { Check, Download, RefreshCw } from 'lucide-react'
import { api } from './api'
import SchemaEditor from './SchemaEditor'
import type { NativeSchema, TrainSettings } from './types'

interface Props {
  value: TrainSettings
  onChange: (value: Record<string, unknown>) => void
  projectId: string
}

export default function NativePanel({ value, onChange, projectId }: Props) {
  const [schema, setSchema] = useState<NativeSchema | null>(null)
  const [error, setError] = useState('')
  const [validation, setValidation] = useState<string[] | null>(null)
  const [busy, setBusy] = useState(false)
  const basic = JSON.stringify({ ...value, native_overrides: {}, tool_options: {}, tool_args: {} })
  useEffect(() => {
    const controller = new AbortController()
    api<NativeSchema>('/native/resolve/train', { method: 'POST', body: basic, signal: controller.signal }).then(setSchema).catch(reason => { if (!controller.signal.aborted) setError(String(reason.message)) })
    return () => controller.abort()
  }, [basic])
  async function validate() {
    if (!(document.getElementById('training-form') as HTMLFormElement | null)?.reportValidity()) return
    setBusy(true)
    try { setValidation((await api<{ errors: string[] }>('/native/validate/train', { method: 'POST', body: JSON.stringify(value) })).errors) }
    catch (reason) { setError(String(reason)) }
    finally { setBusy(false) }
  }
  return <section className="native-panel"><header className="native-panel-heading"><h2>完整训练参数</h2><button type="button" className="button secondary" onClick={validate} disabled={busy}>{busy ? <RefreshCw size={15} className="spin" /> : <Check size={15} />}校验配置</button></header>
    <p className="native-explanation">覆盖值优先生效。修改基础字段时，对应的原生覆盖会被清除。列表与字典使用 JSON。</p>
    <div className="native-links"><a href={`/api/projects/${projectId}/config`}><Download size={14} />导出已保存的 Python 配置</a><a href={`/api/projects/${projectId}/studio.json`}>导出项目 JSON</a></div>
    {error && <div className="inline-error" role="alert">{error}</div>}
    {validation && <div className={validation.length ? 'inline-error' : 'validation-ok'} role="status">{validation.length ? validation.map((item, index) => <p key={index}>{item}</p>) : '原生配置校验通过'}</div>}
    {schema ? <SchemaEditor schema={schema} overrides={value.native_overrides} onChange={next => { setValidation(null); onChange(next) }} /> : <p className="empty-message">正在读取训练器参数…</p>}
  </section>
}
