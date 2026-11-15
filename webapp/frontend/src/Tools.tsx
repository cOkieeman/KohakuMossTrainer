import { useEffect, useState } from 'react'
import { Check, Download, Play, Save } from 'lucide-react'
import { api } from './api'
import Monitor from './Monitor'
import SchemaEditor, { JsonInput } from './SchemaEditor'
import type { Environment, Job, NativeSchema, Operation, TrainSettings } from './types'

const names: Record<string, string> = {
  sample: '独立推理', compare: '检查点对比图', export: '检查点导出', merge: '合并 LoRA 与底模',
  fp16_fix: 'FP16 数值修复', fp8_quant: 'FP8 / MXFP8 量化', comfy_int8: 'INT8 ConvRot 导出', text_quant: '文本编码器量化',
  catalog_build: '建立数据目录', catalog_cache: '目录缓存管理', cache_text: '文本缓存', cache_latents: '潜空间缓存', cache_merge: '缓存分片合并', image_index: '图片索引',
  heal_cache: 'Heal 数据缓存', heal_calibrate: 'Heal 校准', heal_rescale: 'Heal 重缩放', heal_merge: 'Heal 合并', heal_train: 'Heal 训练',
  tune: '内核调优', check_comm: '多 GPU 通信检查', scaling_report: '多 GPU 扩展报告',
}
const groupNames: Record<string, string> = { inference: '推理与对比', export: '导出与量化', data: '数据与缓存', heal: 'Heal 工具', system: '运行环境', distributed: '多 GPU', bench: '基准测试', probe: '模型探针' }

interface Props {
  value: TrainSettings
  environment: Environment
  job?: Job
  projectId: string
  busy: boolean
  onChange: (value: TrainSettings) => void
  onRun: (kind: string) => void
  onSave: () => void
  onStop: (job: Job) => void
  onMessage: (message: string, error?: boolean) => void
}

export default function Tools({ value, environment, job, projectId, busy, onChange, onRun, onSave, onStop, onMessage }: Props) {
  const [operations, setOperations] = useState<Operation[]>([])
  const [kind, setKind] = useState('sample')
  const [schema, setSchema] = useState<NativeSchema | null>(null)
  const [error, setError] = useState('')
  const [validation, setValidation] = useState<string[] | null>(null)
  const [help, setHelp] = useState('')
  const operation = operations.find(item => item.id === kind)
  const baseConfig = JSON.stringify({ ...value, tool_options: {}, tool_args: {} })
  useEffect(() => { api<{ operations: Operation[] }>('/native/operations').then(data => setOperations(data.operations)).catch(reason => setError(String(reason))) }, [])
  useEffect(() => {
    const controller = new AbortController()
    setSchema(null); setError(''); setValidation(null); setHelp('')
    api<NativeSchema>(`/native/resolve/${kind}?project_id=${encodeURIComponent(projectId)}`, { method: 'POST', body: baseConfig, signal: controller.signal }).then(setSchema)
      .catch(reason => { if (!controller.signal.aborted) setError(String(reason.message || reason)) })
    return () => controller.abort()
  }, [kind, baseConfig, projectId])
  async function validate() {
    if (!(document.getElementById('native-tool-form') as HTMLFormElement | null)?.reportValidity()) return
    try { setValidation((await api<{ errors: string[] }>(`/native/validate/${kind}?project_id=${encodeURIComponent(projectId)}`, { method: 'POST', body: JSON.stringify(value) })).errors) }
    catch (reason) { setError(String(reason)) }
  }
  const groups = [...new Set(operations.map(item => item.group))]
  return <main className="tools-main"><div className="tools-editor"><header className="native-panel-heading"><h1>原生工具</h1><a className="text-button" href={`/api/projects/${projectId}/native/${kind}.py`}><Download size={14} />导出已保存配置</a></header>
    <label className="operation-picker"><span>工具</span><select aria-label="选择原生工具" value={kind} onChange={event => setKind(event.target.value)}>
      {groups.map(group => <optgroup key={group} label={groupNames[group] || group}>{operations.filter(item => item.group === group).map(item => <option key={item.id} value={item.id}>{names[item.id] || `${groupNames[group] || group} · ${item.script.replace(/^scripts\/(bench|probe)\//, '').replace(/\.py$/, '')}`}</option>)}</optgroup>)}
    </select></label>
    {operation && <p className="native-script-path">{operation.script}{operation.linux && ' · 原生执行需要 Linux'}</p>}
    <form id="native-tool-form" onSubmit={event => { event.preventDefault(); onRun(kind) }}>
      <div className="tool-scroll">
        {error && <div className="inline-error" role="alert">{error}</div>}
        {validation && <div className={validation.length ? 'inline-error' : 'validation-ok'} role="status">{validation.length ? validation.map((item, index) => <p key={index}>{item}</p>) : '参数格式检查通过，模型加载与执行结果见任务日志'}</div>}
        {schema && operation?.driver === 'python' ? <section className="cli-arguments"><h2>原生命令行参数</h2><p>用 JSON 字符串数组填写参数，每个选项和值分别占一项。</p>
          <JsonInput label="命令行参数" stringArray value={value.tool_args[kind] || []} onChange={next => {
            setValidation(null)
            if (Array.isArray(next) && next.every(item => typeof item === 'string')) onChange({ ...value, tool_args: { ...value.tool_args, [kind]: next } })
          }} />
          <button type="button" className="text-button" onClick={() => api<{ text: string }>(`/native/help/${kind}`).then(data => setHelp(data.text)).catch(reason => setError(String(reason)))}>读取原生参数帮助</button>
          <pre className="native-help">{help || schema.doc}</pre>
        </section> : schema ? <SchemaEditor schema={schema} overrides={value.tool_options[kind] || {}} onChange={next => { setValidation(null); onChange({ ...value, tool_options: { ...value.tool_options, [kind]: next } }) }} /> : <p className="empty-message">正在读取工具参数…</p>}
      </div>
      <footer className="tool-actions"><button className="button secondary" type="button" onClick={onSave} disabled={busy}><Save size={15} />保存工具配置</button><button className="button secondary" type="button" onClick={validate} disabled={!schema}><Check size={15} />检查配置</button><button type="submit" className="button primary" disabled={busy || !schema}><Play size={15} />提交任务</button></footer>
    </form>
  </div><Monitor job={job} environment={environment} onStop={onStop} onMessage={onMessage} /></main>
}
