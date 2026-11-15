import { useEffect, useRef, useState } from 'react'
import { FolderOpen, RotateCcw, Search } from 'lucide-react'
import FileBrowser from './FileBrowser'
import type { NativeField, NativeSchema } from './types'

const groupNames: Record<string, string> = {
  MODEL: '模型', DATA: '数据与标注', ADAPTER: '适配器与训练范围', OBJECTIVE: '训练目标与损失',
  OPTIM: '优化器与学习率', PRECISION: '计算精度', MEMORY: '显存与 Offload', PARALLEL: '多 GPU 与并行',
  RUNTIME: '运行', LOGGING: '日志与分析', SAMPLING: '训练采样与验证', 'CHECKPOINT / EXPORT': '检查点与导出',
  GENERAL: '参数', DEVICES: '设备与放置', PROMPTS: '提示词', SAMPLER: '采样器', OUTPUT: '输出', ENCODER: '编码器', KERNELS: '内核与编译',
}
const labels: Record<string, string> = {
  OPTIMIZER: '优化器', BETAS: 'Betas', EPS: 'Epsilon', WEIGHT_DECAY: '权重衰减', SCHEDULE: '学习率调度',
  GRAD_CLIP: '梯度裁剪', EMA: 'EMA', RESUME: '恢复检查点', INIT_FROM: '加载已有训练权重', DATASETS: '全部数据集',
  VALIDATION: '验证集与固定噪声', VALIDATE_EVERY: '验证间隔', PRESET: '训练层范围', DROPOUT: '适配器 Dropout',
  TIMESTEP: '时间步采样', TIMESTEP_KWARGS: '时间步参数', LOSS: '损失函数', LOSS_KWARGS: '损失参数',
  KEEP_LAST: '保留检查点数量', SAVE_MINUTES: '按分钟保存', SAVE_FORMAT: '导出格式', SAVE_MERGED: '同时合并底模',
  CHECKPOINT: '梯度检查点', STREAM_RESIDENT: '驻留块数', TEXT_ENCODE: '文本编码器计划', VAE_ENCODE: 'VAE 计划',
  PARALLEL: '并行方式', DEVICES: '训练设备', OUTPUT_DIR: '输出根目录', RUN_NAME: '运行名称', CACHE_DIR: '缓存目录',
}
const pathNames = new Set(['WEIGHTS', 'TRANSFORMER', 'TEXT_ENCODER', 'VAE', 'ADAPTER', 'OUT', 'DIT', 'RECORDS', 'TEACHER', 'RESUME', 'INIT_FROM', 'OUTPUT_DIR', 'CACHE_DIR', 'PROMPTS_FILE', 'DIFFUSERS_DIR', 'ROOT', 'RUN', 'PROBE', 'REPORT', 'EMBEDS', 'STUDENT'])

export function JsonInput({ value, onChange, label, id, stringArray = false }: { value: unknown; onChange: (value: unknown) => void; label: string; id?: string; stringArray?: boolean }) {
  const serialized = JSON.stringify(value, null, 2)
  const [text, setText] = useState(serialized)
  const committed = useRef(serialized)
  const area = useRef<HTMLTextAreaElement>(null)
  useEffect(() => {
    if (serialized !== committed.current) {
      setText(serialized); committed.current = serialized; area.current?.setCustomValidity('')
    }
  }, [serialized])
  return <textarea ref={area} id={id} className="native-json" aria-label={label} spellCheck={false} value={text} rows={typeof value === 'object' && value !== null ? 4 : 2}
    onChange={event => {
      setText(event.target.value)
      try {
        const parsed: unknown = JSON.parse(event.target.value)
        if (stringArray && (!Array.isArray(parsed) || !parsed.every(item => typeof item === 'string'))) {
          event.target.setCustomValidity('请输入 JSON 字符串数组')
          return
        }
        event.target.setCustomValidity('')
        committed.current = JSON.stringify(parsed, null, 2)
        onChange(parsed)
      }
      catch { event.target.setCustomValidity('请输入有效 JSON') }
    }} />
}

function Parameter({ field, value, changed, onChange, onReset }: { field: NativeField; value: unknown; changed: boolean; onChange: (value: unknown) => void; onReset: () => void }) {
  const [browse, setBrowse] = useState(false)
  const id = `native-${field.name}`
  const label = labels[field.name] || field.name
  const accessibleName = label === field.name ? label : `${label} (${field.name})`
  let control
  if (typeof value === 'boolean') control = <input id={id} type="checkbox" aria-label={accessibleName} checked={value} onChange={event => onChange(event.target.checked)} />
  else if (typeof value === 'number') control = <input id={id} type="number" step="any" required aria-label={accessibleName} value={value} onChange={event => onChange(Number(event.target.value))} />
  else if (typeof value === 'string') control = field.choices?.length ? <select id={id} aria-label={accessibleName} value={value} onChange={event => onChange(event.target.value)}>
    {!field.choices.includes(value) && <option value={value}>{value || '默认'}</option>}{field.choices.map(choice => <option key={choice} value={choice}>{choice}</option>)}
  </select> : <div className="native-path"><input id={id} aria-label={accessibleName} value={value} spellCheck={false} onChange={event => onChange(event.target.value)} />
    {pathNames.has(field.name) && <button className="icon-button bordered" type="button" aria-label={`浏览 ${field.name}`} onClick={() => setBrowse(true)}><FolderOpen size={16} /></button>}</div>
  else control = <JsonInput id={id} value={value} onChange={onChange} label={accessibleName} />
  return <div className={`native-parameter${changed ? ' overridden' : ''}`}>
    <div className="native-label"><label htmlFor={id}>{label}</label><code>{field.name}</code>
      {changed && <button type="button" className="icon-button" onClick={onReset} aria-label={`恢复 ${field.name}`} title="清除覆盖"><RotateCcw size={14} /></button>}</div>
    <div className="native-control">{control}{field.name === 'OUT' && <p>留空时，运行结果写入本次任务目录。</p>}{field.help && <p>{field.help}</p>}</div>
    {browse && <FileBrowser mode="all" title={`选择 ${field.name} 路径`} initial={typeof value === 'string' ? value : ''} onSelect={path => { onChange(path); setBrowse(false) }} onClose={() => setBrowse(false)} />}
  </div>
}

export default function SchemaEditor({ schema, overrides, onChange }: { schema: NativeSchema; overrides: Record<string, unknown>; onChange: (value: Record<string, unknown>) => void }) {
  const [query, setQuery] = useState('')
  const groups = new Map<string, NativeField[]>()
  for (const field of schema.fields) {
    if (query && !`${field.name} ${labels[field.name] || ''} ${field.help}`.toLowerCase().includes(query.toLowerCase())) continue
    const fields = groups.get(field.group) || []
    fields.push(field)
    groups.set(field.group, fields)
  }
  return <div className="schema-editor"><div className="native-search"><Search size={16} /><input aria-label="搜索原生参数" placeholder="搜索参数名或说明" value={query} onChange={event => setQuery(event.target.value)} /><span>{schema.fields.length} 项参数</span></div>
    {[...groups].map(([group, fields]) => <details className="native-group" key={group} open={query ? true : undefined}><summary>{groupNames[group] || group}<span>{fields.length}</span></summary>
      {fields.map(field => <Parameter key={field.name} field={field} value={Object.hasOwn(overrides, field.name) ? overrides[field.name] : schema.values[field.name]}
        changed={Object.hasOwn(overrides, field.name)} onChange={value => onChange({ ...overrides, [field.name]: value })}
        onReset={() => { const next = { ...overrides }; delete next[field.name]; onChange(next) }} />)}
    </details>)}
    {!groups.size && <p className="empty-message">没有匹配的参数</p>}
  </div>
}
