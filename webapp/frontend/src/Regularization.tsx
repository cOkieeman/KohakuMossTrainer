import { FolderOpen } from 'lucide-react'
import DataSummary from './DataSummary'
import type { Analysis, TrainSettings } from './types'

interface Props {
  value: TrainSettings
  analysis: Analysis | null
  projectId: string
  onChange: <K extends keyof TrainSettings>(key: K, value: TrainSettings[K]) => void
  onBrowse: () => void
}

export default function Regularization({ value, analysis, projectId, onChange, onBrowse }: Props) {
  const regularization = analysis?.regularization
  const trainItems = analysis?.effective_samples ?? analysis?.images ?? 0
  const regItems = regularization?.effective_samples ?? regularization?.images ?? 0
  return <section className="regularization-settings" aria-label="正则数据集设置">
    <label className="regularization-toggle"><input type="checkbox" checked={value.reg_enabled}
      onChange={event => onChange('reg_enabled', event.target.checked)} /><span>使用正则数据集</span></label>
    {value.reg_enabled && <div className="regularization-content">
      <div className="path-field">
        <label htmlFor="reg-dataset-path">正则目录</label>
        <input id="reg-dataset-path" value={value.reg_dataset_path} spellCheck={false} placeholder="选择本项目使用的正则子集"
          onChange={event => onChange('reg_dataset_path', event.target.value)} />
        <button type="button" className="button secondary" onClick={onBrowse} aria-label="浏览正则目录"><FolderOpen size={16} />浏览</button>
      </div>
      <div className="field-grid two-columns">
        <label className="field"><span>正则格式</span><select value={value.reg_source} onChange={event => onChange('reg_source', event.target.value as TrainSettings['reg_source'])}>
          <option value="folder">图片文件夹</option><option value="kohya">Kohya 重复次数目录</option>
        </select></label>
        <label className="field"><span>正则标注</span><select value={value.reg_caption_format} onChange={event => onChange('reg_caption_format', event.target.value as TrainSettings['reg_caption_format'])}>
          <option value="tags">标签</option><option value="nl">自然语言</option>
        </select></label>
        <label className="field"><span>正则重复系数</span><input type="number" required min="0.0001" step="any" value={value.reg_repeat}
          onChange={event => onChange('reg_repeat', Number(event.target.value))} /></label>
        <label className="field"><span>正则损失权重</span><input type="number" required min="0" step="any" value={value.reg_loss_weight}
          onChange={event => onChange('reg_loss_weight', Number(event.target.value))} /></label>
      </div>
      <p className="dataset-hint">按两组有效样本数调整重复系数，可设置约 1∶1 的混合采样。</p>
      {regularization && <>
        <p className="mix-summary">每轮预计训练 {trainItems.toLocaleString('zh-CN', { maximumFractionDigits: 2 })} 份 · 正则 {regItems.toLocaleString('zh-CN', { maximumFractionDigits: 2 })} 份</p>
        <DataSummary analysis={regularization} projectId={projectId} role="regularization" />
      </>}
    </div>}
  </section>
}
