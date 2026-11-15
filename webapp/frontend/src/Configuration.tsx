import { useEffect, useState, type ReactNode } from 'react'
import { ChevronDown, Download, FolderOpen, Settings2 } from 'lucide-react'
import { fileName } from './api'
import DataSummary from './DataSummary'
import Regularization from './Regularization'
import NativePanel from './NativePanel'
import { datasetSettingKeys, type Project, type TrainSettings } from './types'

type Change = <K extends keyof TrainSettings>(key: K, value: TrainSettings[K]) => void
type NumericKey = { [K in keyof TrainSettings]: TrainSettings[K] extends number ? K : never }[keyof TrainSettings]
interface Props {
  project: Project
  value: TrainSettings
  onChange: Change
  onBrowse: (key: 'dataset_path' | 'reg_dataset_path' | 'transformer_path' | 'text_encoder_path' | 'vae_path') => void
  onSave: () => void
  onConfig: () => void
  stage: number
}

function Field({ label, children, wide = false }: { label: string; children: ReactNode; wide?: boolean }) {
  return <label className={`field${wide ? ' wide-field' : ''}`}><span>{label}</span>{children}</label>
}

export default function Configuration({ project, value, onChange, onBrowse, onSave, onConfig, stage }: Props) {
  const [advanced, setAdvanced] = useState(false)
  const analysis = datasetSettingKeys.every(key => JSON.stringify(project.config[key]) === JSON.stringify(value[key])) ? project.analysis : null
  const showAdvanced = advanced
  useEffect(() => { if (stage === 1) setAdvanced(true) }, [stage])
  const number = (key: NumericKey, label: string, minimum = 0, step: number | string = 1) =>
    <Field key={key} label={label}><input type="number" required min={minimum} step={step} value={value[key]}
      onChange={event => onChange(key, Number(event.target.value))} /></Field>

  if (stage === 4) return <form id="training-form" onSubmit={event => { event.preventDefault(); onSave() }}>
    <NativePanel value={value} projectId={project.id} onChange={next => onChange('native_overrides', next)} />
  </form>

  return <form id="training-form" onSubmit={event => { event.preventDefault(); onSave() }}>
    <section className="form-section" id="data-settings">
      <h2>数据与底模</h2>
      {Array.isArray(value.native_overrides.DATASETS) && <p className="dataset-hint">数据来源由完整参数中的 DATASETS 提供。编辑基础数据设置会清除该覆盖。</p>}
      <div className="path-field">
        <label htmlFor="dataset-path">数据集目录</label>
        <input id="dataset-path" value={value.dataset_path} onChange={event => onChange('dataset_path', event.target.value)}
          placeholder="选择图片与标注目录" spellCheck={false} />
        <button type="button" className="button secondary" onClick={() => onBrowse('dataset_path')}><FolderOpen size={16} />浏览</button>
      </div>
      <div className="field-grid three-columns">
        <Field label="源数据格式"><select value={value.source} onChange={event => onChange('source', event.target.value as TrainSettings['source'])}>
          <option value="folder">图片文件夹</option><option value="kohya">Kohya 重复次数目录</option>
        </select></Field>
        <Field label="标注格式"><select value={value.caption_format} onChange={event => onChange('caption_format', event.target.value as TrainSettings['caption_format'])}>
          <option value="nl">自然语言</option><option value="tags">标签</option>
        </select></Field>
        {number('train_repeat', '训练重复系数', 0.0001, 'any')}
      </div>
      {analysis && <DataSummary analysis={analysis} projectId={project.id} />}
      {Array.isArray(value.native_overrides.DATASETS) && analysis?.datasets?.slice(1).map((item, index) => <section key={index} className="regularization-settings"><h3>数据源 {index + 2}</h3><DataSummary analysis={item} projectId={project.id} role={String(index + 1)} /></section>)}
      <Regularization value={value} analysis={analysis} projectId={project.id} onChange={onChange} onBrowse={() => onBrowse('reg_dataset_path')} />
      <div className="path-field model-field">
        <label htmlFor="transformer-path">底模模型</label>
        <input id="transformer-path" title={value.transformer_path} value={value.transformer_path}
          onChange={event => onChange('transformer_path', event.target.value)} spellCheck={false} />
        <button type="button" className="icon-button bordered" onClick={() => onBrowse('transformer_path')} aria-label="选择底模文件"><FolderOpen size={17} /></button>
      </div>
      <p className="model-note">{fileName(value.transformer_path)}<span>MOSS-Image</span></p>
    </section>

    <section className="form-section" id="train-settings">
      <h2>训练参数</h2>
      <div className="field-grid three-columns">
        <Field label="算法"><select value={value.algorithm} onChange={event => onChange('algorithm', event.target.value as TrainSettings['algorithm'])}>
          <option value="lora">LoRA</option><option value="lokr">LoKr</option><option value="loha">LoHa</option><option value="dora">DoRA</option><option value="oft">OFT</option>
          <option value="locon">LoCon</option><option value="boft">BOFT</option><option value="glora">GLoRA</option><option value="ia3">IA3</option><option value="dylora">DyLoRA</option><option value="tlora">TLoRA</option><option value="full">全量微调</option>
        </select></Field>
        {number('rank', value.algorithm === 'oft' ? 'Block size' : 'Rank', 1)}
        {number('alpha', 'Alpha', 0.01, 'any')}
        {number('learning_rate', '学习率', 0.00000001, 'any')}
        {number('steps', '总步数', 1)}
        {number('resolution', '目标分辨率', 16, 16)}
        {value.batching === 'count' ? number('batch_size', 'Batch size', 1) : number('max_tokens', 'Token 预算', 2)}
        <Field label="计算精度"><select value={value.precision} onChange={event => onChange('precision', event.target.value as TrainSettings['precision'])}>
          <option value="bf16">BF16</option><option value="fp16">FP16</option>
        </select></Field>
        <Field label="显存方案"><select value={value.memory_profile} onChange={event => onChange('memory_profile', event.target.value as TrainSettings['memory_profile'])}>
          {!['low', 'minimal', 'balanced', 'speed', 'auto'].includes(value.memory_profile) && <option value={value.memory_profile}>{value.memory_profile}</option>}
          <option value="low">16GB / low</option><option value="minimal">最省显存</option><option value="balanced">均衡</option><option value="speed">速度优先</option><option value="auto">自动选择</option>
        </select></Field>
        {value.algorithm === 'lokr' && number('factor', 'LoKr factor', 1)}
      </div>
    </section>

    <section className="advanced-section" id="cache-settings">
      <button className="section-toggle" type="button" aria-expanded={showAdvanced} onClick={() => setAdvanced(!showAdvanced)}>
        <ChevronDown size={17} className={showAdvanced ? 'rotated' : ''} /><span>{stage === 1 ? '缓存与更多设置' : '更多设置'}</span><Settings2 size={16} />
      </button>
      {showAdvanced && <div className="advanced-content">
        <h3>缓存与执行</h3>
        <div className="field-grid two-columns">
          <Field label="批次方式"><select value={value.batching} onChange={event => onChange('batching', event.target.value as TrainSettings['batching'])}>
            <option value="tokens">按 Token 预算打包</option><option value="count">固定图片张数</option>
          </select></Field>
          {value.batching === 'count' && number('max_tokens', '单图 Token 上限', 2)}
          <Field label="缓存方式"><select value={value.cache_mode} onChange={event => onChange('cache_mode', event.target.value as TrainSettings['cache_mode'])}>
            <option value="offline">离线缓存，开训前自动准备</option><option value="online">在线缓存</option><option value="off">逐批编码</option>
          </select></Field>
          <Field label="内核调优"><select value={value.tune} onChange={event => onChange('tune', event.target.value as TrainSettings['tune'])}>
            <option value="table">自动调优并缓存</option><option value="off">关闭计时调优</option>
          </select></Field>
          {number('grad_acc', '梯度累积', 1)}{number('warmup', '预热步数')}
          {number('workers', '数据线程')}{number('caption_variants', '标注缓存份数', 1)}
          {number('text_drop', '空提示比例', 0, 'any')}{number('seed', '随机种子')}
          {number('save_every', '保存间隔')}
        </div>
        <h3>训练采样</h3>
        <Field label="正面提示词" wide><textarea rows={3} value={value.sample_prompt} onChange={event => onChange('sample_prompt', event.target.value)} placeholder="输入提示词后启用定期采样出图" /></Field>
        <Field label="负面提示词" wide><textarea rows={2} value={value.sample_negative} onChange={event => onChange('sample_negative', event.target.value)} /></Field>
        <div className="field-grid two-columns">
          {number('sample_every', '采样间隔')}{number('sample_steps', '采样步数', 1)}
          {number('sample_cfg', 'CFG', 0, 'any')}{number('sample_seed', '采样种子')}
        </div>
        <h3>配套权重</h3>
        {(['text_encoder_path', 'vae_path'] as const).map(key => <div className="path-field" key={key}>
          <label htmlFor={key}>{key === 'vae_path' ? 'VAE' : '文本编码器'}</label>
          <input id={key} value={value[key]} onChange={event => onChange(key, event.target.value)} spellCheck={false} />
          <button className="icon-button bordered" type="button" onClick={() => onBrowse(key)} aria-label={key === 'vae_path' ? '选择 VAE 文件' : '选择文本编码器文件'}><FolderOpen size={17} /></button>
        </div>)}
      </div>}
    </section>
    <div className="config-tools"><button type="button" onClick={onConfig}><Download size={15} />预览与导出配置</button><span>配置与任务保存在本机</span></div>
  </form>
}
