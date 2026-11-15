export interface TrainSettings {
  dataset_path: string
  source: 'folder' | 'kohya'
  caption_format: 'nl' | 'tags'
  train_repeat: number
  reg_enabled: boolean
  reg_dataset_path: string
  reg_source: 'folder' | 'kohya'
  reg_caption_format: 'nl' | 'tags'
  reg_repeat: number
  reg_loss_weight: number
  transformer_path: string
  text_encoder_path: string
  vae_path: string
  algorithm: 'lora' | 'locon' | 'lokr' | 'loha' | 'dora' | 'oft' | 'boft' | 'glora' | 'ia3' | 'dylora' | 'tlora' | 'full'
  rank: number
  alpha: number
  factor: number
  learning_rate: number
  steps: number
  resolution: number
  max_tokens: number
  batching: 'tokens' | 'count'
  batch_size: number
  precision: 'bf16' | 'fp16'
  memory_profile: string
  grad_acc: number
  warmup: number
  save_every: number
  sample_every: number
  sample_prompt: string
  sample_negative: string
  sample_steps: number
  sample_cfg: number
  sample_seed: number
  seed: number
  text_drop: number
  caption_variants: number
  workers: number
  cache_mode: 'offline' | 'online' | 'off'
  tune: 'table' | 'off'
  native_overrides: Record<string, unknown>
  tool_options: Record<string, Record<string, unknown>>
  tool_args: Record<string, string[]>
}

export const datasetSettingKeys = [
  'dataset_path', 'source', 'caption_format', 'train_repeat', 'reg_enabled',
  'reg_dataset_path', 'reg_source', 'reg_caption_format', 'reg_repeat', 'native_overrides',
] as const

export interface DatasetAnalysis {
  checked_at: number
  root: string
  images: number
  captions: number
  missing_captions: number
  invalid_images: number
  effective_samples?: number
  issues: { file: string; kind: string; message: string }[]
  preview: { path: string; width: number; height: number; caption: string }[]
}

export interface Analysis extends DatasetAnalysis {
  regularization?: DatasetAnalysis | null
  datasets?: DatasetAnalysis[]
}

export interface Project {
  id: string
  name: string
  config: TrainSettings
  analysis: Analysis | null
  created_at: number
  updated_at: number
  archived?: number
}

export type JobKind = string
export interface Job {
  id: string
  project_id: string
  project_name: string
  kind: JobKind
  status: string
  phase: string
  error: string | null
  config: TrainSettings
  folder: string
  output_dir: string
  origin: 'studio' | 'external'
  can_retry: number
  run_label: string
  created_at: number
  started_at: number | null
  finished_at: number | null
}

export interface Artifact { name: string; path: string; size?: number; modified: number; job_id?: string; step?: number | null; inherited?: boolean; source_started_at?: number }
export interface SamplePoint {
  id: string
  step: number
  epoch: number | null
  job_id: string
  source_started_at: number
  inherited: boolean
  modified: number
  images: Artifact[]
  settings: { prompts?: string[]; seeds?: number[]; steps?: number; cfg?: number; sizes?: number[] }
}
export interface GalleryPage {
  points: SamplePoint[]
  history: SamplePoint[] | null
  history_revision: string
  total: number
  next_offset: number | null
  runs: { id: string; started_at: number | null; cutoff: number | null }[]
}
export interface JobDetail {
  job: Job
  metrics: Record<string, number>[]
  artifacts: { weights: Artifact[]; samples: Artifact[]; files?: Artifact[]; checkpoints?: Artifact[] }
}

export interface NativeField {
  name: string
  default: unknown
  group: string
  help: string
  choices?: string[]
}

export interface NativeSchema {
  script: string
  fields: NativeField[]
  values: Record<string, unknown>
  defaults: Record<string, unknown>
  doc: string
}

export interface Operation {
  id: string
  label: string
  group: string
  script: string
  driver: string
  linux: boolean
}

export interface Environment {
  root: string
  python: string
  ready: boolean
  checked_at: number | null
  packages: Record<string, string | null>
  gpu: { name: string; total_mib: number; used_mib: number; utilization: number } | null
  assets: { key: string; path: string; name: string; exists: boolean; size: number }[]
  doctor: { missing?: { what: string; fix: string }[]; backend_failures?: Record<string, string> }
}

export const activeStatuses = new Set(['preparing', 'queued', 'starting', 'running', 'stopping'])
export const kindLabels: Record<JobKind, string> = {
  check: '数据检查', cache: '建立缓存', train: '训练', doctor: '环境检测', text_encoder_benchmark: '文字编码器测速',
}
export const statusLabels: Record<string, string> = {
  preparing: '准备任务', queued: '排队中', starting: '正在启动', running: '运行中', stopping: '正在停止', completed: '已完成',
  failed: '失败', cancelled: '已停止', interrupted: '已中断', imported: '已导入',
}
export const phaseLabels: Record<string, string> = {
  validate: '校验配置', check: '数据检查', text_cache: '文本缓存',
  latent_cache: 'Latent 缓存', train: '训练', doctor: '环境检测',
  loading: '加载模型', warmup: '预热', encoding: '编码测速',
}
