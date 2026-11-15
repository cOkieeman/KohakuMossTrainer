import { fileName } from './api'
import type { DatasetAnalysis } from './types'

interface Props {
  analysis: DatasetAnalysis
  projectId: string
  role?: string
}

export default function DataSummary({ analysis, projectId, role = 'train' }: Props) {
  return <div className="dataset-results">
    <div className="dataset-summary">
      <span><strong>{analysis.images}</strong> 张图片</span>
      <span><strong>{analysis.captions}</strong> 份标注</span>
      <span className={analysis.missing_captions || analysis.invalid_images ? 'warning-text' : ''}>
        {analysis.missing_captions ? `${analysis.missing_captions} 张缺少标注` : '标注已配对'}
        {analysis.invalid_images > 0 && ` · ${analysis.invalid_images} 张无法读取`}
      </span>
    </div>
    {analysis.preview.length > 0 && <div className="dataset-preview">
      {analysis.preview.slice(0, 8).map(item => <figure key={item.path} title={`${item.path}\n${item.width} × ${item.height}\n${item.caption}`}>
        <img src={`/api/projects/${projectId}/thumbnail?dataset=${role}&relative=${encodeURIComponent(item.path)}`} alt={item.path} loading="lazy" />
        <figcaption>{fileName(item.path)}</figcaption>
      </figure>)}
    </div>}
    {analysis.issues.length > 0 && <details className="issue-details"><summary>查看检查提示</summary>
      <ul>{analysis.issues.map(issue => <li key={`${issue.file}-${issue.kind}`}><span className="path-text">{issue.file}</span><span>{issue.kind === 'caption' ? '标注缺失或为空' : '图片或标注无法读取'}</span></li>)}</ul>
    </details>}
  </div>
}
