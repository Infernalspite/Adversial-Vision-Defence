import type { ArgusAnalysisResult } from '../../types'
import { ProgressBar } from '../common/ProgressBar'
import { formatPercent } from '../../utils/formatting'

export function AttackScoreCard({ result }: { result: ArgusAnalysisResult | null }) {
  if (!result) return <article className="panel score-card"><div className="section-kicker">ADVERSARIAL RISK</div><div className="score-empty">—</div><p>Run analysis to collect detector evidence.</p></article>
  return <article className="panel score-card"><div className="section-kicker">ADVERSARIAL RISK</div><div className="score-line"><strong>{formatPercent(result.attack_score)}</strong><span className={result.attack_detected ? 'status-badge danger' : 'status-badge calm'}>{result.attack_detected ? 'DETECTED' : 'NOT DETECTED'}</span></div><div className="threshold-line">MVP threshold {formatPercent(result.attack_detection_threshold)}</div><div className="detector-list">{Object.entries(result.detectors).map(([name, detector]) => <div className="detector-row" key={name}><div><span>{name.replaceAll('_', ' ')}</span><b>{formatPercent(detector.score)}</b></div><ProgressBar value={detector.score} tone={detector.detected ? 'red' : 'amber'} /></div>)}</div></article>
}
