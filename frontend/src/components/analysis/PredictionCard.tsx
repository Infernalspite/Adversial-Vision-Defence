import type { ArgusAnalysisResult } from '../../types'
import { MetricCard } from '../common/MetricCard'
import { formatPercent } from '../../utils/formatting'

export function PredictionCard({ result }: { result: ArgusAnalysisResult | null }) {
  if (!result) return <MetricCard label="MODEL INTERPRETATION" value="Awaiting input" detail="Original and defended predictions will appear here." />
  const recovered = result.original_prediction.class_id === result.defended_prediction.class_id
  return <article className="panel insight-card"><div className="section-kicker">MODEL INTERPRETATION</div><div className="prediction-columns"><div><span>ORIGINAL</span><strong>{result.original_prediction.class_name}</strong><b>{formatPercent(result.original_confidence)}</b></div><div><span>AFTER DEFENSE</span><strong>{result.defended_prediction.class_name}</strong><b>{formatPercent(result.defended_confidence)}</b></div></div><div className={recovered ? 'recovery-pill positive' : 'recovery-pill'}>{recovered ? '✓ PREDICTION RECOVERED' : '↗ INTERPRETATION CHANGED'}</div></article>
}
