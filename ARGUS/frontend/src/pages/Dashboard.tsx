import type { ArgusAnalysisResult } from '../types'
import { ImageUploader } from '../components/analysis/ImageUploader'
import { ImageComparison } from '../components/analysis/ImageComparison'
import { PredictionCard } from '../components/analysis/PredictionCard'
import { AttackScoreCard } from '../components/analysis/AttackScoreCard'
import { TrustScoreCard } from '../components/analysis/TrustScoreCard'
import { TrustMapViewer } from '../components/analysis/TrustMapViewer'
import { SuspiciousRegions } from '../components/analysis/SuspiciousRegions'
import { DefenseCard } from '../components/analysis/DefenseCard'
import { VerificationCard } from '../components/analysis/VerificationCard'
import { DecisionCard } from '../components/analysis/DecisionCard'
import { AuditTimeline } from '../components/audit/AuditTimeline'
import { ErrorState } from '../components/common/ErrorState'
import { LoadingState } from '../components/common/LoadingState'
import { SystemStatus } from '../components/common/SystemStatus'

export function Dashboard({ previewUrl, result, loading, error, onSelect, onReset, onRun, onRetry }: { previewUrl: string | null; result: ArgusAnalysisResult | null; loading: boolean; error: string | null; onSelect: (file: File) => void; onReset: () => void; onRun: () => void; onRetry: () => void }) {
  return <><section className="hero-band"><div><span className="eyebrow cyan">AUTONOMOUS VISUAL DEFENSE FABRIC</span><h2>Mission control for uncertain vision.</h2><p>Trace the evidence chain from image input to a conservative final decision.</p></div><button className="button button-primary run-button" disabled={!previewUrl || loading} onClick={onRun}>{loading ? 'ANALYZING…' : 'RUN ANALYSIS'}</button></section><SystemStatus /><section className="input-row"><ImageUploader previewUrl={previewUrl} onSelect={onSelect} onReset={onReset} disabled={loading} />{loading && <LoadingState />}{error && <ErrorState message={error} onRetry={onRetry} />}</section><ImageComparison originalUrl={previewUrl} result={result} /><div className="metric-grid"><PredictionCard result={result} /><AttackScoreCard result={result} /><TrustScoreCard result={result} /></div><div className="analysis-grid"><TrustMapViewer result={result} /><DecisionCard result={result} /></div><div className="analysis-grid lower"><SuspiciousRegions result={result} /><DefenseCard result={result} /></div><VerificationCard result={result} /><AuditTimeline result={result} /></>
}
