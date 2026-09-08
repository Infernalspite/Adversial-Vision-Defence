import type { ArgusAnalysisResult } from '../types'
import { AttackScoreCard } from '../components/analysis/AttackScoreCard'
import { TrustScoreCard } from '../components/analysis/TrustScoreCard'
import { TrustMapViewer } from '../components/analysis/TrustMapViewer'
import { SuspiciousRegions } from '../components/analysis/SuspiciousRegions'
import { DefenseCard } from '../components/analysis/DefenseCard'
import { VerificationCard } from '../components/analysis/VerificationCard'
import { DecisionCard } from '../components/analysis/DecisionCard'
import { AuditTimeline } from '../components/audit/AuditTimeline'

export function Analysis({ result }: { result: ArgusAnalysisResult | null }) { return <><section className="page-intro"><span className="eyebrow cyan">TECHNICAL BREAKDOWN</span><h2>Analysis dossier</h2><p>Every stage below is sourced from the backend pipeline. No frontend security scores are synthesized.</p></section><div className="metric-grid"><AttackScoreCard result={result} /><TrustScoreCard result={result} /><VerificationCard result={result} /></div><TrustMapViewer result={result} /><div className="analysis-grid lower"><SuspiciousRegions result={result} /><DefenseCard result={result} /></div><DecisionCard result={result} /><AuditTimeline result={result} /></> }
