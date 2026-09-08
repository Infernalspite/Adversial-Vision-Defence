import { useState } from 'react'
import { Dashboard } from './pages/Dashboard'
import { AttackLab } from './pages/AttackLab'
import { Analysis } from './pages/Analysis'
import { Audit } from './pages/Audit'
import { Evaluation } from './pages/Evaluation'
import { Learning } from './pages/Learning'
import { Sidebar, type AppPage } from './components/layout/Sidebar'
import { Header } from './components/layout/Header'
import { PageContainer } from './components/layout/PageContainer'
import { useAnalysis } from './hooks/useAnalysis'

export default function App() {
  const [page, setPage] = useState<AppPage>('dashboard')
  const analysis = useAnalysis()
  const titles: Record<AppPage, string> = { dashboard: 'Mission control', 'attack-lab': 'Attack Lab', analysis: 'Analysis dossier', audit: 'Audit trail', evaluation: 'Evaluation lab', learning: 'Learning workbench' }
  return <div className="app-shell"><Sidebar page={page} onNavigate={setPage} /><div className="main-shell"><Header title={titles[page]} /><PageContainer>{page === 'dashboard' && <Dashboard previewUrl={analysis.previewUrl} result={analysis.result} loading={analysis.loading} error={analysis.error} onSelect={analysis.selectImage} onReset={analysis.reset} onRun={analysis.runAnalysis} onRetry={analysis.runAnalysis} />} {page === 'attack-lab' && <AttackLab onAnalysis={result => { analysis.adoptResult(result); setPage('analysis') }} />} {page === 'analysis' && <Analysis result={analysis.result} />} {page === 'audit' && <Audit />} {page === 'evaluation' && <Evaluation />} {page === 'learning' && <Learning />}</PageContainer></div></div>
}
