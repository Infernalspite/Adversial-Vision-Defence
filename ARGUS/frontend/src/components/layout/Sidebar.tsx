import { Activity, BarChart3, BrainCircuit, FileClock, FlaskConical, LayoutDashboard, ScanSearch, ShieldCheck } from 'lucide-react'

export type AppPage = 'dashboard' | 'attack-lab' | 'analysis' | 'audit' | 'evaluation' | 'learning'

export function Sidebar({ page, onNavigate }: { page: AppPage; onNavigate: (page: AppPage) => void }) {
  const links = [
    ['dashboard', 'Dashboard', LayoutDashboard],
    ['attack-lab', 'Attack Lab', FlaskConical],
    ['analysis', 'Analysis', ScanSearch],
    ['audit', 'Audit Trail', FileClock],
    ['evaluation', 'Evaluation', BarChart3],
    ['learning', 'Learning', BrainCircuit],
  ] as const
  return <aside className="sidebar"><div className="brand"><ShieldCheck size={24} /><div><strong>ARGUS-AEGIS</strong><span>visual defense fabric</span></div></div><nav>{links.map(([id, label, Icon]) => <button key={id} className={page === id ? 'nav-item active' : 'nav-item'} onClick={() => onNavigate(id)}><Icon size={17} />{label}</button>)}</nav><div className="sidebar-status"><Activity size={15} /><span>LOCAL ENGINE ONLINE</span></div></aside>
}
