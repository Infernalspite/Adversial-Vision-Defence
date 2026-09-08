import { CircleHelp, Wifi } from 'lucide-react'

export function Header({ title }: { title: string }) {
  return <header className="app-header"><div><span className="eyebrow">ARGUS-AEGIS / OPERATIONS</span><h1>{title}</h1></div><div className="header-actions"><span className="online"><Wifi size={14} /> API LINKED</span><button className="icon-button" title="System information"><CircleHelp size={17} /></button></div></header>
}
