import type { ReactNode } from 'react'

export function PredictionCard() { return <Panel title="BASE PREDICTION"><div className="text-2xl font-bold text-white">Not evaluated</div><div className="mt-2 text-sm text-slate-400">Confidence —</div></Panel> }
function Panel({ title, children }: { title: string; children: ReactNode }) { return <section className="rounded-lg border border-slate-700/60 bg-slate-900/70 p-5"><p className="mono text-xs tracking-widest text-slate-500">{title}</p><div className="mt-4">{children}</div></section> }
