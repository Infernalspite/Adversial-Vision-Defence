import type { ReactNode } from 'react'

export function MetricCard({ label, value, detail, tone = 'cyan', children }: { label: string; value: string; detail?: string; tone?: 'cyan' | 'amber' | 'green' | 'red'; children?: ReactNode }) {
  return <article className={`metric-card tone-${tone}`}><div className="metric-label">{label}</div><div className="metric-value">{value}</div>{detail && <div className="metric-detail">{detail}</div>}{children}</article>
}
