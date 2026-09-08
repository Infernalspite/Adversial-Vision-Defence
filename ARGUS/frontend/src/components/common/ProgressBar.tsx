export function ProgressBar({ value, tone = 'cyan' }: { value: number; tone?: 'cyan' | 'amber' | 'green' | 'red' }) {
  return <div className="progress-track" aria-label={`${Math.round(value * 100)} percent`}><span className={`progress-fill tone-${tone}`} style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} /></div>
}
