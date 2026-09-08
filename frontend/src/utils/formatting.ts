export function formatPercent(value: number | null): string {
  return value === null ? '—' : `${Math.round(value * 100)}%`
}

export function formatDuration(seconds: number | null): string {
  return seconds === null ? '—' : `${seconds.toFixed(2)}s`
}
