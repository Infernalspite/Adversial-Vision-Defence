export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return <div className="error-state"><strong>Analysis unavailable</strong><p>{message}</p>{onRetry && <button className="button button-secondary" onClick={onRetry}>Retry</button>}</div>
}
