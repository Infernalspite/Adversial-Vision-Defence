import type { AuditRecord } from '../../types'

export function EvidencePanel({ record }: { record: AuditRecord | null }) { return <section className="panel evidence-panel"><div className="section-kicker">EVIDENCE DETAIL</div>{record ? <><h2>{record.stage.replaceAll('_', ' ')}</h2><p>{record.message}</p><code>{JSON.stringify(record.details, null, 2)}</code></> : <p className="muted">Select an audit record to inspect its evidence.</p>}</section> }
