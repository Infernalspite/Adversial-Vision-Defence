import type { AuditRecord } from '../../types'

export function AuditEvent({ record, onClick }: { record: AuditRecord; onClick: () => void }) { return <button className="audit-record-row" onClick={onClick}><span>{new Date(record.timestamp).toLocaleString()}</span><b>{record.request_id.slice(0, 8)}</b><span>{record.stage}</span><span>{record.message}</span></button> }
