import { useEffect, useState } from 'react'
import { getAuditRecords } from '../services/api'
import type { AuditRecord } from '../types'
import { AuditEvent } from '../components/audit/AuditEvent'
import { EvidencePanel } from '../components/audit/EvidencePanel'
import { ErrorState } from '../components/common/ErrorState'
import { LoadingState } from '../components/common/LoadingState'

export function Audit() { const [records, setRecords] = useState<AuditRecord[]>([]); const [selected, setSelected] = useState<AuditRecord | null>(null); const [error, setError] = useState<string | null>(null); const [loading, setLoading] = useState(true); async function load() { setLoading(true); try { setRecords(await getAuditRecords()) } catch (cause) { setError(cause instanceof Error ? cause.message : 'Audit records unavailable.') } finally { setLoading(false) } } useEffect(() => { void load() }, []); return <><section className="page-intro"><span className="eyebrow violet">CHAIN OF CUSTODY</span><h2>Audit trail</h2><p>Persisted decision metadata. Image bytes are never stored in the audit record.</p></section>{loading && <LoadingState />}{error && <ErrorState message={error} onRetry={load} />}{!loading && !error && <div className="audit-layout"><section className="panel audit-records"><div className="panel-heading"><div><div className="section-kicker">RECENT RECORDS</div><h2>Analysis history</h2></div><button className="button button-secondary" onClick={load}>Refresh</button></div>{records.length ? records.map(record => <AuditEvent key={`${record.request_id}-${record.timestamp}`} record={record} onClick={() => setSelected(record)} />) : <p className="muted">No analyses have been persisted yet.</p>}</section><EvidencePanel record={selected} /></div>}</> }
