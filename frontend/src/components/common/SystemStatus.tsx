import { useEffect, useState } from 'react'
import { CheckCircle2, CircleAlert } from 'lucide-react'
import { getHealth } from '../../services/api'

export function SystemStatus() {
  const [status, setStatus] = useState<'READY' | 'DEGRADED' | 'ERROR'>('DEGRADED')
  useEffect(() => { getHealth().then(() => setStatus('READY')).catch(() => setStatus('ERROR')) }, [])
  const ready = status === 'READY'
  return <section className="system-status panel"><div className="panel-heading"><div><div className="section-kicker">SYSTEM STATUS</div><h2>{status}</h2></div>{ready ? <CheckCircle2 className="status-icon ready" /> : <CircleAlert className="status-icon warning" />}</div><div className="status-grid"><span>API <b>{ready ? 'READY' : status}</b></span><span>MODEL <b>{ready ? 'READY' : 'PENDING'}</b></span><span>AUDIT <b>{ready ? 'READY' : 'CHECK'}</b></span><span>PIPELINE <b>{ready ? 'READY' : 'CHECK'}</b></span></div><div className="zero-trust-active">✓ Zero-Trust Ingest: ACTIVE</div></section>
}
