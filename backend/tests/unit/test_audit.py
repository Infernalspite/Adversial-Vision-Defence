from datetime import datetime, timezone

from app.audit.audit_logger import AuditEvent, AuditLogger


def test_audit_logger_persists_and_retrieves_records(tmp_path):
    path = tmp_path / "audit_records.json"
    logger = AuditLogger(path)
    event = AuditEvent("request-1", "decision", "ABSTAIN", datetime.now(timezone.utc), {"final_state": "ABSTAIN"})
    logger.record(event)
    assert logger.for_request("request-1")[0]["details"]["final_state"] == "ABSTAIN"
    reloaded = AuditLogger(path)
    assert reloaded.recent(10)[0]["request_id"] == "request-1"
