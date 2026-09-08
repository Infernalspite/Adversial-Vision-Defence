"""Audit record retrieval endpoints."""

from fastapi import APIRouter, Query

from app.api.schemas.audit import AuditRecord
from app.audit.audit_logger import AuditLogger

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("", response_model=list[AuditRecord])
def list_audit(limit: int = Query(default=20, ge=1, le=100)) -> list[AuditRecord]:
    """Return recent persisted audit records."""
    return AuditLogger().recent(limit)


@router.get("/{request_id}", response_model=list[AuditRecord])
def get_audit(request_id: str) -> list[AuditRecord]:
    """Fetch audit events for a request."""
    return AuditLogger().for_request(request_id)
