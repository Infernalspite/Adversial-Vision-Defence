"""Audit API contracts."""

from datetime import datetime
from pydantic import BaseModel, Field


class AuditRecord(BaseModel):
    """Explainable record for one request."""

    request_id: str
    stage: str
    message: str
    timestamp: datetime
    details: dict[str, object] = Field(default_factory=dict)
