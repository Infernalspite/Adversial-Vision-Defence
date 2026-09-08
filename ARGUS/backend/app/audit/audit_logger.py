"""Audit logging boundary."""

import json
import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AuditEvent:
    """One append-only security-relevant event."""

    request_id: str
    stage: str
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: dict[str, Any] = field(default_factory=dict)
    record_hash: str | None = None
    previous_record_hash: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize the event for an API or durable sink."""
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload

    def canonical_payload(self) -> str:
        payload = self.as_dict(); payload.pop("record_hash", None)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


class AuditLogger:
    """Small in-memory structured audit sink for the MVP."""

    def __init__(self, path: str | Path = "data/audit/audit_records.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.events: list[AuditEvent] = []
        self._load()

    def record(self, event: AuditEvent) -> None:
        """Record metadata without retaining image bytes."""
        event.previous_record_hash = self.events[-1].record_hash if self.events else None
        event.record_hash = hashlib.sha256(event.canonical_payload().encode("utf-8")).hexdigest()
        self.events.append(event)
        self.path.write_text(json.dumps([item.as_dict() for item in self.events], indent=2, default=str), encoding="utf-8")

    def for_request(self, request_id: str) -> list[dict[str, Any]]:
        """Return serialized events for one request."""
        return [event.as_dict() for event in self.events if event.request_id == request_id]

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent audit records."""
        return [event.as_dict() for event in self.events[-max(1, min(limit, 100)) :]][::-1]

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw_records = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for record in raw_records if isinstance(raw_records, list) else []:
            try:
                timestamp = datetime.fromisoformat(record["timestamp"])
                self.events.append(AuditEvent(record["request_id"], record["stage"], record["message"], timestamp, record.get("details", {}), record.get("record_hash"), record.get("previous_record_hash")))
            except (KeyError, TypeError, ValueError):
                continue

    def verify_integrity(self) -> bool:
        """Verify the simple tamper-evident hash chain."""
        previous = None
        for event in self.events:
            if event.previous_record_hash != previous: return False
            expected = hashlib.sha256(event.canonical_payload().encode("utf-8")).hexdigest()
            if event.record_hash != expected: return False
            previous = event.record_hash
        return True
