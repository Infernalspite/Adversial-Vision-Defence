"""Health endpoints."""

from fastapi import APIRouter

from app.audit.audit_logger import AuditLogger
from app.models.model_registry import model_registry
from app.security.security_config import security_settings

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health() -> dict[str, object]:
    """Return non-sensitive service health details."""
    audit = AuditLogger()
    return {"status": "ok", "service": "argus-aegis", "model_loaded": "resnet18" in model_registry._models, "configuration": "ok", "audit": "ok" if audit.verify_integrity() else "degraded"}


@router.get("/ready")
def ready() -> dict[str, object]:
    """Return readiness without forcing an expensive model load."""
    return {"ready": True, "security_limits_configured": security_settings.max_image_bytes > 0}
