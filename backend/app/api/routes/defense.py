"""Phase 5 defense endpoints."""

from __future__ import annotations

import base64
from uuid import uuid4

from fastapi import APIRouter, HTTPException, UploadFile, status

from app.api.schemas.defense import DefenseApplyResponse, DefenseSummary
from app.core.config import settings
from app.models.model_registry import get_model
from app.pipeline.defense_pipeline import ArgusDefensePipeline
from app.security.input_validator import InputValidationError, validate_image_payload
from app.utils.image import encode_image

router = APIRouter(prefix="/defense", tags=["defense"])


@router.get("/strategies", response_model=list[str])
def strategies() -> list[str]:
    """List planned defense strategies."""
    return ["transform", "mask", "purification", "certified"]


@router.post("/preview", response_model=DefenseSummary)
def preview() -> DefenseSummary:
    """Return the defense summary contract without running a defense."""
    return DefenseSummary()


def _reference(image: object) -> str:
    """Encode a defended image as a temporary response reference."""
    return f"data:image/png;base64,{base64.b64encode(encode_image(image)).decode('ascii')}"  # type: ignore[arg-type]


@router.post("/apply", response_model=DefenseApplyResponse)
async def apply_defense(image: UploadFile) -> DefenseApplyResponse:
    """Run detection, trust mapping, policy defense, and defended re-inference."""
    payload = await image.read()
    if len(payload) > settings.max_image_bytes:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Image is too large")
    try:
        decoded = validate_image_payload(payload, image.filename, image.content_type)
        result = ArgusDefensePipeline().run(decoded, get_model("resnet18"))
    except (InputValidationError, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    defended_reference = _reference(result.orchestration.defense.defended_image)
    trace = result.orchestration.trace
    trace_payload = {
        "timestamp": trace.timestamp,
        "attack_score": trace.attack_score,
        "selected_defense": trace.selected_defense,
        "reason": trace.reason,
        "input_trust_score": trace.input_trust_score,
        "suspicious_regions": trace.suspicious_regions,
        "defense_parameters": trace.defense_parameters,
        "processing_time_ms": trace.processing_time_ms,
        "success": trace.success,
        "output_image_reference": defended_reference,
    }
    return DefenseApplyResponse(
        request_id=str(uuid4()),
        original_prediction={
            "class_id": result.original_prediction.class_id,
            "class_name": result.original_prediction.class_name,
            "confidence": result.original_prediction.confidence,
        },
        original_confidence=result.original_prediction.confidence,
        attack_score=result.detection.attack_score,
        attack_detected=result.detection.attack_detected,
        global_trust_score=float(result.trust.trust_map.values.mean()),
        suspicious_regions=result.trust.localization.regions,
        selected_defense=result.orchestration.decision.selected_defense,
        defense_reason=result.orchestration.decision.reason,
        defense_parameters=result.orchestration.decision.parameters,
        defense_applied=result.orchestration.defense.defense_applied,
        defended_prediction={
            "class_id": result.defended_prediction.class_id,
            "class_name": result.defended_prediction.class_name,
            "confidence": result.defended_prediction.confidence,
        },
        defended_confidence=result.defended_prediction.confidence,
        prediction_changed=result.prediction_changed,
        defended_image_reference=defended_reference,
        defense_trace=trace_payload,
        processing_time_ms=result.processing_time_ms,
    )
