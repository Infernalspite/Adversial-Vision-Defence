"""Phase 3 adversarial detection endpoint."""

from dataclasses import asdict
from uuid import uuid4

from fastapi import APIRouter, Form, HTTPException, UploadFile, status

from app.api.schemas.detection import DetectionResponse
from app.core.config import settings
from app.detectors.detection_pipeline import DetectionPipeline
from app.models.model_registry import get_model
from app.security.input_validator import InputValidationError, validate_image_payload

router = APIRouter(prefix="/detection", tags=["detection"])


@router.post("/analyze", response_model=DetectionResponse)
async def analyze_detection(
    image: UploadFile,
    detectors: str | None = Form(default=None),
    threshold: float | None = Form(default=None),
) -> DetectionResponse:
    """Analyze an uploaded image for evidence of adversarial manipulation."""
    payload = await image.read()
    if len(payload) > settings.max_image_bytes:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Image is too large")
    try:
        decoded = validate_image_payload(payload, image.filename, image.content_type)
        detector_names = [item.strip() for item in detectors.split(",") if item.strip()] if detectors else None
        result = DetectionPipeline().analyze(decoded, get_model("resnet18"), detector_names, threshold)
    except (InputValidationError, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    return DetectionResponse(
        request_id=str(uuid4()),
        attack_detected=result.attack_detected,
        attack_score=result.attack_score,
        detection_threshold=result.detection_threshold,
        detectors={name: asdict(detector) for name, detector in result.detectors.items()},
        explanation=result.explanation,
        processing_time_ms=result.processing_time_ms,
    )