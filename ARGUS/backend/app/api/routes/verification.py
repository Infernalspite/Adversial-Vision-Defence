"""Phase 6 semantic verification endpoint."""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, HTTPException, UploadFile, status

from app.api.schemas.verification import VerificationResponse
from app.core.config import settings
from app.models.model_registry import get_model
from app.security.input_validator import InputValidationError, validate_image_payload
from app.verification.verification_pipeline import VerificationPipeline

router = APIRouter(prefix="/verification", tags=["verification"])


def _prediction_payload(prediction: object) -> dict[str, object]:
    """Serialize a model prediction into the public contract."""
    return {
        "class_id": prediction.class_id,
        "class_name": prediction.class_name,
        "confidence": prediction.confidence,
    }


@router.post("/analyze", response_model=VerificationResponse)
async def analyze_verification(original_image: UploadFile, defended_image: UploadFile) -> VerificationResponse:
    """Compare original and defended images with the Semantic Reality Anchor."""
    original_payload = await original_image.read()
    defended_payload = await defended_image.read()
    if len(original_payload) > settings.max_image_bytes or len(defended_payload) > settings.max_image_bytes:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Image is too large")
    try:
        original = validate_image_payload(original_payload, original_image.filename, original_image.content_type)
        defended = validate_image_payload(defended_payload, defended_image.filename, defended_image.content_type)
        model = get_model("resnet18")
        original_prediction = model.predict(original)
        defended_prediction = model.predict(defended)
        result = VerificationPipeline().run(original, defended, original_prediction, defended_prediction)
    except (InputValidationError, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    verification = result.verification
    return VerificationResponse(
        request_id=str(uuid4()),
        original_prediction=_prediction_payload(result.original_prediction),
        original_confidence=result.original_prediction.confidence,
        defended_prediction=_prediction_payload(result.defended_prediction),
        defended_confidence=result.defended_prediction.confidence,
        prediction_changed=result.prediction_changed,
        confidence_change=result.confidence_change,
        top_k_overlap=result.top_k_overlap,
        object_consistency={"score": verification.object_consistency["object_consistency_score"], "evidence": verification.object_consistency},
        geometry_consistency={"score": verification.geometry_consistency["structural_similarity"], "evidence": verification.geometry_consistency},
        scene_consistency={"score": verification.scene_consistency["scene_similarity_score"], "evidence": verification.scene_consistency},
        verification_score=verification.verification_score,
        explanation=verification.explanation,
        processing_time_ms=result.processing_time_ms,
    )
