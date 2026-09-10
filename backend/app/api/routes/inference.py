"""Baseline inference and complete ARGUS-AEGIS analysis endpoints."""

import base64
from dataclasses import asdict
from uuid import uuid4

from fastapi import APIRouter, Form, HTTPException, UploadFile, status

from app.api.schemas.decision import AnalyzeResponse
from app.api.schemas.inference import InferenceResponse
from app.core.config import settings
from app.models.model_registry import get_model, model_registry, person_aware
from app.models.tier_router import route_tier
from app.pipeline.pipeline import ArgusPipeline, PipelineContext
from app.security.input_validator import InputValidationError, validate_image_payload
from app.utils.image import encode_image
from app.utils.visualization import heatmap_overlay, pixel_trace_summary, suspicion_heatmap_png, trust_map_png

router = APIRouter(prefix="/inference", tags=["inference"])
analysis_router = APIRouter(tags=["analysis"])


@router.post("", response_model=InferenceResponse)
async def infer(image: UploadFile) -> InferenceResponse:
    """Validate, decode, and classify one uploaded image."""
    payload = await image.read()
    try:
        decoded_image = validate_image_payload(payload, image.filename, image.content_type)
    except InputValidationError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error

    prediction = person_aware(get_model("resnet18")).predict(decoded_image)
    return InferenceResponse(
        request_id=str(uuid4()),
        model_name="resnet18",
        predicted_class=prediction.class_name,
        predicted_class_id=prediction.class_id,
        confidence=prediction.confidence,
        top_predictions=[
            {
                "class_id": item.class_id,
                "class_name": item.class_name,
                "confidence": item.confidence,
            }
            for item in prediction.top_predictions
        ],
        inference_time_ms=prediction.inference_time_ms,
    )


def _prediction_payload(prediction: object) -> dict[str, object]:
    return {"class_id": prediction.class_id, "class_name": prediction.class_name, "confidence": prediction.confidence}


def _person_evidence(model: object) -> dict[str, object] | None:
    """Person-detector evidence for the last analyzed image (None if the
    detector feature is unavailable or the model is not person-aware)."""
    getter = getattr(model, "last_person_detection", None)
    if getter is None:
        return None
    detection = getter()
    detector = getattr(model, "_detector", None)
    detector_available = detector is not None and detector.load_error is None
    return {
        "detector_available": detector_available,
        "present": detection.present,
        "count": detection.count,
        "max_confidence": detection.max_confidence,
        "boxes": [asdict(box) for box in detection.boxes],
        "inference_time_ms": detection.inference_time_ms,
    }


def _reference(payload: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(payload).decode('ascii')}"


@analysis_router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    image: UploadFile,
    mode: str = Form("standard"),
    attack_type: str | None = Form(default=None),
) -> AnalyzeResponse:
    """Run the complete Phase 3-7 pipeline and return exactly one final state."""
    payload = await image.read()
    if len(payload) > settings.max_image_bytes:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Image is too large")
    try:
        decoded = validate_image_payload(payload, image.filename, image.content_type)
        request_id = str(uuid4())
        baseline_model = person_aware(get_model("resnet18"))
        baseline_prediction = baseline_model.predict(decoded)
        robust_model = model_registry.get("robust") if model_registry.available("robust") else None
        routing = route_tier(baseline_prediction, baseline_model, robust_model)
        person_evidence = _person_evidence(baseline_model)
        result = ArgusPipeline().run(PipelineContext(request_id, decoded, {"model": routing.analysis_model, "mode": mode, "attack_type": attack_type, "tier": routing.tier, "tier_reason": routing.reason}))
        trust_reference = _reference(trust_map_png(result.defense.trust.trust_map))
        overlay_reference = _reference(heatmap_overlay(decoded, result.defense.trust.trust_map, result.defense.trust.localization.regions))
        suspicion_reference = _reference(suspicion_heatmap_png(result.defense.trust.trust_map))
        defended_reference = _reference(encode_image(result.defense.orchestration.defense.defended_image))
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    verification = result.verification.verification
    decision = result.decision
    trace = result.defense.orchestration.trace
    return AnalyzeResponse(
        request_id=request_id,
        tier=routing.tier,
        tier_reason=routing.reason,
        matched_defended_class=routing.matched_defended_class,
        person_detection=person_evidence,
        suspicion_heatmap_reference=suspicion_reference,
        pixel_trace=pixel_trace_summary(result.defense.trust.trust_map, result.defense.trust.localization.regions),
        attack_score=result.defense.detection.attack_score,
        attack_detected=result.defense.detection.attack_detected,
        attack_detection_threshold=result.defense.detection.detection_threshold,
        detectors={name: asdict(item) for name, item in result.defense.detection.detectors.items()},
        global_trust_score=float(result.defense.trust.trust_map.values.mean()),
        trust_map_reference=trust_reference,
        trust_map_overlay_reference=overlay_reference,
        suspicious_regions=result.defense.trust.localization.regions,
        defense_applied=result.defense.orchestration.defense.defense_applied,
        defense_method=result.defense.orchestration.decision.selected_defense,
        defended_image_reference=defended_reference,
        original_prediction=_prediction_payload(result.defense.original_prediction),
        defended_prediction=_prediction_payload(result.defense.defended_prediction),
        original_confidence=result.defense.original_prediction.confidence,
        defended_confidence=result.defense.defended_prediction.confidence,
        prediction_changed=result.defense.prediction_changed,
        object_consistency={"score": verification.object_consistency["object_consistency_score"], "evidence": verification.object_consistency},
        geometry_consistency={"score": verification.geometry_consistency["structural_similarity"], "evidence": verification.geometry_consistency},
        scene_consistency={"score": verification.scene_consistency["scene_similarity_score"], "evidence": verification.scene_consistency},
        verification_score=verification.verification_score,
        final_state=decision.final_state,
        final_prediction=decision.final_prediction,
        final_confidence=decision.final_confidence,
        decision_score=decision.decision_score,
        decision_reasons=list(decision.decision_reasons),
        explanation=decision.explanation,
        fallback_action=result.fallback.action if result.fallback else None,
        fallback_reason=result.fallback.reason if result.fallback else None,
        defense_trace={
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
        },
        audit=result.audit,
        processing_time_ms=result.defense.processing_time_ms + result.verification.processing_time_ms,
    )
