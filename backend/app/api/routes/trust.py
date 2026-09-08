"""Phase 4 spatial trust-map endpoint."""

from __future__ import annotations

import base64
import json
from uuid import uuid4

import cv2
from fastapi import APIRouter, Form, HTTPException, UploadFile, status

from app.api.schemas.trust import TrustMapAnalyzeResponse
from app.core.config import settings
from app.models.model_registry import get_model
from app.trust.trust_pipeline import TrustPipeline
from app.security.input_validator import InputValidationError, validate_image_payload
from app.utils.visualization import heatmap_overlay, trust_map_png

router = APIRouter(prefix="/trust-map", tags=["trust-map"])


def _data_reference(payload: bytes, media_type: str = "image/png") -> str:
    """Return a compact temporary data reference for an image artifact."""
    return f"data:{media_type};base64,{base64.b64encode(payload).decode('ascii')}"


@router.post("/analyze", response_model=TrustMapAnalyzeResponse)
async def analyze_trust_map(
    image: UploadFile,
    trust_weights: str | None = Form(default=None),
    localization_threshold: float = Form(0.30),
    minimum_region_area: int = Form(16),
    patch_mask: UploadFile | None = None,
) -> TrustMapAnalyzeResponse:
    """Run Phase 3 and produce spatial trust and suspicious regions."""
    if not 0 <= localization_threshold <= 1 or minimum_region_area < 1:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid localization settings")
    payload = await image.read()
    if len(payload) > settings.max_image_bytes:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Image is too large")
    try:
        decoded = validate_image_payload(payload, image.filename, image.content_type)
        mask_array = None
        if patch_mask is not None:
            mask_payload = await patch_mask.read()
            mask_array = cv2.cvtColor(validate_image_payload(mask_payload, patch_mask.filename, patch_mask.content_type), cv2.COLOR_RGB2GRAY)
        weights = json.loads(trust_weights) if trust_weights else None
        if weights is not None and (not isinstance(weights, dict) or any(float(value) < 0 for value in weights.values())):
            raise ValueError("trust_weights must be a JSON object with non-negative values")
        result = TrustPipeline().analyze(
            decoded,
            get_model("resnet18"),
            trust_weights={str(key): float(value) for key, value in weights.items()} if weights else None,
            localization_threshold=localization_threshold,
            minimum_region_area=minimum_region_area,
            patch_mask=mask_array,
        )
        trust_bytes = trust_map_png(result.trust_map)
        overlay_bytes = heatmap_overlay(decoded, result.trust_map, result.localization.regions)
    except (InputValidationError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    return TrustMapAnalyzeResponse(
        request_id=str(uuid4()),
        trust_map={
            "width": result.trust_map.width,
            "height": result.trust_map.height,
            "map_reference": _data_reference(trust_bytes),
            "overlay_reference": _data_reference(overlay_bytes),
            "resolution": result.trust_map.resolution,
            "normalization": result.trust_map.normalization,
            "contributing_detectors": list(result.trust_map.contributing_detectors),
        },
        global_trust_score=float(result.trust_map.values.mean()),
        suspicious_regions=result.localization.regions,
        suspicious_area_percentage=result.localization.suspicious_area_percentage,
        spatial_evidence=result.spatial_evidence,
        processing_time_ms=result.processing_time_ms,
    )