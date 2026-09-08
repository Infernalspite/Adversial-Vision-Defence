"""Attack Lab generation and comparison endpoints."""

import base64
from uuid import uuid4

import cv2
import numpy as np
from fastapi import APIRouter, Form, HTTPException, UploadFile, status

from app.api.schemas.attack import AttackEvaluationResponse, AttackRequest, AttackResponse
from app.core.config import settings
from app.attacks import get_attack, supported_attacks as registered_attacks
from app.models.model_registry import get_model
from app.security.input_validator import InputValidationError, validate_image_payload
from app.utils.image import encode_image

router = APIRouter(prefix="/attacks", tags=["attacks"])


@router.get("/supported", response_model=list[str])
def supported_attacks() -> list[str]:
    """List planned attack adapters."""
    return registered_attacks()


def _request_from_form(
    attack: str,
    epsilon: float,
    step_size: float,
    iterations: int,
    random_start: bool,
    patch_size: float,
    location: str,
    learning_rate: float,
) -> AttackRequest:
    """Validate multipart attack parameters through the shared Pydantic contract."""
    try:
        return AttackRequest(
            attack=attack,
            epsilon=epsilon,
            step_size=step_size,
            iterations=iterations,
            random_start=random_start,
            patch_size=patch_size,
            location=location,
            learning_rate=learning_rate,
        )
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error


async def _generate(
    image: UploadFile,
    request: AttackRequest,
) -> tuple[object, object]:
    """Decode an upload and run the selected registered attack."""
    payload = await image.read()
    if len(payload) > settings.max_image_bytes:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Image is too large")
    try:
        decoded = validate_image_payload(payload, image.filename, image.content_type)
        model = get_model("resnet18")
        attack = get_attack(request.attack)
        result = attack.generate(
            decoded,
            model,
            epsilon=request.epsilon,
            step_size=request.step_size,
            iterations=request.iterations,
            random_start=request.random_start,
            patch_size=request.patch_size,
            location=request.location,
            learning_rate=request.learning_rate,
        )
    except (InputValidationError, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    return request, result


def _image_reference(image: object) -> str:
    """Encode an image as a temporary in-response data reference."""
    encoded = encode_image(image)  # type: ignore[arg-type]
    return f"data:image/png;base64,{base64.b64encode(encoded).decode('ascii')}"


def _mask_reference(mask: np.ndarray | None) -> str | None:
    """Encode a binary patch mask as a temporary PNG data reference."""
    if mask is None:
        return None
    success, encoded = cv2.imencode(".png", mask * 255)
    if not success:
        raise HTTPException(status_code=500, detail="Unable to encode patch mask")
    return f"data:image/png;base64,{base64.b64encode(encoded.tobytes()).decode('ascii')}"


def _response(request_id: str, request: AttackRequest, result: object) -> AttackResponse:
    """Map a domain attack result to the transport response."""
    return AttackResponse(
        request_id=request_id,
        attack_type=result.attack_type,  # type: ignore[attr-defined]
        attack_parameters=request.model_dump(exclude={"attack"}),
        original_prediction={
            "class_id": result.original_prediction.class_id,  # type: ignore[attr-defined]
            "class_name": result.original_prediction.class_name,  # type: ignore[attr-defined]
            "confidence": result.original_prediction.confidence,  # type: ignore[attr-defined]
        },
        original_confidence=result.original_prediction.confidence,  # type: ignore[attr-defined]
        adversarial_prediction={
            "class_id": result.adversarial_prediction.class_id,  # type: ignore[attr-defined]
            "class_name": result.adversarial_prediction.class_name,  # type: ignore[attr-defined]
            "confidence": result.adversarial_prediction.confidence,  # type: ignore[attr-defined]
        },
        adversarial_confidence=result.adversarial_prediction.confidence,  # type: ignore[attr-defined]
        attack_success=result.attack_success,  # type: ignore[attr-defined]
        perturbation_magnitude=result.perturbation_magnitude,  # type: ignore[attr-defined]
        generation_time_ms=result.generation_time_ms,  # type: ignore[attr-defined]
        adversarial_image=_image_reference(result.adversarial_image),  # type: ignore[attr-defined]
        patch_mask=_mask_reference(result.patch_mask),  # type: ignore[attr-defined]
    )


@router.post("/generate", response_model=AttackResponse)
async def generate_attack(
    image: UploadFile,
    attack: str = Form(...),
    epsilon: float = Form(0.01),
    step_size: float = Form(0.005),
    iterations: int = Form(10),
    random_start: bool = Form(True),
    patch_size: float = Form(0.2),
    location: str = Form("center"),
    learning_rate: float = Form(0.05),
) -> AttackResponse:
    """Generate an adversarial image without persisting it."""
    request = _request_from_form(attack, epsilon, step_size, iterations, random_start, patch_size, location, learning_rate)
    _, result = await _generate(image, request)
    return _response(str(uuid4()), request, result)


@router.post("/evaluate", response_model=AttackEvaluationResponse)
async def evaluate_attack(
    image: UploadFile,
    attack: str = Form(...),
    epsilon: float = Form(0.01),
    step_size: float = Form(0.005),
    iterations: int = Form(10),
    random_start: bool = Form(True),
    patch_size: float = Form(0.2),
    location: str = Form("center"),
    learning_rate: float = Form(0.05),
) -> AttackEvaluationResponse:
    """Generate an attack and return direct clean-versus-adversarial comparison."""
    request = _request_from_form(attack, epsilon, step_size, iterations, random_start, patch_size, location, learning_rate)
    _, result = await _generate(image, request)
    response = _response(str(uuid4()), request, result)
    return AttackEvaluationResponse(
        **response.model_dump(),
        confidence_change=response.adversarial_confidence - response.original_confidence,
        prediction_changed=response.adversarial_prediction.class_id != response.original_prediction.class_id,
    )
