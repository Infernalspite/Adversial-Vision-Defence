"""Attack Lab API contracts."""

from typing import Literal

from pydantic import BaseModel, Field

AttackName = Literal["fgsm", "pgd", "adversarial_patch"]


class AttackRequest(BaseModel):
    """Validated parameters accepted by the Attack Lab."""

    attack: AttackName
    epsilon: float = Field(default=0.01, gt=0, le=0.25)
    step_size: float = Field(default=0.005, gt=0, le=0.25)
    iterations: int = Field(default=10, ge=1, le=100)
    random_start: bool = True
    patch_size: float = Field(default=0.2, gt=0, le=1)
    location: Literal["center", "top_left", "top_right", "bottom_left", "bottom_right"] = "center"
    learning_rate: float = Field(default=0.05, gt=0, le=1)


class AttackTopPrediction(BaseModel):
    """Prediction summary used in an attack response."""

    class_id: int
    class_name: str
    confidence: float = Field(ge=0, le=1)


class AttackResponse(BaseModel):
    """Generated adversarial image and comparison metadata."""

    request_id: str
    attack_type: str
    attack_parameters: dict[str, object]
    original_prediction: AttackTopPrediction
    original_confidence: float = Field(ge=0, le=1)
    adversarial_prediction: AttackTopPrediction
    adversarial_confidence: float = Field(ge=0, le=1)
    attack_success: bool
    perturbation_magnitude: float = Field(ge=0)
    generation_time_ms: float = Field(ge=0)
    adversarial_image: str
    patch_mask: str | None = None


class AttackEvaluationResponse(AttackResponse):
    """Attack response with direct comparison fields."""

    confidence_change: float
    prediction_changed: bool


class AttackEvidence(BaseModel):
    """One detector signal exposed by the API."""

    detector_name: str
    score: float | None = Field(default=None, ge=0, le=1)
    evidence: dict[str, object] = Field(default_factory=dict)


class AttackSummary(BaseModel):
    """Unified adversarial triage result."""

    detected: bool = False
    score: float = Field(default=0, ge=0, le=1)
    attack_type: str | None = None
    localization: list[dict[str, object]] = Field(default_factory=list)
    evidence: list[AttackEvidence] = Field(default_factory=list)
