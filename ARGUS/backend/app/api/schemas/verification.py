"""Phase 6 semantic verification API contracts."""

from pydantic import BaseModel, Field


class VerificationPrediction(BaseModel):
    """Prediction summary for before/after comparison."""

    class_id: int
    class_name: str
    confidence: float = Field(ge=0, le=1)


class ConsistencyResult(BaseModel):
    """Generic consistency score and evidence payload."""

    score: float = Field(ge=0, le=1)
    evidence: dict[str, object] = Field(default_factory=dict)


class VerificationResponse(BaseModel):
    """Semantic Reality Anchor result without a final decision state."""

    request_id: str
    original_prediction: VerificationPrediction
    original_confidence: float = Field(ge=0, le=1)
    defended_prediction: VerificationPrediction
    defended_confidence: float = Field(ge=0, le=1)
    prediction_changed: bool
    confidence_change: float
    top_k_overlap: float = Field(ge=0, le=1)
    object_consistency: ConsistencyResult
    geometry_consistency: ConsistencyResult
    scene_consistency: ConsistencyResult
    verification_score: float = Field(ge=0, le=1)
    explanation: str
    processing_time_ms: float = Field(ge=0)
