"""Phase 5 defense API contracts."""

from pydantic import BaseModel, Field

from app.api.schemas.trust import SuspiciousRegionResponse


class DefenseRequest(BaseModel):
    """Optional explicit defense context for future non-upload callers."""

    attack_score: float = Field(default=0, ge=0, le=1)
    attack_type: str | None = None
    confidence: float = Field(default=0, ge=0, le=1)
    evidence: dict[str, object] = Field(default_factory=dict)


class DefenseSummary(BaseModel):
    """Small defense action summary retained for existing API consumers."""

    applied: bool = False
    method: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class PredictionSummary(BaseModel):
    """Model prediction used by the defense response."""

    class_id: int
    class_name: str
    confidence: float = Field(ge=0, le=1)


class DefenseTraceResponse(BaseModel):
    """Explainable policy and execution trace."""

    timestamp: str
    attack_score: float = Field(ge=0, le=1)
    selected_defense: str
    reason: str
    input_trust_score: float = Field(ge=0, le=1)
    suspicious_regions: int = Field(ge=0)
    defense_parameters: dict[str, object] = Field(default_factory=dict)
    processing_time_ms: float = Field(ge=0)
    success: bool
    output_image_reference: str | None = None


class DefenseApplyResponse(BaseModel):
    """Complete Phase 5 response without final trust-state logic."""

    request_id: str
    original_prediction: PredictionSummary
    original_confidence: float = Field(ge=0, le=1)
    attack_score: float = Field(ge=0, le=1)
    attack_detected: bool
    global_trust_score: float = Field(ge=0, le=1)
    suspicious_regions: list[SuspiciousRegionResponse]
    selected_defense: str
    defense_reason: str
    defense_parameters: dict[str, object] = Field(default_factory=dict)
    defense_applied: bool
    defended_prediction: PredictionSummary
    defended_confidence: float = Field(ge=0, le=1)
    prediction_changed: bool
    defended_image_reference: str
    defense_trace: DefenseTraceResponse
    processing_time_ms: float = Field(ge=0)
