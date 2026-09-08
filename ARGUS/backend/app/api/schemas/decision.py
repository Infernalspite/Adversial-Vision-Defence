"""Phase 7 final analysis response contracts."""

from pydantic import BaseModel, Field

from app.api.schemas.defense import DefenseTraceResponse, PredictionSummary
from app.api.schemas.detection import DetectorResultResponse
from app.api.schemas.trust import SuspiciousRegionResponse
from app.api.schemas.verification import ConsistencyResult
from app.pipeline.states import FinalState


class DecisionResponse(BaseModel):
    """Final conservative decision with supporting evidence."""

    request_id: str
    final_state: FinalState
    final_prediction: str | None = None
    final_confidence: float | None = Field(default=None, ge=0, le=1)
    decision_score: float = Field(ge=0, le=1)
    attack_score: float = Field(ge=0, le=1)
    global_trust_score: float = Field(ge=0, le=1)
    trust_map_reference: str
    trust_map_overlay_reference: str
    verification_score: float = Field(ge=0, le=1)
    defense_applied: bool
    defense_method: str
    original_prediction: PredictionSummary
    original_confidence: float = Field(ge=0, le=1)
    defended_prediction: PredictionSummary
    defended_confidence: float = Field(ge=0, le=1)
    decision_reasons: list[str]
    explanation: str
    evidence_summary: dict[str, object] = Field(default_factory=dict)
    fallback_action: str | None = None
    fallback_reason: str | None = None
    processing_time_ms: float = Field(ge=0)


class AnalyzeResponse(BaseModel):
    """Complete Phase 3-7 response for the main API."""

    request_id: str
    attack_score: float = Field(ge=0, le=1)
    attack_detected: bool
    attack_detection_threshold: float = Field(ge=0, le=1)
    detectors: dict[str, DetectorResultResponse]
    global_trust_score: float = Field(ge=0, le=1)
    trust_map_reference: str
    trust_map_overlay_reference: str
    suspicious_regions: list[SuspiciousRegionResponse]
    defense_applied: bool
    defense_method: str
    defended_image_reference: str
    original_prediction: PredictionSummary
    defended_prediction: PredictionSummary
    original_confidence: float = Field(ge=0, le=1)
    defended_confidence: float = Field(ge=0, le=1)
    prediction_changed: bool
    object_consistency: ConsistencyResult
    geometry_consistency: ConsistencyResult
    scene_consistency: ConsistencyResult
    verification_score: float = Field(ge=0, le=1)
    final_state: FinalState
    final_prediction: str | None = None
    final_confidence: float | None = Field(default=None, ge=0, le=1)
    decision_score: float = Field(ge=0, le=1)
    decision_reasons: list[str]
    explanation: str
    fallback_action: str | None = None
    fallback_reason: str | None = None
    defense_trace: DefenseTraceResponse
    audit: list[dict[str, object]] = Field(default_factory=list)
    processing_time_ms: float = Field(ge=0)
