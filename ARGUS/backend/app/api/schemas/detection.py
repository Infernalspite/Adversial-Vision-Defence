"""Stable Phase 3 detection API contracts."""

from pydantic import BaseModel, Field


class DetectorResultResponse(BaseModel):
    """Evidence emitted by one detector."""

    detector_name: str
    score: float = Field(ge=0, le=1)
    detected: bool
    confidence: float = Field(ge=0, le=1)
    evidence: dict[str, object] = Field(default_factory=dict)
    processing_time_ms: float = Field(ge=0)
    metadata: dict[str, object] = Field(default_factory=dict)


class DetectionResponse(BaseModel):
    """Unified Phase 3 result consumed by future phases."""

    request_id: str
    attack_detected: bool
    attack_score: float = Field(ge=0, le=1)
    detection_threshold: float = Field(ge=0, le=1)
    detectors: dict[str, DetectorResultResponse]
    explanation: str
    processing_time_ms: float = Field(ge=0)