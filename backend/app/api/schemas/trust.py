"""Phase 4 trust-map API contracts."""

from pydantic import BaseModel, Field


class SuspiciousRegionResponse(BaseModel):
    """One connected low-trust region."""

    region_id: int
    x: int
    y: int
    width: int
    height: int
    area_pixels: int
    area_percentage: float = Field(ge=0)
    anomaly_score: float = Field(ge=0, le=1)
    trust_score: float = Field(ge=0, le=1)


class TrustMapResponse(BaseModel):
    """Compact spatial trust-map response for future consumers."""

    width: int
    height: int
    map_reference: str
    overlay_reference: str
    resolution: tuple[int, int]
    normalization: dict[str, object]
    contributing_detectors: list[str]


class SpatialEvidenceResponse(BaseModel):
    """Availability of spatial evidence sources."""

    saliency_available: bool
    frequency_available: bool
    local_anomaly_available: bool
    patch_mask_available: bool


class TrustMapAnalyzeResponse(BaseModel):
    """Complete Phase 4 result without a final trust state."""

    request_id: str
    trust_map: TrustMapResponse
    global_trust_score: float = Field(ge=0, le=1)
    suspicious_regions: list[SuspiciousRegionResponse]
    suspicious_area_percentage: float = Field(ge=0)
    spatial_evidence: SpatialEvidenceResponse
    processing_time_ms: float = Field(ge=0)
