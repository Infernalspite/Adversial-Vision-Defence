"""Phase 6 verification orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from app.models.base_model import ModelPrediction
from app.trust.trust_fusion import VerificationFusion, VerificationFusionConfig
from app.verification.semantic_anchor import SemanticRealityAnchor, VerificationResult


@dataclass(frozen=True, slots=True)
class VerificationPipelineResult:
    """Verification output with original/defended comparison fields."""

    original_prediction: ModelPrediction
    defended_prediction: ModelPrediction
    prediction_changed: bool
    confidence_change: float
    top_k_overlap: float
    verification: VerificationResult
    processing_time_ms: float


class VerificationPipeline:
    """Run semantic verification using existing model results where available."""

    def __init__(self, anchor: SemanticRealityAnchor | None = None, fusion: VerificationFusion | None = None) -> None:
        self.anchor = anchor or SemanticRealityAnchor()
        self.fusion = fusion or VerificationFusion()

    def run(
        self,
        original_image: np.ndarray,
        defended_image: np.ndarray,
        original_prediction: ModelPrediction,
        defended_prediction: ModelPrediction,
        trust_map: object | None = None,
        suspicious_regions: list[dict[str, object]] | None = None,
        detection_result: object | None = None,
    ) -> VerificationPipelineResult:
        """Verify without repeating inference already performed by Phase 5."""
        started = time.perf_counter()
        verification = self.anchor.verify(
            original_image,
            defended_image,
            original_prediction,
            defended_prediction,
            {"trust_map": trust_map, "suspicious_regions": suspicious_regions, "detection": detection_result},
        )
        return VerificationPipelineResult(
            original_prediction=original_prediction,
            defended_prediction=defended_prediction,
            prediction_changed=original_prediction.class_id != defended_prediction.class_id,
            confidence_change=defended_prediction.confidence - original_prediction.confidence,
            top_k_overlap=verification.object_consistency["top_k_overlap"],
            verification=verification,
            processing_time_ms=(time.perf_counter() - started) * 1000,
        )