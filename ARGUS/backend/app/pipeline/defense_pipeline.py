"""Phase 5 defense and re-inference pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.defense.orchestrator import DefenseOrchestrator, OrchestrationResult
from app.detectors.detection_pipeline import DetectionPipeline
from app.models.base_model import ModelPrediction
from app.trust.trust_pipeline import TrustPipeline, TrustPipelineResult


@dataclass(frozen=True, slots=True)
class DefensePipelineResult:
    """Detection, trust, defense, and defended inference outputs."""

    original_prediction: ModelPrediction
    defended_prediction: ModelPrediction
    detection: Any
    trust: TrustPipelineResult
    orchestration: OrchestrationResult
    prediction_changed: bool
    processing_time_ms: float


class ArgusDefensePipeline:
    """Compose Phase 3 detection, Phase 4 trust, and Phase 5 defense."""

    def __init__(
        self,
        detection_pipeline: DetectionPipeline | None = None,
        trust_pipeline: TrustPipeline | None = None,
        orchestrator: DefenseOrchestrator | None = None,
    ) -> None:
        self.detection_pipeline = detection_pipeline or DetectionPipeline()
        self.trust_pipeline = trust_pipeline or TrustPipeline(self.detection_pipeline)
        self.orchestrator = orchestrator or DefenseOrchestrator()

    def run(self, image: np.ndarray, model: object, patch_mask: np.ndarray | None = None) -> DefensePipelineResult:
        """Execute defense selection and re-inference without final trust-state logic."""
        started = time.perf_counter()
        original_prediction = model.predict(image)
        detection = self.detection_pipeline.analyze(image, model)
        trust = self.trust_pipeline.analyze(image, model, detection=detection, patch_mask=patch_mask)
        orchestration = self.orchestrator.execute(
            image,
            {
                "attack_score": detection.attack_score,
                "attack_type": None,
                "detector_evidence": detection.detectors,
                "global_trust_score": float(trust.trust_map.values.mean()),
                "suspicious_regions": trust.localization.regions,
                "suspicious_area_percentage": trust.localization.suspicious_area_percentage,
                "trust_map": trust.trust_map,
            },
        )
        defended_prediction = model.predict(orchestration.defense.defended_image)
        return DefensePipelineResult(
            original_prediction=original_prediction,
            defended_prediction=defended_prediction,
            detection=detection,
            trust=trust,
            orchestration=orchestration,
            prediction_changed=original_prediction.class_id != defended_prediction.class_id,
            processing_time_ms=(time.perf_counter() - started) * 1000,
        )
