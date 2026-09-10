"""Phase 5 defense and re-inference pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import cv2
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
    defended_stability: float = 0.0


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
        # The detector's calibrated operating point is the single source of
        # truth for "is this flagged" — the defense policy must not apply a
        # different hardcoded cutoff, or flagged images fall into ABSTAIN
        # purgatory (flagged but never defended).
        defense_threshold = detection.detection_threshold
        # The detector score alone cannot name the attack family, but the
        # spatial signature can: a small suspicious area suggests a localized
        # (patch) attack, diffuse suspicion suggests a global perturbation.
        suspicious_area = float(trust.localization.suspicious_area_percentage)
        localized_signature = bool(trust.localization.regions) and suspicious_area <= 15.0
        # Token matches DefensePolicy's adversarial_patch branch.
        attack_signature = "adversarial_patch" if localized_signature else None
        orchestration = self.orchestrator.execute(
            image,
            {
                "attack_score": detection.attack_score,
                "attack_type": attack_signature,
                "detector_evidence": detection.detectors,
                "global_trust_score": float(trust.trust_map.values.mean()),
                "suspicious_regions": trust.localization.regions,
                "suspicious_area_percentage": trust.localization.suspicious_area_percentage,
                "defense_threshold": defense_threshold,
                "trust_map": trust.trust_map,
            },
        )
        # When no defense is applied, the "defended image" is the original —
        # re-inferring on it and letting verification compare a pixel to
        # itself produces vacuous 1.0 consistency scores. Skip re-inference
        # and pass through the original prediction so downstream evidence
        # stays honest.
        if orchestration.defense.defense_applied:
            defended_prediction = model.predict(orchestration.defense.defended_image)
            # Stability probe: the defense only *recovered* the input if its
            # output survives the same benign perturbations an adversary's
            # manufactured boundary would not. If the defended image still
            # flips labels under a mild squeeze, the decision remains
            # adversarially fragile and downstream must not certify it.
            probes = 0
            flips = 0
            levels = (2**3) - 1
            for variant in (
                np.round(orchestration.defense.defended_image.astype(np.float32) / 255 * levels) / levels * 255,
                cv2.GaussianBlur(orchestration.defense.defended_image, (5, 5), 0),
            ):
                probe_prediction = model.predict(variant.astype(np.uint8))
                probes += 1
                if probe_prediction.class_id != defended_prediction.class_id:
                    flips += 1
            defended_stability = (probes - flips) / probes if probes else 0.0
        else:
            defended_prediction = original_prediction
            defended_stability = 0.0
        return DefensePipelineResult(
            original_prediction=original_prediction,
            defended_prediction=defended_prediction,
            detection=detection,
            trust=trust,
            orchestration=orchestration,
            prediction_changed=original_prediction.class_id != defended_prediction.class_id,
            processing_time_ms=(time.perf_counter() - started) * 1000,
            defended_stability=defended_stability,
        )
