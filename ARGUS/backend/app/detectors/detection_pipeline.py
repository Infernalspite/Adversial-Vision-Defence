"""Phase 3 detector orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from app.detectors import get_detector, supported_detectors
from app.detectors.attack_scorer import AttackScore, AttackScorerConfig, UnifiedAttackScorer
from app.detectors.base_detector import DetectorResult


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Unified Phase 3 result consumed by later phases."""

    attack_score: float
    detection_threshold: float
    attack_detected: bool
    detectors: dict[str, DetectorResult]
    explanation: str
    processing_time_ms: float


class DetectionPipeline:
    """Run configured detectors and fuse their evidence."""

    def __init__(self, scorer: UnifiedAttackScorer | None = None) -> None:
        self.scorer = scorer or UnifiedAttackScorer()

    def analyze(
        self,
        image: np.ndarray,
        model: object,
        detector_names: list[str] | None = None,
        threshold: float | None = None,
    ) -> DetectionResult:
        """Validate inputs, run detectors, and return the unified score."""
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Expected an RGB uint8 image")
        names = detector_names or supported_detectors()
        if not names:
            raise ValueError("At least one detector is required")
        config = self.scorer.config
        if threshold is not None:
            config = AttackScorerConfig(config.weights, threshold)
        context = {"model": model, "detector_threshold": config.detection_threshold}
        started = time.perf_counter()
        results = [get_detector(name).detect(image, context) for name in names]
        fused: AttackScore = UnifiedAttackScorer(config).score(results)
        return DetectionResult(
            attack_score=fused.score,
            detection_threshold=fused.detection_threshold,
            attack_detected=fused.detected,
            detectors=fused.detector_results,
            explanation=fused.explanation,
            processing_time_ms=(time.perf_counter() - started) * 1000,
        )