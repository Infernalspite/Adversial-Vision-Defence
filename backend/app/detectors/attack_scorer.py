"""Configurable unified attack scoring."""

from dataclasses import dataclass, field
from typing import Iterable

from app.detectors.base_detector import DetectorResult


@dataclass(frozen=True, slots=True)
class AttackScore:
    """Fused evidence and the Phase 3 detection decision."""

    score: float
    detected: bool
    detection_threshold: float
    detector_results: dict[str, DetectorResult]
    explanation: str


DEFAULT_DETECTOR_WEIGHTS = {
    "feature_squeezing": 0.30,
    "frequency_analysis": 0.25,
    "confidence_instability": 0.25,
    "saliency_analysis": 0.20,
}


@dataclass(frozen=True, slots=True)
class AttackScorerConfig:
    """MVP calibration settings for transparent score fusion."""

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_DETECTOR_WEIGHTS))
    detection_threshold: float = 0.70

    def __post_init__(self) -> None:
        if not 0 <= self.detection_threshold <= 1:
            raise ValueError("detection_threshold must be between 0 and 1")
        if not self.weights or any(weight < 0 for weight in self.weights.values()):
            raise ValueError("detector weights must be non-empty and non-negative")
        if sum(self.weights.values()) <= 0:
            raise ValueError("detector weights must have a positive sum")


class UnifiedAttackScorer:
    """Combine independent detector evidence into one risk result."""

    def __init__(self, config: AttackScorerConfig | None = None) -> None:
        self.config = config or AttackScorerConfig()

    def score(self, results: Iterable[DetectorResult]) -> AttackScore:
        """Fuse configured detector evidence into one bounded score."""
        detector_results = {result.detector_name: result for result in results}
        total_weight = sum(self.config.weights.get(name, 0.0) for name in detector_results)
        if total_weight <= 0:
            raise ValueError("No detector results have configured weights")
        weighted_score = sum(
            self.config.weights.get(name, 0.0) * result.score
            for name, result in detector_results.items()
        ) / total_weight
        score = max(0.0, min(1.0, weighted_score))
        detected = score >= self.config.detection_threshold
        contributors = sorted(detector_results.values(), key=lambda result: result.score, reverse=True)
        explanation = (
            "Multiple detectors indicate anomalous visual characteristics."
            if detected
            else "Detector evidence remains below the configured MVP threshold."
        )
        if contributors:
            explanation += f" Highest evidence: {contributors[0].detector_name} ({contributors[0].score:.2f})."
        return AttackScore(score, detected, self.config.detection_threshold, detector_results, explanation)
