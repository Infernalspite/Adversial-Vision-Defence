"""Configurable unified attack scoring."""

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

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
DETECTOR_FEATURE_ORDER = (
    "feature_squeezing",
    "frequency_analysis",
    "confidence_instability",
    "saliency_analysis",
)


def valid_logistic_payload(payload: dict[str, Any] | None) -> bool:
    """Return whether a calibration payload has the complete expected schema."""
    if not isinstance(payload, dict):
        return False
    try:
        mean = np.asarray(payload["mean"], dtype=float)
        scale = np.asarray(payload["scale"], dtype=float)
        coefficients = np.asarray(payload["coefficients"], dtype=float)
        threshold = float(payload["threshold"])
        return (
            payload.get("schema") == "argus.logistic_regression"
            and payload.get("version") == 1
            and tuple(payload["feature_order"]) == DETECTOR_FEATURE_ORDER
            and mean.shape == (4,)
            and scale.shape == (4,)
            and coefficients.shape == (4,)
            and np.isfinite(mean).all()
            and np.isfinite(scale).all()
            and np.isfinite(coefficients).all()
            and np.all(scale > 0)
            and np.isfinite(float(payload["intercept"]))
            and 0 <= threshold <= 1
        )
    except (KeyError, TypeError, ValueError):
        return False


@dataclass(frozen=True, slots=True)
class AttackScorerConfig:
    """MVP calibration settings for transparent score fusion."""

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_DETECTOR_WEIGHTS))
    detection_threshold: float = 0.70
    logistic_regression: dict[str, Any] | None = None

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
        score = self._logistic_score(detector_results)
        if score is None:
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

    def _logistic_score(self, detector_results: dict[str, DetectorResult]) -> float | None:
        payload = self.config.logistic_regression
        if not valid_logistic_payload(payload) or any(name not in detector_results for name in DETECTOR_FEATURE_ORDER):
            return None
        try:
            mean = np.asarray(payload["mean"], dtype=float)
            scale = np.asarray(payload["scale"], dtype=float)
            coefficients = np.asarray(payload["coefficients"], dtype=float)
            intercept = float(payload["intercept"])
            features = np.asarray([detector_results[name].score for name in DETECTOR_FEATURE_ORDER], dtype=float)
            logit = float(np.dot((features - mean) / scale, coefficients) + intercept)
            probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -60.0, 60.0)))
            return float(probability) if np.isfinite(probability) else None
        except (KeyError, TypeError, ValueError):
            return None
