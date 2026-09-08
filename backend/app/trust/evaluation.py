"""Trust-map localization evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.trust.localization import LocalizationResult
from app.trust.trust_map import TrustMap


@dataclass(frozen=True, slots=True)
class LocalizationMetrics:
    """Metrics for comparing suspicious pixels with an available ground-truth mask."""

    intersection_over_union: float
    precision: float
    recall: float
    suspicious_area_percentage: float


def evaluate_localization(trust_map: TrustMap, ground_truth_mask: np.ndarray, threshold: float = 0.30) -> LocalizationMetrics:
    """Calculate IoU, pixel precision/recall, and predicted suspicious area."""
    predicted = trust_map.values < threshold
    truth = np.asarray(ground_truth_mask) > 0
    if truth.shape != predicted.shape:
        raise ValueError("Ground-truth mask must match trust-map resolution")
    intersection = int(np.logical_and(predicted, truth).sum())
    union = int(np.logical_or(predicted, truth).sum())
    predicted_count = int(predicted.sum())
    truth_count = int(truth.sum())
    return LocalizationMetrics(
        intersection_over_union=intersection / union if union else 1.0,
        precision=intersection / predicted_count if predicted_count else 0.0,
        recall=intersection / truth_count if truth_count else 0.0,
        suspicious_area_percentage=float(predicted_count / predicted.size * 100),
    )
