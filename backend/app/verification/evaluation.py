"""Semantic verification evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class VerificationMetrics:
    """Aggregate consistency and recovery measurements."""

    sample_count: int
    average_verification_score: float
    average_object_consistency: float
    average_geometry_consistency: float
    average_scene_consistency: float
    prediction_recovery_rate: float
    average_confidence_recovery: float


def evaluate_verification(records: list[dict[str, Any]]) -> VerificationMetrics:
    """Aggregate verification results without treating recovery as proof of correctness."""
    if not records:
        return VerificationMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    recovered = []
    confidence_recovery = []
    for record in records:
        original = record["original_prediction"]
        defended = record["defended_prediction"]
        recovered.append(float(original["class_id"] == defended["class_id"]))
        confidence_recovery.append(float(defended["confidence"] - record.get("adversarial_confidence", original["confidence"])))
    return VerificationMetrics(
        sample_count=len(records),
        average_verification_score=float(np.mean([record["verification_score"] for record in records])),
        average_object_consistency=float(np.mean([record["object_consistency"] for record in records])),
        average_geometry_consistency=float(np.mean([record["geometry_consistency"] for record in records])),
        average_scene_consistency=float(np.mean([record["scene_consistency"] for record in records])),
        prediction_recovery_rate=float(np.mean(recovered)),
        average_confidence_recovery=float(np.mean(confidence_recovery)),
    )
