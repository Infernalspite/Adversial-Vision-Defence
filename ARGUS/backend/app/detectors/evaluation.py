"""Detector evaluation utilities kept separate from detector implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from app.detectors.detection_pipeline import DetectionPipeline


@dataclass(frozen=True, slots=True)
class DetectionEvaluation:
    """Aggregate clean-versus-adversarial detector metrics."""

    clean_samples: int
    adversarial_samples: int
    detection_rate: float
    false_positive_rate: float
    true_positive_rate: float
    false_negative_rate: float
    average_clean_score: float
    average_adversarial_score: float


def evaluate_detector(
    samples: Iterable[tuple[np.ndarray, bool]],
    model: object,
    pipeline: DetectionPipeline | None = None,
) -> DetectionEvaluation:
    """Evaluate labelled clean/adversarial images without changing calibration."""
    runner = pipeline or DetectionPipeline()
    clean_scores: list[float] = []
    adversarial_scores: list[float] = []
    true_positive = false_negative = false_positive = 0
    for image, is_adversarial in samples:
        result = runner.analyze(image, model)
        if is_adversarial:
            adversarial_scores.append(result.attack_score)
            if result.attack_detected:
                true_positive += 1
            else:
                false_negative += 1
        else:
            clean_scores.append(result.attack_score)
            if result.attack_detected:
                false_positive += 1
    clean_count = len(clean_scores)
    adversarial_count = len(adversarial_scores)
    return DetectionEvaluation(
        clean_samples=clean_count,
        adversarial_samples=adversarial_count,
        detection_rate=true_positive / adversarial_count if adversarial_count else 0.0,
        false_positive_rate=false_positive / clean_count if clean_count else 0.0,
        true_positive_rate=true_positive / adversarial_count if adversarial_count else 0.0,
        false_negative_rate=false_negative / adversarial_count if adversarial_count else 0.0,
        average_clean_score=float(np.mean(clean_scores)) if clean_scores else 0.0,
        average_adversarial_score=float(np.mean(adversarial_scores)) if adversarial_scores else 0.0,
    )
