"""Confidence instability detector."""

import time
from typing import Any

import cv2
import numpy as np

from app.detectors.base_detector import BaseDetector, DetectorResult, clip_score, detector_context


class ConfidenceInstabilityDetector(BaseDetector):
    """Measure prediction instability under small benign transformations."""

    name = "confidence_instability"

    def detect(self, image: np.ndarray, context: dict[str, Any] | None = None) -> DetectorResult:
        """Run deterministic noise, brightness, and blur probes."""
        started = time.perf_counter()
        model, threshold = detector_context(context)
        rng = np.random.default_rng(7)
        noise = np.clip(image.astype(np.int16) + rng.normal(0, 1.5, image.shape), 0, 255).astype(np.uint8)
        brighter = np.clip(image.astype(np.float32) * 1.02, 0, 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        variants = [noise, brighter, blurred]
        original = model.predict(image)
        predictions = [model.predict(variant) for variant in variants]
        stability = sum(prediction.class_id == original.class_id for prediction in predictions) / len(predictions)
        confidence_variance = float(np.var([prediction.confidence for prediction in predictions]))
        score = clip_score(0.55 * (1 - stability) + 0.45 * min(1.0, confidence_variance * 25))
        elapsed = (time.perf_counter() - started) * 1000
        return DetectorResult(
            detector_name=self.name,
            score=score,
            detected=score >= threshold,
            confidence=score,
            evidence={"prediction_stability": stability, "confidence_variance": confidence_variance, "transform_count": len(variants)},
            processing_time_ms=elapsed,
            metadata={"transforms": ["gaussian_noise", "brightness", "blur"]},
        )
