"""Feature squeezing detector."""

import time
from typing import Any

import cv2
import numpy as np

from app.detectors.base_detector import BaseDetector, DetectorResult, clip_score, detector_context


class FeatureSqueezingDetector(BaseDetector):
    """Compare baseline predictions before and after benign input squeezing."""

    name = "feature_squeezing"

    def detect(self, image: np.ndarray, context: dict[str, Any] | None = None) -> DetectorResult:
        """Measure prediction and confidence changes after bit-depth reduction."""
        started = time.perf_counter()
        model, threshold = detector_context(context)
        bits = int((context or {}).get("bit_depth", 5))
        bits = max(2, min(8, bits))
        levels = (2**bits) - 1
        squeezed = np.round(image.astype(np.float32) / 255 * levels) / levels * 255
        squeezed = squeezed.astype(np.uint8)
        if bool((context or {}).get("smooth", False)):
            squeezed = cv2.GaussianBlur(squeezed, (3, 3), 0)
        original_prediction = model.predict(image)
        squeezed_prediction = model.predict(squeezed)
        confidence_difference = abs(original_prediction.confidence - squeezed_prediction.confidence)
        prediction_changed = original_prediction.class_id != squeezed_prediction.class_id
        score = clip_score((0.65 if prediction_changed else 0.0) + 0.35 * confidence_difference)
        elapsed = (time.perf_counter() - started) * 1000
        return DetectorResult(
            detector_name=self.name,
            score=score,
            detected=score >= threshold,
            confidence=score,
            evidence={
                "prediction_changed": prediction_changed,
                "confidence_difference": confidence_difference,
                "squeezing_method": "bit_depth",
            },
            processing_time_ms=elapsed,
            metadata={"bit_depth": bits, "smoothing": bool((context or {}).get("smooth", False))},
        )
