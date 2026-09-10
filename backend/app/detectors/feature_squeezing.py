"""Feature squeezing detector."""

import time
from typing import Any

import cv2
import numpy as np

from app.detectors.base_detector import BaseDetector, DetectorResult, clip_score, detector_context


def _top_margin(prediction: Any) -> float:
    """Margin between the top-1 and top-2 class confidences.

    Falls back to the plain confidence when a model exposes no ranked
    predictions (stub models in tests, minimal adapters).
    """
    tops = getattr(prediction, "top_predictions", None)
    if tops and len(tops) >= 2:
        return float(tops[0].confidence - tops[1].confidence)
    if tops:
        return float(tops[0].confidence)
    confidence = getattr(prediction, "confidence", None)
    return float(confidence) if confidence is not None else 0.0


class FeatureSqueezingDetector(BaseDetector):
    """Compare baseline predictions before and after benign input squeezing.

    One gentle squeeze hides adversarial perturbations that are robust to
    quantization. Multiple squeezes of different strengths (aggressive
    bit-depth, median filtering) plus a top-1/top-2 margin signal expose
    inputs whose decision is unusually fragile under benign resampling.
    """

    name = "feature_squeezing"

    def _squeezed_variants(self, image: np.ndarray) -> list[tuple[str, np.ndarray]]:
        variants: list[tuple[str, np.ndarray]] = []
        for bits in (3, 5):
            levels = (2**bits) - 1
            squeezed = np.round(image.astype(np.float32) / 255 * levels) / levels * 255
            variants.append((f"bit_depth_{bits}", squeezed.astype(np.uint8)))
        variants.append(("median_3", cv2.medianBlur(image, 3)))
        return variants

    def detect(self, image: np.ndarray, context: dict[str, Any] | None = None) -> DetectorResult:
        """Measure prediction, confidence, and margin changes under squeezing."""
        started = time.perf_counter()
        model, threshold = detector_context(context)
        original_prediction = model.predict(image)
        original_margin = _top_margin(original_prediction)

        flips = 0
        confidence_deltas: list[float] = []
        margin_deltas: list[float] = []
        for _, squeezed in self._squeezed_variants(image):
            squeezed_prediction = model.predict(squeezed)
            if squeezed_prediction.class_id != original_prediction.class_id:
                flips += 1
            confidence_deltas.append(abs(float(original_prediction.confidence) - float(squeezed_prediction.confidence)))
            margin_deltas.append(abs(original_margin - _top_margin(squeezed_prediction)))

        flip_fraction = flips / 3.0
        mean_confidence_delta = float(np.mean(confidence_deltas))
        mean_margin_delta = float(np.mean(margin_deltas))
        # Weighted blend: a flip is the strongest single signal; margin
        # collapse (fragile decision boundary) next; raw confidence shifts
        # are the weakest because strong clean predictions also wobble.
        score = clip_score(
            0.40 * min(1.0, flip_fraction * 2.0)
            + 0.35 * min(1.0, mean_margin_delta * 4.0)
            + 0.25 * min(1.0, mean_confidence_delta * 4.0)
        )
        elapsed = (time.perf_counter() - started) * 1000
        return DetectorResult(
            detector_name=self.name,
            score=score,
            detected=score >= threshold,
            confidence=score,
            evidence={
                "prediction_flips": flips,
                "flip_fraction": flip_fraction,
                "mean_confidence_delta": mean_confidence_delta,
                "mean_margin_delta": mean_margin_delta,
                "confidence_deltas": confidence_deltas,
                "squeezing_method": "multi_variant",
            },
            processing_time_ms=elapsed,
            metadata={"variants": ["bit_depth_3", "bit_depth_5", "median_3"]},
        )
