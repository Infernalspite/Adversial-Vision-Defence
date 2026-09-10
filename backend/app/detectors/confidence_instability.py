"""Confidence instability detector."""

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


class ConfidenceInstabilityDetector(BaseDetector):
    """Measure prediction instability under small benign transformations.

    The original probes (noise sigma 1.5, brightness 1.02, blur) were too
    gentle to perturb a confident prediction even on adversarial inputs. The
    probes here remain benign for clean images but are strong enough to
    wobble a decision that sits on an adversarially manufactured boundary:
    heavier noise, larger brightness/contrast swings, downscale-upscale
    resampling, and a top-1/top-2 margin signal.
    """

    name = "confidence_instability"

    def _variants(self, image: np.ndarray) -> list[tuple[str, np.ndarray]]:
        rng = np.random.default_rng(7)
        noise = np.clip(image.astype(np.int16) + rng.normal(0, 4.0, image.shape), 0, 255).astype(np.uint8)
        brighter = np.clip(image.astype(np.float32) * 1.06 + 4, 0, 255).astype(np.uint8)
        darker = np.clip(image.astype(np.float32) * 0.94 - 4, 0, 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        height, width = image.shape[:2]
        resampled = cv2.resize(
            cv2.resize(image, (max(1, width // 2), max(1, height // 2)), interpolation=cv2.INTER_AREA),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        return [("noise", noise), ("brighter", brighter), ("darker", darker), ("blur", blurred), ("resample", resampled)]

    def detect(self, image: np.ndarray, context: dict[str, Any] | None = None) -> DetectorResult:
        """Run deterministic probe transformations and aggregate instability."""
        started = time.perf_counter()
        model, threshold = detector_context(context)
        variants = self._variants(image)
        original = model.predict(image)
        original_margin = _top_margin(original)
        predictions = [model.predict(variant) for _, variant in variants]

        flips = sum(prediction.class_id != original.class_id for prediction in predictions)
        stability = (len(predictions) - flips) / len(predictions)
        confidence_deltas = [abs(float(prediction.confidence) - float(original.confidence)) for prediction in predictions]
        margin_deltas = [abs(_top_margin(prediction) - original_margin) for prediction in predictions]
        mean_confidence_delta = float(np.mean(confidence_deltas))
        mean_margin_delta = float(np.mean(margin_deltas))
        score = clip_score(
            0.40 * (1 - stability)
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
                "prediction_flips": int(flips),
                "prediction_stability": stability,
                "mean_confidence_delta": mean_confidence_delta,
                "mean_margin_delta": mean_margin_delta,
                "confidence_deltas": confidence_deltas,
                "transform_count": len(variants),
            },
            processing_time_ms=elapsed,
            metadata={"transforms": [name for name, _ in variants]},
        )
