"""Coarse scene-level consistency evidence."""

from typing import Any

import cv2
import numpy as np

from app.models.base_model import ModelPrediction


class SceneConsistencyVerifier:
    """Compare color-distribution and classification-level scene characteristics."""

    def verify(
        self,
        original: np.ndarray,
        defended: np.ndarray,
        original_prediction: ModelPrediction | None = None,
        defended_prediction: ModelPrediction | None = None,
    ) -> dict[str, Any]:
        """Return coarse histogram and optional class evidence."""
        original_hist = cv2.calcHist([original], [0, 1], None, [16, 16], [0, 256, 0, 256])
        defended_hist = cv2.calcHist([defended], [0, 1], None, [16, 16], [0, 256, 0, 256])
        cv2.normalize(original_hist, original_hist)
        cv2.normalize(defended_hist, defended_hist)
        correlation = float(cv2.compareHist(original_hist, defended_hist, cv2.HISTCMP_CORREL))
        histogram_similarity = max(0.0, min(1.0, (correlation + 1) / 2))
        class_similarity = 1.0
        if original_prediction is not None and defended_prediction is not None:
            class_similarity = float(original_prediction.class_id == defended_prediction.class_id)
        score = max(0.0, min(1.0, 0.7 * histogram_similarity + 0.3 * class_similarity))
        return {
            "scene_similarity_score": score,
            "feature_similarity": histogram_similarity,
            "class_similarity": class_similarity,
            "histogram_correlation": correlation,
        }
