"""Classification-based object consistency evidence."""

from typing import Any

from app.models.base_model import ModelPrediction


class ObjectConsistencyVerifier:
    """Compare model interpretations without treating disagreement as proof of attack."""

    def verify(self, original: ModelPrediction, defended: ModelPrediction) -> dict[str, Any]:
        """Return prediction identity, top-k overlap, and confidence evidence."""
        original_top = {item.class_id for item in original.top_predictions}
        defended_top = {item.class_id for item in defended.top_predictions}
        union = original_top | defended_top
        overlap = len(original_top & defended_top) / len(union) if union else 1.0
        prediction_same = original.class_id == defended.class_id
        confidence_difference = abs(original.confidence - defended.confidence)
        score = max(0.0, min(1.0, 0.55 * float(prediction_same) + 0.30 * overlap + 0.15 * (1 - confidence_difference)))
        return {
            "prediction_before": original.class_name,
            "prediction_after": defended.class_name,
            "prediction_changed": not prediction_same,
            "top_k_overlap": overlap,
            "confidence_before": original.confidence,
            "confidence_after": defended.confidence,
            "object_consistency_score": score,
        }
