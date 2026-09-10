"""Two-tier model routing for the analysis pipeline.

Tier 1 (frozen ImageNet-1000 baseline) recognizes any uploaded image.
Tier 2 (robust specialist) deepens defense for the 10 defended classes.

Routing rule: if any of the baseline's top predictions matches a defended
class, the robust specialist handles analysis and re-inference; otherwise
the baseline serves the label and re-infers purified images. Either way the
caller gets full detection, localization, and purification for the request.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.learning.classes import CLASS_NAMES
from app.models.base_model import BaseVisionModel, ModelPrediction

DEFENDED_CLASS_NAMES = frozenset(CLASS_NAMES)


def _matches_defended(class_name: str) -> str | None:
    """Tolerant name match: ImageNet categories carry scientific suffixes
    ("tench, Tinca tinca", "English springer, English springer spaniel")."""
    normalized = class_name.strip().lower().split(",")[0].strip()
    for defended in DEFENDED_CLASS_NAMES:
        defended_lower = defended.lower()
        if defended_lower in normalized or normalized in defended_lower:
            return defended
    return None


@dataclass(frozen=True, slots=True)
class TierRouting:
    """Which tier handles this request and why."""

    tier: str  # "baseline" | "robust"
    analysis_model: BaseVisionModel
    reclassify_model: BaseVisionModel
    reason: str
    matched_defended_class: str | None = None


def route_tier(
    baseline_prediction: ModelPrediction,
    baseline_model: BaseVisionModel,
    robust_model: BaseVisionModel | None,
) -> TierRouting:
    """Pick the serving tier from the baseline's own top predictions.

    Falls back to the baseline when no specialist checkpoint is available
    so the pipeline degrades gracefully instead of erroring on arbitrary
    uploads.
    """
    if robust_model is None:
        return TierRouting("baseline", baseline_model, baseline_model, "robust specialist unavailable")
    match = next(
        (matched for item in baseline_prediction.top_predictions if (matched := _matches_defended(item.class_name))),
        None,
    )
    if match is not None:
        return TierRouting("robust", robust_model, robust_model, f"defended class in top predictions: {match}", match)
    return TierRouting("baseline", baseline_model, baseline_model, "no defended class among baseline top predictions")
