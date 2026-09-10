"""Transparent MVP defense selection policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class DefensePolicyConfig:
    """Calibratable thresholds for least-destructive defense selection."""

    no_defense_threshold: float = 0.30
    transform_threshold: float = 0.60
    extreme_threshold: float = 0.85
    localized_area_percentage: float = 15.0

    def __post_init__(self) -> None:
        values = (self.no_defense_threshold, self.transform_threshold, self.extreme_threshold)
        if any(not 0 <= value <= 1 for value in values):
            raise ValueError("policy score thresholds must be between 0 and 1")
        if not self.no_defense_threshold <= self.transform_threshold <= self.extreme_threshold:
            raise ValueError("policy thresholds must be ordered")
        if self.localized_area_percentage <= 0:
            raise ValueError("localized area percentage must be positive")


@dataclass(frozen=True, slots=True)
class DefenseDecision:
    """Policy output consumed by the orchestrator."""

    selected_defense: str
    reason: str
    priority: int
    confidence: float
    parameters: dict[str, Any] = field(default_factory=dict)


class DefensePolicy:
    """Choose one defense from global and spatial evidence."""

    def __init__(self, config: DefensePolicyConfig | None = None) -> None:
        self.config = config or DefensePolicyConfig()

    def decide(
        self,
        attack_score: float,
        attack_type: str | None,
        detector_evidence: Any,
        global_trust_score: float,
        suspicious_regions: list[Any],
        suspicious_area_percentage: float = 0.0,
        detection_threshold: float | None = None,
    ) -> DefenseDecision:
        """Return the least destructive suitable strategy; thresholds are MVP calibration.

        ``detection_threshold`` is the calibrated detector operating point
        (from ``DetectionPipeline``). Passing it keeps the defense policy and
        the detector threshold consistent: anything the detector flags is
        eligible for defense, instead of being silently dropped into ABSTAIN
        by a stale hardcoded cutoff.
        """
        score = max(0.0, min(1.0, attack_score))
        localized = bool(suspicious_regions) and suspicious_area_percentage <= self.config.localized_area_percentage
        # Two operating modes: with a calibrated detection threshold the bands
        # are detector-relative (the fused score scale is small, so absolute
        # 0.60/0.85 cutoffs were unreachable); without one, fall back to the
        # legacy absolute config thresholds.
        if detection_threshold is None:
            no_defense_at = self.config.no_defense_threshold
            borderline_at = self.config.transform_threshold
            strong_at = self.config.extreme_threshold
        else:
            no_defense_at = max(0.0, min(1.0, float(detection_threshold)))
            borderline_at = min(1.0, no_defense_at + 0.10)
            strong_at = min(1.0, no_defense_at + 0.15)
        if score < no_defense_at:
            return DefenseDecision("none", "Attack evidence is below the defense threshold.", 0, 1 - score)
        # Strong evidence first: purification handles global perturbations,
        # which masking cannot (an L-inf perturbation lives in every pixel).
        if attack_type == "pgd" or score >= strong_at:
            return DefenseDecision(
                "purification", "Strong or iterative-attack evidence warrants purification.", 3, score,
                {"strength": 1},
            )
        if attack_type == "adversarial_patch" or localized:
            return DefenseDecision(
                "mask", "Suspicious evidence is spatially localized; mask those regions.", 2, score,
                {"mask_padding": 4},
            )
        if attack_type == "fgsm" or score < borderline_at:
            return DefenseDecision(
                "transform", "Borderline evidence warrants a lightweight image transform.", 1, score,
                {"method": "gaussian_blur", "kernel_size": 3},
            )
        return DefenseDecision(
            "purification", "Unclassified but flagged evidence is handled conservatively with purification.", 3, score,
            {"strength": 1},
        )