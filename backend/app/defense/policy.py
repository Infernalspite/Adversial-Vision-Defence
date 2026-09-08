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
    ) -> DefenseDecision:
        """Return the least destructive suitable strategy; thresholds are MVP calibration."""
        score = max(0.0, min(1.0, attack_score))
        localized = bool(suspicious_regions) and suspicious_area_percentage <= self.config.localized_area_percentage
        if score < self.config.no_defense_threshold:
            return DefenseDecision("none", "Attack evidence is below the defense threshold.", 0, 1 - score)
        if attack_type == "adversarial_patch" or (score >= self.config.transform_threshold and localized):
            return DefenseDecision(
                "mask", "High evidence is spatially localized; mask suspicious regions.", 2, score,
                {"mask_padding": 4},
            )
        if attack_type == "pgd" or score >= self.config.extreme_threshold:
            return DefenseDecision(
                "purification", "High or iterative-attack evidence warrants stronger MVP purification.", 3, score,
                {"strength": 1},
            )
        if attack_type == "fgsm" or score < self.config.transform_threshold:
            return DefenseDecision(
                "transform", "Moderate evidence warrants a lightweight image transform.", 1, score,
                {"method": "gaussian_blur", "kernel_size": 3},
            )
        return DefenseDecision(
            "purification", "Unknown attack evidence is handled conservatively with purification.", 3, score,
            {"strength": 1},
        )