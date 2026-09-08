"""Engineering ranking score for attack evolution."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class DifficultyConfig:
    weights: dict[str, float] = field(default_factory=lambda: {"attack_success": 0.30, "confidence": 0.25, "detector_evasion": 0.20, "defense_failure": 0.15, "verification_failure": 0.10})


class DifficultyScorer:
    """Rank attacks; this is an engineering metric, not scientific validation."""

    def __init__(self, config: DifficultyConfig | None = None) -> None:
        self.config = config or DifficultyConfig()

    def score(self, result: dict[str, Any]) -> float:
        values = {
            "attack_success": float(result.get("attack_success", False)),
            "confidence": float(result.get("adversarial_confidence", 0.0)),
            "detector_evasion": 1 - float(result.get("attack_score", 0.0)),
            "defense_failure": float(not result.get("defense_recovered", False)),
            "verification_failure": 1 - float(result.get("verification_score", 0.0)),
        }
        total = sum(self.config.weights.values()) or 1.0
        return max(0.0, min(1.0, sum(self.config.weights.get(key, 0) * value for key, value in values.items()) / total))
