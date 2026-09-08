"""Typed records for offline self-healing experiments."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class HardNegative:
    request_id: str
    category: str
    explanation: str
    timestamp: str | None = None
    attack_type: str | None = None
    attack_score: float | None = None
    trust_score: float | None = None
    verification_score: float | None = None
    final_state: str | None = None
    original_prediction: str | None = None
    defended_prediction: str | None = None
    expected_behavior: str = "REVIEW"
    image_reference: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FailureAnalysis:
    category: str
    explanation: str
    severity: float


@dataclass(frozen=True, slots=True)
class AttackConfig:
    attack_type: str
    epsilon: float = 0.01
    step_size: float = 0.005
    iterations: int = 10
    random_start: bool = False
    patch_size: float = 0.2
    patch_location: str = "center"
    transformation_strength: float = 0.0
    target_class: int | None = None
    seed: int = 7

    def validate(self) -> None:
        if self.attack_type not in {"fgsm", "pgd", "patch"}:
            raise ValueError(f"Unsupported evolution attack: {self.attack_type}")
        if not 0.001 <= self.epsilon <= 0.25:
            raise ValueError("epsilon must be between 0.001 and 0.25")
        if not 0.0005 <= self.step_size <= 0.25:
            raise ValueError("step_size must be between 0.0005 and 0.25")
        if not 1 <= self.iterations <= 100:
            raise ValueError("iterations must be between 1 and 100")
        if not 0.01 <= self.patch_size <= 1:
            raise ValueError("patch_size must be between 0.01 and 1")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
