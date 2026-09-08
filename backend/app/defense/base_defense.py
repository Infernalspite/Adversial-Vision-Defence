"""Common defense interface and result contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class DefenseResult:
    """Result of one defense strategy without a final trust decision."""

    defense_name: str
    defense_applied: bool
    defended_image: np.ndarray
    parameters: dict[str, Any]
    affected_regions: list[dict[str, Any]]
    change_magnitude: float
    processing_time_ms: float
    explanation: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def image(self) -> np.ndarray:
        """Compatibility alias used by older defense callers."""
        return self.defended_image


class BaseDefense(ABC):
    """Contract for an autonomous image defense strategy."""

    @abstractmethod
    def defend(self, image: np.ndarray, context: dict[str, Any] | None = None) -> DefenseResult:
        """Transform an image using the provided detection/trust context."""
        raise NotImplementedError


class NoDefense(BaseDefense):
    """No-op strategy used when evidence does not warrant intervention."""

    def defend(self, image: np.ndarray, context: dict[str, Any] | None = None) -> DefenseResult:
        """Preserve the input without applying a defense."""
        return DefenseResult(
            defense_name="none",
            defense_applied=False,
            defended_image=image.copy(),
            parameters={},
            affected_regions=[],
            change_magnitude=0.0,
            processing_time_ms=0.0,
            explanation="No defense applied because attack evidence is below policy threshold.",
        )


def change_magnitude(original: np.ndarray, defended: np.ndarray) -> float:
    """Return mean absolute pixel change normalized to [0, 1]."""
    return float(np.abs(defended.astype(np.float32) - original.astype(np.float32)).mean() / 255.0)
