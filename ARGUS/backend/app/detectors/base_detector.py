"""Base detector contract and shared model protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch


class DetectorModel(Protocol):
    """Minimum model surface required by Phase 3 detectors."""

    device: torch.device
    model: torch.nn.Module

    def predict(self, image: np.ndarray) -> Any: ...

    def pixel_to_model_input(self, image: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class DetectorResult:
    """Structured evidence emitted by a detector."""

    detector_name: str
    score: float
    detected: bool
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseDetector(ABC):
    """Contract for one independent adversarial signal."""

    @abstractmethod
    def detect(self, image: np.ndarray, context: dict[str, Any] | None = None) -> DetectorResult:
        """Analyze an image and return structured detector evidence."""
        raise NotImplementedError


def detector_context(context: dict[str, Any] | None) -> tuple[DetectorModel, float]:
    """Extract the required model and local detector threshold from context."""
    values = context or {}
    model = values.get("model")
    if model is None:
        raise ValueError("Detector context must include a model")
    threshold = float(values.get("detector_threshold", 0.5))
    return model, max(0.0, min(1.0, threshold))


def clip_score(value: float) -> float:
    """Normalize a detector score to the public [0, 1] interval."""
    return max(0.0, min(1.0, float(value)))
