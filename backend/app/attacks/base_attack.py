"""Common attack interface and shared result helpers."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch

from app.models.base_model import ModelPrediction
from app.utils.image import tensor_to_image


class AttackModel(Protocol):
    """Minimum model surface required by gradient-based attacks."""

    device: torch.device
    model: torch.nn.Module

    def predict(self, image: np.ndarray) -> ModelPrediction: ...

    def pixel_to_model_input(self, image: torch.Tensor) -> torch.Tensor: ...

    def predict_tensor(self, image: torch.Tensor) -> ModelPrediction: ...


@dataclass(frozen=True, slots=True)
class AttackResult:
    """Adversarial image output and comparison metadata."""

    original_image: np.ndarray
    adversarial_image: np.ndarray
    attack_type: str
    attack_parameters: dict[str, Any]
    perturbation_magnitude: float
    original_prediction: ModelPrediction
    adversarial_prediction: ModelPrediction
    attack_success: bool
    generation_time_ms: float
    patch_mask: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def image(self) -> np.ndarray:
        """Backward-compatible alias for the adversarial image."""
        return self.adversarial_image


class BaseAttack(ABC):
    """Contract for an attack generator used by the Attack Lab and evaluations."""

    @abstractmethod
    def generate(self, image: np.ndarray, model: AttackModel, target: int | None = None, **kwargs: Any) -> AttackResult:
        """Generate an adversarial example."""
        raise NotImplementedError

    @staticmethod
    def _start(image: np.ndarray) -> tuple[np.ndarray, float]:
        """Copy the source image and start the generation timer."""
        return image.copy(), time.perf_counter()

    @staticmethod
    def _result(
        original: np.ndarray,
        adversarial: torch.Tensor,
        attack_type: str,
        parameters: dict[str, Any],
        model: AttackModel,
        started: float,
        patch_mask: np.ndarray | None = None,
        target: int | None = None,
    ) -> AttackResult:
        """Build shared metadata after an attack tensor has been produced."""
        adversarial_image = tensor_to_image(adversarial)
        original_prediction = model.predict(original)
        adversarial_prediction = model.predict(adversarial_image)
        perturbation = adversarial_image.astype(np.float32) / 255 - original.astype(np.float32) / 255
        return AttackResult(
            original_image=original,
            adversarial_image=adversarial_image,
            attack_type=attack_type,
            attack_parameters=parameters,
            perturbation_magnitude=float(np.max(np.abs(perturbation))),
            original_prediction=original_prediction,
            adversarial_prediction=adversarial_prediction,
            attack_success=(
                adversarial_prediction.class_id != original_prediction.class_id
                if target is None
                else adversarial_prediction.class_id == target
            ),
            generation_time_ms=(time.perf_counter() - started) * 1000,
            patch_mask=patch_mask,
        )
