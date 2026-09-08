"""Vision-model interfaces and the pretrained ResNet-18 baseline."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torchvision.models import ResNet18_Weights, resnet18

from app.utils.image import normalize_tensor, preprocess_image


@dataclass(frozen=True, slots=True)
class TopPrediction:
    """One ranked ImageNet prediction."""

    class_id: int
    class_name: str
    confidence: float


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    """Structured output from a vision model."""

    class_id: int
    class_name: str
    confidence: float
    top_predictions: tuple[TopPrediction, ...]
    inference_time_ms: float


class BaseVisionModel(ABC):
    """Adapter contract for a baseline or future vision model."""

    @abstractmethod
    def predict(self, image: np.ndarray) -> ModelPrediction:
        """Return a structured prediction for an RGB image."""
        raise NotImplementedError

    @abstractmethod
    def prepare_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Return a differentiable-ready normalized model input."""
        raise NotImplementedError

    @abstractmethod
    def predict_tensor(self, image: torch.Tensor) -> ModelPrediction:
        """Predict from a normalized, batched tensor."""
        raise NotImplementedError


class ResNet18VisionModel(BaseVisionModel):
    """Pretrained ImageNet ResNet-18 inference adapter."""

    name = "resnet18"

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=self.weights).to(self.device)
        self.model.eval()
        self.class_names = self.weights.meta["categories"]

    def predict(self, image: np.ndarray) -> ModelPrediction:
        """Run one clean-image inference and return top-five probabilities."""
        input_tensor = self.prepare_tensor(image).unsqueeze(0)
        return self.predict_tensor(input_tensor)

    def prepare_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert an RGB image to a normalized tensor on the model device."""
        pixel_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255).to(self.device)
        return self.pixel_to_model_input(pixel_tensor)

    def pixel_to_model_input(self, image: torch.Tensor) -> torch.Tensor:
        """Convert a differentiable RGB pixel tensor to normalized model input."""
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("Expected a tensor with shape (3, height, width)")
        height, width = image.shape[1:]
        scale = 256 / min(height, width)
        resized_height = max(224, round(height * scale))
        resized_width = max(224, round(width * scale))
        resized = torch.nn.functional.interpolate(
            image.unsqueeze(0), size=(resized_height, resized_width), mode="bilinear", align_corners=False
        ).squeeze(0)
        top = (resized_height - 224) // 2
        left = (resized_width - 224) // 2
        cropped = resized[:, top : top + 224, left : left + 224]
        return normalize_tensor(cropped)

    def predict_tensor(self, image: torch.Tensor) -> ModelPrediction:
        """Run inference from a normalized, batched tensor."""
        input_tensor = image.to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_time_ms = (time.perf_counter() - started) * 1000

        top_count = min(5, probabilities.shape[0])
        top_values, top_indices = torch.topk(probabilities, k=top_count)
        top_predictions = tuple(
            TopPrediction(
                class_id=int(class_id),
                class_name=self.class_names[int(class_id)],
                confidence=float(confidence),
            )
            for confidence, class_id in zip(top_values.cpu(), top_indices.cpu())
        )
        best = top_predictions[0]
        return ModelPrediction(
            class_id=best.class_id,
            class_name=best.class_name,
            confidence=best.confidence,
            top_predictions=top_predictions,
            inference_time_ms=inference_time_ms,
        )
