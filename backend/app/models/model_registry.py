"""Small lazy registry for baseline and robust vision models."""

from collections.abc import Callable
from pathlib import Path

from app.models.base_model import BaseVisionModel, ResNet18VisionModel, RobustVisionModel


class ModelRegistry:
    """Resolve named model factories and cache each loaded model."""

    def __init__(self) -> None:
        robust_checkpoint = Path(__file__).resolve().parents[3] / "data" / "models" / "robust_model.pt"
        self._factories: dict[str, Callable[[], BaseVisionModel]] = {
            "resnet18": ResNet18VisionModel,
            "robust": lambda: RobustVisionModel(checkpoint_path=robust_checkpoint),
        }
        self._models: dict[str, BaseVisionModel] = {}

    def register(self, name: str, factory: Callable[[], BaseVisionModel]) -> None:
        """Register a model factory under a stable name."""
        self._factories[name] = factory

    def get(self, name: str) -> BaseVisionModel:
        """Load and return a named model, caching it for later requests."""
        if name not in self._factories:
            raise KeyError(f"Unknown model: {name}")
        if name not in self._models:
            self._models[name] = self._factories[name]()
        return self._models[name]


model_registry = ModelRegistry()


def get_model(name: str = "resnet18") -> BaseVisionModel:
    """Return the configured default model, preferring a robust checkpoint when present."""
    if name == "resnet18" and (Path(__file__).resolve().parents[3] / "data" / "models" / "robust_model.pt").exists():
        return model_registry.get("robust")
    return model_registry.get(name)
