"""Small lazy registry for baseline vision models."""

from collections.abc import Callable

from app.models.base_model import BaseVisionModel, ResNet18VisionModel


class ModelRegistry:
    """Resolve named model factories and cache each loaded model."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], BaseVisionModel]] = {
            "resnet18": ResNet18VisionModel,
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
    """Return the configured baseline model."""
    return model_registry.get(name)
