"""Small lazy registry for baseline and robust vision models.

Tier routing is explicit: ``resnet18`` is always the frozen ImageNet-1000
baseline (recognizes any image); ``robust`` is always the adversarially
trained specialist for the defended classes. Nothing swaps models silently
behind the caller's back.
"""

from collections.abc import Callable
from pathlib import Path

from app.models.base_model import BaseVisionModel, ResNet18VisionModel, RobustVisionModel
from app.models.person_detector import PersonAwareModel, get_person_detector

ROBUST_CHECKPOINT = Path(__file__).resolve().parents[3] / "data" / "models" / "robust_model.pt"


class ModelRegistry:
    """Resolve named model factories and cache each loaded model."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], BaseVisionModel]] = {
            "resnet18": ResNet18VisionModel,
            "robust": lambda: RobustVisionModel(checkpoint_path=ROBUST_CHECKPOINT),
        }
        self._models: dict[str, BaseVisionModel] = {}

    def register(self, name: str, factory: Callable[[], BaseVisionModel]) -> None:
        """Register a model factory under a stable name."""
        self._factories[name] = factory

    def get(self, name: str) -> BaseVisionModel:
        """Load and return a named model, caching it for later requests.

        Models here are raw: person-aware labeling is a serving-path concern
        applied by the API routes (see ``person_aware``), so evaluation and
        calibration scripts always see the unmodified classifier.
        """
        if name not in self._factories:
            raise KeyError(f"Unknown model: {name}")
        if name not in self._models:
            self._models[name] = self._factories[name]()
        return self._models[name]

    def available(self, name: str) -> bool:
        """Return whether a model can be constructed right now."""
        if name == "robust":
            return ROBUST_CHECKPOINT.exists()
        return name in self._factories


model_registry = ModelRegistry()


def person_aware(model: BaseVisionModel) -> BaseVisionModel:
    """Wrap a serving-path model with person-aware labeling.

    Used only by API routes: human photos are labeled ``person`` (a real
    YOLO detection, since ImageNet-1000 has no person class) while evaluation
    and calibration code keep the unmodified classifier. Returns the model
    unchanged when the detector feature is unavailable.
    """
    detector = get_person_detector()
    if detector is None:
        return model
    return PersonAwareModel(model, detector)


def get_model(name: str = "resnet18") -> BaseVisionModel:
    """Return the explicitly requested model tier.

    ``resnet18`` always returns the frozen ImageNet-1000 baseline. ``robust``
    requires its checkpoint to exist and raises otherwise, so callers can
    never unknowingly run against a stale or missing specialist.
    """
    if name == "robust" and not ROBUST_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Robust specialist checkpoint not found at {ROBUST_CHECKPOINT}; run scripts/train_robust_model.py first"
        )
    return model_registry.get(name)
