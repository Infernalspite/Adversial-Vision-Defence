import numpy as np
import pytest
import torch

from app.models.base_model import ModelPrediction, TopPrediction
from app.models.tier_router import route_tier


def prediction(*names: str) -> ModelPrediction:
    top = tuple(
        TopPrediction(class_id=index, class_name=name, confidence=0.9 - 0.1 * index)
        for index, name in enumerate(names)
    )
    return ModelPrediction(class_id=top[0].class_id, class_name=top[0].class_name, confidence=top[0].confidence, top_predictions=top, inference_time_ms=1.0)


class StubModel:
    name = "stub"

    def predict(self, image: np.ndarray) -> ModelPrediction:  # pragma: no cover - not used by router
        raise NotImplementedError

    def prepare_tensor(self, image: np.ndarray) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def predict_tensor(self, image: torch.Tensor) -> ModelPrediction:  # pragma: no cover
        raise NotImplementedError


def test_robust_tier_engages_on_scientific_suffix_match():
    baseline = prediction("tench, Tinca tinca", "goldfish", "car mirror")
    robust = StubModel()
    routing = route_tier(baseline, StubModel(), robust)
    assert routing.tier == "robust"
    assert routing.analysis_model is robust
    assert routing.reclassify_model is robust
    assert routing.matched_defended_class == "tench"


def test_robust_tier_engages_on_multiword_fuzzy_match():
    baseline = prediction("English springer, English springer spaniel")
    routing = route_tier(baseline, StubModel(), StubModel())
    assert routing.tier == "robust"
    assert routing.matched_defended_class == "English springer spaniel"


def test_baseline_tier_for_non_defended_image():
    baseline = prediction("car mirror", "sunscreen", "comic book")
    baseline_model = StubModel()
    routing = route_tier(baseline, baseline_model, StubModel())
    assert routing.tier == "baseline"
    assert routing.analysis_model is baseline_model
    assert routing.matched_defended_class is None


def test_baseline_tier_when_specialist_missing():
    baseline = prediction("tench, Tinca tinca")
    baseline_model = StubModel()
    routing = route_tier(baseline, baseline_model, None)
    assert routing.tier == "baseline"
    assert routing.analysis_model is baseline_model
    assert "unavailable" in routing.reason
