import numpy as np
import pytest
import torch

from app.detectors import get_detector
from app.detectors.attack_scorer import AttackScorerConfig, UnifiedAttackScorer
from app.detectors.detection_pipeline import DetectionPipeline
from app.detectors.base_detector import DetectorResult


class FakeModel:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, 3))
        self.model.eval()

    def pixel_to_model_input(self, image: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(
            image.unsqueeze(0), size=(16, 16), mode="bilinear", align_corners=False
        ).squeeze(0)

    def predict(self, image: np.ndarray):
        source = torch.from_numpy(image).permute(2, 0, 1).float().div(255)
        values = torch.softmax(self.model(self.pixel_to_model_input(source).unsqueeze(0)), dim=1)[0]
        class_id = int(values.argmax())
        return type(
            "Prediction",
            (),
            {"class_id": class_id, "class_name": str(class_id), "confidence": float(values[class_id].detach())},
        )()


@pytest.fixture
def image() -> np.ndarray:
    value = np.zeros((32, 32, 3), dtype=np.uint8)
    value[:, :, 1] = 220
    return value


@pytest.mark.parametrize(
    "name",
    ["feature_squeezing", "frequency_analysis", "confidence_instability", "saliency_analysis"],
)
def test_each_detector_returns_bounded_evidence(name: str, image: np.ndarray) -> None:
    result = get_detector(name).detect(image, {"model": FakeModel()})
    assert 0 <= result.score <= 1
    assert 0 <= result.confidence <= 1
    assert result.evidence
    assert result.processing_time_ms >= 0


def test_saliency_returns_valid_map(image: np.ndarray) -> None:
    result = get_detector("saliency_analysis").detect(image, {"model": FakeModel()})
    saliency_map = result.evidence["saliency_map"]
    assert len(saliency_map) == image.shape[0]
    assert len(saliency_map[0]) == image.shape[1]


def test_scorer_combines_weights_and_threshold() -> None:
    config = AttackScorerConfig(
        weights={"a": 0.75, "b": 0.25},
        detection_threshold=0.7,
    )
    scorer = UnifiedAttackScorer(config)
    result = scorer.score(
        [
            DetectorResult("a", 0.8, True, 0.8),
            DetectorResult("b", 0.4, False, 0.4),
        ]
    )
    assert result.score == pytest.approx(0.7)
    assert result.detected is True
    assert result.detection_threshold == 0.7


def test_invalid_scorer_configuration_fails_cleanly() -> None:
    with pytest.raises(ValueError):
        AttackScorerConfig(weights={"a": -1})
    with pytest.raises(ValueError):
        AttackScorerConfig(weights={"a": 0})
    with pytest.raises(ValueError):
        AttackScorerConfig(detection_threshold=2)


def test_detection_pipeline_collects_all_detectors(image: np.ndarray) -> None:
    result = DetectionPipeline().analyze(image, FakeModel())
    assert set(result.detectors) == {
        "feature_squeezing", "frequency_analysis", "confidence_instability", "saliency_analysis",
    }
    assert 0 <= result.attack_score <= 1
