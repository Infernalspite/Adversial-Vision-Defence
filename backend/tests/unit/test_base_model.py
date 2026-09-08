import numpy as np

from app.models.base_model import ResNet18VisionModel


def test_model_loads():
    model = ResNet18VisionModel()
    assert model.model.training is False
    assert model.device.type in {"cpu", "cuda"}


def test_inference_returns_expected_fields():
    prediction = ResNet18VisionModel().predict(np.zeros((224, 224, 3), dtype=np.uint8))
    assert isinstance(prediction.class_id, int)
    assert isinstance(prediction.class_name, str)
    assert 0 <= prediction.confidence <= 1
    assert 0 <= prediction.inference_time_ms
    assert len(prediction.top_predictions) == 5


def test_top_predictions_are_ranked():
    prediction = ResNet18VisionModel().predict(np.zeros((224, 224, 3), dtype=np.uint8))
    confidences = [item.confidence for item in prediction.top_predictions]
    assert confidences == sorted(confidences, reverse=True)