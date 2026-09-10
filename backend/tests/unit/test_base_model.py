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


def test_predictions_are_the_models_own_labels():
    """No label override: skin-tone pixels must not force "person".

    This pins the removal of the old skin-mask heuristic, which rewrote the
    top-1 label of nearly every natural photo (wood, sand, fur, food all
    match skin-tone ranges) to "person" with a fabricated 0.72 confidence.
    """
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[40:180, 70:150] = [180, 140, 120]
    image[120:200, 40:180] = [90, 70, 55]
    image[20:40, 70:150] = [200, 170, 150]

    prediction = ResNet18VisionModel().predict(image)

    assert prediction.class_name != "person"
    assert prediction.class_name in ResNet18VisionModel().class_names
    names = [item.class_name for item in prediction.top_predictions]
    assert all(name != "person" for name in names)