"""Tests for the YOLO-backed person-detection tier."""

import numpy as np
import pytest
import torch

from app.models.base_model import ModelPrediction, ResNet18VisionModel, TopPrediction
from app.models.model_registry import person_aware
from app.models.person_detector import PersonAwareModel, PersonDetector, get_person_detector


class FakeYOLO:
    """Mimics the ultralytics YOLO predict surface for tests."""

    def __init__(self, rows: list[tuple[float, float, float, float, float, float]]) -> None:
        self._rows = rows

    def predict(self, image, conf=0.25, verbose=False):  # noqa: ARG002
        # NB: instance attributes, not class-body assignment — `data = data`
        # inside a class body raises NameError (class scope gotcha).
        data = torch.tensor(self._rows, dtype=torch.float32).reshape(-1, 6)
        boxes = type("_Boxes", (), {})()
        boxes.data = data
        result = type("_Result", (), {})()
        result.boxes = boxes
        return [result]


def _prediction(label: str = "swimsuit", confidence: float = 0.61) -> ModelPrediction:
    return ModelPrediction(
        class_id=611,
        class_name=label,
        confidence=confidence,
        top_predictions=(TopPrediction(611, label, confidence),),
        inference_time_ms=1.0,
    )


class StubInner:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, image):  # noqa: ARG002
        self.calls += 1
        return _prediction()

    def prepare_tensor(self, image):  # noqa: ARG002
        raise AssertionError("must pass through, not be reimplemented")

    def __getattr__(self, name):
        raise AttributeError(name)


def _image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)


def test_detector_finds_person_boxes_and_filters_other_classes():
    rows = [
        (10, 10, 50, 90, 0.91, 0),   # person
        (10, 10, 50, 90, 0.80, 5),   # bus — filtered
        (0, 0, 20, 20, 0.10, 0),     # too low — filtered
        (30, 30, 60, 80, 0.55, 0),   # person
    ]
    detector = PersonDetector(yolo_factory=lambda _: FakeYOLO(rows))
    detection = detector.detect(_image())
    assert detection.present and detection.count == 2
    assert detection.max_confidence == pytest.approx(0.91, abs=1e-6)
    assert [box.confidence for box in detection.boxes] == pytest.approx([0.91, 0.55], abs=1e-6)
    assert detection.boxes[0].x1 == 10 and detection.boxes[0].y2 == 90


def test_detector_degrades_gracefully_when_factory_fails():
    def boom(_):
        raise RuntimeError("no weights")

    detector = PersonDetector(yolo_factory=boom)
    detection = detector.detect(_image())
    assert not detection.present and detection.count == 0
    assert detector.load_error is not None


def test_wrapper_labels_person_when_detector_is_confident():
    rows = [(10, 10, 50, 90, 0.92, 0)]
    model = PersonAwareModel(StubInner(), PersonDetector(yolo_factory=lambda _: FakeYOLO(rows)), label_threshold=0.5)
    prediction = model.predict(_image(1))
    assert prediction.class_name == "person"
    assert prediction.class_id == -1
    assert prediction.confidence == pytest.approx(0.92, abs=1e-6)
    assert prediction.top_predictions[0].class_name == "person"


def test_wrapper_passes_through_without_a_confident_person():
    rows = [(10, 10, 50, 90, 0.30, 0)]  # below label threshold
    inner = StubInner()
    model = PersonAwareModel(inner, PersonDetector(yolo_factory=lambda _: FakeYOLO(rows)), label_threshold=0.5)
    prediction = model.predict(_image(2))
    assert prediction.class_name == "swimsuit"
    assert inner.calls == 1


def test_wrapper_caches_detection_for_identical_pixels():
    rows = [(10, 10, 50, 90, 0.92, 0)]
    detector = PersonDetector(yolo_factory=lambda _: FakeYOLO(rows))
    model = PersonAwareModel(StubInner(), detector, label_threshold=0.5)
    image = _image(3)
    first = model.last_person_detection(image)
    second = model.last_person_detection(image)
    assert first is second  # cached, no second YOLO run
    other = model.last_person_detection(_image(4))
    assert other is not first


def test_person_aware_wraps_only_when_detector_available(monkeypatch):
    class StubDetector:
        load_error = None

        def detect(self, image):  # noqa: ARG002
            from app.models.person_detector import PersonDetection

            return PersonDetection(False, 0, 0.0, (), 0.0)

    monkeypatch.setattr("app.models.model_registry.get_person_detector", lambda: StubDetector())
    wrapped = person_aware(ResNet18VisionModel())
    assert isinstance(wrapped, PersonAwareModel)
    # Tensor machinery still reaches the real backbone through passthrough.
    assert isinstance(wrapped._inner, ResNet18VisionModel)

    monkeypatch.setattr("app.models.model_registry.get_person_detector", lambda: None)
    assert not isinstance(person_aware(ResNet18VisionModel()), PersonAwareModel)


def test_person_aware_returns_same_instance_without_feature(monkeypatch):
    monkeypatch.setattr("app.models.model_registry.get_person_detector", lambda: None)
    model = ResNet18VisionModel()
    assert person_aware(model) is model


def test_singleton_returns_none_without_ultralytics(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ultralytics":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    import app.models.person_detector as module

    monkeypatch.setattr(module, "_detector", None)
    assert get_person_detector() is None
