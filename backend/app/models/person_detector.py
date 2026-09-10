"""Person-detection tier backed by a real object detector (YOLO).

ImageNet-1000 has no "person" class, so a classifier can only guess around
humans (swimsuit, cinematographer, ...). This module adds a genuine detector:
YOLOv8n runs alongside the classifier tier, and when it finds a person the
analysis model reports ``person`` — before *and* after defense — so every
downstream stage (consistency verification, decision engine) sees coherent
labels. Images of the 10 defended classes keep the robust-specialist routing.

The detector is optional infrastructure: if ultralytics or the weights are
unavailable, ``get_person_detector()`` returns ``None`` and the pipeline
serves plain classifier labels exactly as before.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

PERSON_CONFIDENCE_THRESHOLD = 0.45
DEFAULT_YOLO_CONFIDENCE = 0.25
COCO_PERSON_CLASS_ID = 0
DEFAULT_WEIGHTS_NAME = "yolov8n.pt"


@dataclass(frozen=True, slots=True)
class PersonBox:
    """One detected person instance in image pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


@dataclass(frozen=True, slots=True)
class PersonDetection:
    """Aggregated person evidence for one image."""

    present: bool
    count: int
    max_confidence: float
    boxes: tuple[PersonBox, ...]
    inference_time_ms: float
    error: str | None = None


class PersonDetector:
    """YOLO-backed person detector with an injectable model factory."""

    def __init__(
        self,
        yolo_factory: Callable[[str], Any] | None = None,
        weights_name: str = DEFAULT_WEIGHTS_NAME,
        report_confidence: float = DEFAULT_YOLO_CONFIDENCE,
    ) -> None:
        self._yolo_factory = yolo_factory or _default_yolo_factory
        self._weights_name = weights_name
        self._report_confidence = report_confidence
        self._model: Any | None = None
        self._load_error: str | None = None
        self._lock = threading.Lock()

    def _ensure_model(self) -> Any | None:
        with self._lock:
            if self._model is not None or self._load_error is not None:
                return self._model
            try:
                self._model = self._yolo_factory(self._weights_name)
            except Exception as error:  # noqa: BLE001 - optional feature must never crash the API
                self._load_error = str(error)
                return None
            return self._model

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def detect(self, image: np.ndarray) -> PersonDetection:
        """Detect person instances in an RGB image (gracefully empty on failure)."""
        model = self._ensure_model()
        started = time.perf_counter()
        if model is None:
            return PersonDetection(False, 0, 0.0, (), 0.0, self._load_error)
        try:
            results = model.predict(image, conf=self._report_confidence, verbose=False)
            boxes_data = results[0].boxes.data.detach().cpu().numpy() if len(results) else np.empty((0, 6))
        except Exception as error:  # noqa: BLE001 - inference failure degrades to "no person", visibly
            return PersonDetection(False, 0, 0.0, (), (time.perf_counter() - started) * 1000, f"{type(error).__name__}: {error}")
        boxes: list[PersonBox] = []
        for row in np.asarray(boxes_data).reshape(-1, 6):
            x1, y1, x2, y2, confidence, class_id = (float(v) for v in row)
            if int(class_id) != COCO_PERSON_CLASS_ID or confidence < self._report_confidence:
                continue
            boxes.append(PersonBox(int(x1), int(y1), int(x2), int(y2), confidence))
        boxes.sort(key=lambda box: box.confidence, reverse=True)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return PersonDetection(
            present=bool(boxes),
            count=len(boxes),
            max_confidence=boxes[0].confidence if boxes else 0.0,
            boxes=tuple(boxes),
            inference_time_ms=elapsed_ms,
        )


def _default_yolo_factory(weights_name: str) -> Any:
    from ultralytics import YOLO

    weights_dir = Path("data") / "models"
    weights_dir.mkdir(parents=True, exist_ok=True)
    return YOLO(str(weights_dir / weights_name))


_detector: PersonDetector | None = None
_detector_lock = threading.Lock()


def get_person_detector() -> PersonDetector | None:
    """Process-wide detector singleton; ``None`` when the feature is unavailable."""
    global _detector
    with _detector_lock:
        if _detector is None:
            try:
                import ultralytics  # noqa: F401

                _detector = PersonDetector()
            except ImportError:
                return None
        return _detector


class PersonAwareModel:
    """Wraps a vision model and reports ``person`` when a person is detected.

    All tensor-level methods pass through untouched so detectors and attack
    machinery keep operating on the wrapped backbone. ``predict()`` swaps the
    classifier label for a person label only when the detector is confident;
    everything else is a pure passthrough. A tiny cache keyed on the last
    image's bytes avoids running YOLO twice when the pipeline probes the same
    pixels more than once (defense re-inference uses the purified image, which
    differs and is therefore detected fresh).
    """

    def __init__(self, inner: Any, detector: PersonDetector, label_threshold: float = PERSON_CONFIDENCE_THRESHOLD) -> None:
        self._inner = inner
        self._detector = detector
        self._label_threshold = label_threshold
        self._last_key: bytes | None = None
        self._last_detection: PersonDetection | None = None

    def last_person_detection(self, image: np.ndarray | None = None) -> PersonDetection:
        """Detection evidence for this image (cached when identical to the last probe)."""
        key = image.tobytes() if image is not None else self._last_key
        if key is not None and key == self._last_key and self._last_detection is not None:
            return self._last_detection
        if image is None:
            return PersonDetection(False, 0, 0.0, (), 0.0)
        detection = self._detector.detect(image)
        self._last_key = key
        self._last_detection = detection
        return detection

    def predict(self, image: np.ndarray):
        detection = self.last_person_detection(image)
        prediction = self._inner.predict(image)
        if detection.present and detection.max_confidence >= self._label_threshold:
            from app.models.base_model import ModelPrediction, TopPrediction

            top = tuple(
                TopPrediction(class_id=-1, class_name="person", confidence=box.confidence)
                for box in detection.boxes[:5]
            ) + prediction.top_predictions
            return ModelPrediction(
                class_id=-1,
                class_name="person",
                confidence=detection.max_confidence,
                top_predictions=top,
                inference_time_ms=prediction.inference_time_ms + detection.inference_time_ms,
            )
        return prediction

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
