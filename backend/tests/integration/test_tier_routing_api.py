import io

import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient

from app.api.routes import inference
from app.main import app

client = TestClient(app)


class NamedClassModel:
    """FakeModel-style stub with a real 1000-class head; only the top-1 label is controlled.

    The detectors consume real logits, while the tier router sees a chosen
    class name (e.g. ImageNet's "tench, Tinca tinca") in top predictions.
    """

    device = torch.device("cpu")

    def __init__(self, top_class_name: str) -> None:
        self.top_class_name = top_class_name
        self.model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, 1000))
        self.model.eval()

    def pixel_to_model_input(self, image: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(image.unsqueeze(0), size=(16, 16), mode="bilinear", align_corners=False).squeeze(0)

    def predict(self, image: np.ndarray):
        from app.models.base_model import ModelPrediction, TopPrediction

        source = torch.from_numpy(image).permute(2, 0, 1).float().div(255)
        values = torch.softmax(self.model(self.pixel_to_model_input(source).unsqueeze(0)), dim=1)[0]
        top = tuple(TopPrediction(int(index), self.top_class_name if position == 0 else str(int(index)), float(values[index].detach())) for position, index in enumerate(torch.topk(values, 5).indices))
        return ModelPrediction(class_id=top[0].class_id, class_name=top[0].class_name, confidence=top[0].confidence, top_predictions=top, inference_time_ms=0.1)

    def predict_tensor(self, image: torch.Tensor):
        return self.predict(np.zeros((16, 16, 3), dtype=np.uint8))

    def prepare_tensor(self, image: np.ndarray) -> torch.Tensor:
        source = torch.from_numpy(image).permute(2, 0, 1).float().div(255)
        return self.pixel_to_model_input(source)


def encoded(image: np.ndarray) -> bytes:
    success, payload = cv2.imencode(".png", image)
    assert success
    return payload.tobytes()


def _post(image: np.ndarray) -> dict:
    response = client.post(
        "/api/v1/analyze",
        data={"mode": "standard"},
        files={"image": ("sample.png", io.BytesIO(encoded(image)), "image/png")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_analyze_reports_baseline_tier_without_specialist(monkeypatch):
    baseline = NamedClassModel("car mirror")
    monkeypatch.setattr(inference, "get_model", lambda name: baseline)
    monkeypatch.setattr(inference.model_registry, "available", lambda name: False)
    payload = _post(np.zeros((32, 32, 3), dtype=np.uint8))
    assert payload["tier"] == "baseline"
    assert "unavailable" in payload["tier_reason"]
    assert payload["final_state"] in {"TRUSTED", "DEFENDED", "ABSTAIN"}


def test_analyze_routes_defended_class_to_robust_tier(monkeypatch):
    baseline = NamedClassModel("tench, Tinca tinca")
    robust = NamedClassModel("tench")
    monkeypatch.setattr(inference, "get_model", lambda name: robust if name == "robust" else baseline)
    monkeypatch.setattr(inference.model_registry, "available", lambda name: True)
    monkeypatch.setattr(inference.model_registry, "get", lambda name: robust if name == "robust" else baseline)
    payload = _post(np.zeros((32, 32, 3), dtype=np.uint8))
    assert payload["tier"] == "robust"
    assert payload["matched_defended_class"] == "tench"
    assert payload["suspicion_heatmap_reference"].startswith("data:image/png;base64,")
    assert set(payload["pixel_trace"]) >= {"suspicion_mean", "suspicious_pixel_fraction", "num_suspicious_regions"}
