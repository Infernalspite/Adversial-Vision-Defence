import io

import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient

from app.api.routes import inference
from app.main import app

client = TestClient(app)


class FakeModel:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, 1000))
        self.model.eval()

    def pixel_to_model_input(self, image: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(image.unsqueeze(0), size=(16, 16), mode="bilinear", align_corners=False).squeeze(0)

    def predict(self, image: np.ndarray):
        from app.models.base_model import TopPrediction
        source = torch.from_numpy(image).permute(2, 0, 1).float().div(255)
        values = torch.softmax(self.model(self.pixel_to_model_input(source).unsqueeze(0)), dim=1)[0]
        class_id = int(values.argmax())
        confidence = float(values[class_id].detach())
        top = tuple(TopPrediction(int(index), str(int(index)), float(values[index].detach())) for index in torch.topk(values, 5).indices)
        return type("Prediction", (), {"class_id": class_id, "class_name": str(class_id), "confidence": confidence, "top_predictions": top, "inference_time_ms": 0.1})()


def encoded(image: np.ndarray) -> bytes:
    success, payload = cv2.imencode(".png", image)
    assert success
    return payload.tobytes()


def test_main_analyze_returns_one_final_state(monkeypatch):
    monkeypatch.setattr(inference, "get_model", lambda name: FakeModel())
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :, 1] = 220
    response = client.post(
        "/api/v1/analyze",
        data={"mode": "evaluation", "attack_type": "fgsm"},
        files={"image": ("sample.png", io.BytesIO(encoded(image)), "image/png")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["final_state"] in {"TRUSTED", "DEFENDED", "ABSTAIN"}
    assert payload["final_state"] not in {"SAFE", "UNSAFE", "UNKNOWN"}
    assert "attack_score" in payload
    assert "trust_map_reference" in payload
    assert "verification_score" in payload
    assert "decision_reasons" in payload
    assert payload["audit"]
    if payload["final_state"] == "ABSTAIN":
        assert payload["final_prediction"] is None
        assert payload["fallback_action"] == "NO_ACTION"
