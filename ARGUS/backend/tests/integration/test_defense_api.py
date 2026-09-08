import io

import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient

from app.api.routes import defense
from app.main import app

client = TestClient(app)


class FakeModel:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, 3))
        self.model.eval()

    def pixel_to_model_input(self, image: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(image.unsqueeze(0), size=(16, 16), mode="bilinear", align_corners=False).squeeze(0)

    def predict(self, image: np.ndarray):
        source = torch.from_numpy(image).permute(2, 0, 1).float().div(255)
        values = torch.softmax(self.model(self.pixel_to_model_input(source).unsqueeze(0)), dim=1)[0]
        class_id = int(values.argmax())
        return type("Prediction", (), {"class_id": class_id, "class_name": str(class_id), "confidence": float(values[class_id].detach())})()


def encoded(image: np.ndarray) -> bytes:
    success, payload = cv2.imencode(".png", image)
    assert success
    return payload.tobytes()


def test_defense_apply_returns_full_phase5_contract(monkeypatch):
    monkeypatch.setattr(defense, "get_model", lambda name: FakeModel())
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :, 1] = 220
    response = client.post(
        "/api/v1/defense/apply",
        files={"image": ("sample.png", io.BytesIO(encoded(image)), "image/png")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["original_prediction"]
    assert payload["defended_prediction"]
    assert 0 <= payload["attack_score"] <= 1
    assert 0 <= payload["global_trust_score"] <= 1
    assert payload["selected_defense"] in {"none", "transform", "mask", "purification"}
    assert payload["defended_image_reference"].startswith("data:image/png;base64,")
    assert payload["defense_trace"]["success"] is True


def test_invalid_defense_upload_is_rejected(monkeypatch):
    monkeypatch.setattr(defense, "get_model", lambda name: FakeModel())
    response = client.post(
        "/api/v1/defense/apply",
        files={"image": ("sample.txt", b"not image", "text/plain")},
    )
    assert response.status_code == 400