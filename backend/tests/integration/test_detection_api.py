import io

import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient

from app.attacks import get_attack
from app.api.routes import detection
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


def encoded_image(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


def test_clean_and_phase2_images_have_detection_contract(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(detection, "get_model", lambda name: model)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :, 1] = 220
    for sample in [image, get_attack("fgsm").generate(image, model, epsilon=0.01).adversarial_image,
                   get_attack("pgd").generate(image, model, epsilon=0.03, step_size=0.01, iterations=1, random_start=False).adversarial_image,
                   get_attack("adversarial_patch").generate(image, model, patch_size=0.2, iterations=1).adversarial_image]:
        response = client.post(
            "/api/v1/detection/analyze",
            files={"image": ("sample.png", io.BytesIO(encoded_image(sample)), "image/png")},
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert 0 <= payload["attack_score"] <= 1
        assert "feature_squeezing" in payload["detectors"]
        assert "explanation" in payload


def test_invalid_detection_upload_is_rejected(monkeypatch):
    monkeypatch.setattr(detection, "get_model", lambda name: FakeModel())
    response = client.post(
        "/api/v1/detection/analyze",
        files={"image": ("sample.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400