import io

import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class FakeModel:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, 3))
        self.model.eval()

    def pixel_to_model_input(self, image: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(
            image.unsqueeze(0), size=(16, 16), mode="bilinear", align_corners=False
        ).squeeze(0)

    def predict_tensor(self, image: torch.Tensor):
        values = torch.softmax(self.model(image), dim=1)[0]
        class_id = int(values.argmax())
        return type(
            "Prediction",
            (),
            {
                "class_id": class_id,
                "class_name": str(class_id),
                "confidence": float(values[class_id].detach()),
            },
        )()

    def predict(self, image: np.ndarray):
        source = torch.from_numpy(image).permute(2, 0, 1).float().div(255)
        return self.predict_tensor(self.pixel_to_model_input(source).unsqueeze(0))


def make_test_image() -> bytes:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :, 1] = 255
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


def test_attack_generation_and_evaluation(monkeypatch):
    monkeypatch.setattr("app.api.routes.attacks.get_model", lambda name: FakeModel())
    for attack in ("fgsm", "pgd", "adversarial_patch"):
        response = client.post(
            "/api/v1/attacks/generate",
            data={"attack": attack, "iterations": "1", "random_start": "false"},
            files={"image": ("sample.png", io.BytesIO(make_test_image()), "image/png")},
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["attack_type"] == attack
        assert payload["original_prediction"]
        assert payload["adversarial_prediction"]
        assert payload["adversarial_image"].startswith("data:image/png;base64,")

    response = client.post(
        "/api/v1/attacks/evaluate",
        data={"attack": "fgsm", "epsilon": "0.02"},
        files={"image": ("sample.png", io.BytesIO(make_test_image()), "image/png")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert "confidence_change" in payload
    assert "prediction_changed" in payload


def test_invalid_attack_file_is_rejected(monkeypatch):
    monkeypatch.setattr("app.api.routes.attacks.get_model", lambda name: FakeModel())
    response = client.post(
        "/api/v1/attacks/generate",
        data={"attack": "fgsm"},
        files={"image": ("sample.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
