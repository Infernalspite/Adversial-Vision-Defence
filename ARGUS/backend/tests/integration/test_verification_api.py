import io

import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient

from app.api.routes import verification
from app.attacks import get_attack
from app.main import app
from app.pipeline.defense_pipeline import ArgusDefensePipeline
from app.verification.verification_pipeline import VerificationPipeline

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
        return type("Prediction", (), {"class_id": class_id, "class_name": str(class_id), "confidence": float(values[class_id].detach()), "top_predictions": ()})()


def encoded(image: np.ndarray) -> bytes:
    success, payload = cv2.imencode(".png", image)
    assert success
    return payload.tobytes()


def test_verification_api_returns_all_components(monkeypatch):
    monkeypatch.setattr(verification, "get_model", lambda name: FakeModel())
    original = np.zeros((32, 32, 3), dtype=np.uint8)
    original[:, :, 1] = 220
    defended = original.copy()
    response = client.post(
        "/api/v1/verification/analyze",
        files={
            "original_image": ("original.png", io.BytesIO(encoded(original)), "image/png"),
            "defended_image": ("defended.png", io.BytesIO(encoded(defended)), "image/png"),
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["original_prediction"]
    assert payload["defended_prediction"]
    assert "object_consistency" in payload
    assert "geometry_consistency" in payload
    assert "scene_consistency" in payload
    assert 0 <= payload["verification_score"] <= 1
    assert payload["explanation"]


def test_invalid_verification_upload_is_rejected(monkeypatch):
    monkeypatch.setattr(verification, "get_model", lambda name: FakeModel())
    response = client.post(
        "/api/v1/verification/analyze",
        files={
            "original_image": ("original.txt", b"not image", "text/plain"),
            "defended_image": ("defended.png", b"not image", "image/png"),
        },
    )
    assert response.status_code == 400


def test_full_phase3_to_phase6_flow_for_clean_and_phase2_attacks():
    model = FakeModel()
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :, 1] = 220
    samples = [
        image,
        get_attack("fgsm").generate(image, model, epsilon=0.01).adversarial_image,
        get_attack("pgd").generate(image, model, epsilon=0.03, step_size=0.01, iterations=1, random_start=False).adversarial_image,
        get_attack("adversarial_patch").generate(image, model, patch_size=0.2, iterations=1).adversarial_image,
    ]
    for sample in samples:
        defended = ArgusDefensePipeline().run(sample, model)
        verified = VerificationPipeline().run(
            sample,
            defended.orchestration.defense.defended_image,
            defended.original_prediction,
            defended.defended_prediction,
            defended.trust.trust_map,
            defended.trust.localization.regions,
            defended.detection,
        )
        assert 0 <= verified.verification.verification_score <= 1
        assert verified.verification.object_consistency
        assert verified.verification.geometry_consistency
        assert verified.verification.scene_consistency