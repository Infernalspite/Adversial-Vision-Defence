import numpy as np
import torch

from app.attacks import get_attack
from app.attacks.adversarial_patch import AdversarialPatchAttack
from app.attacks.fgsm import FGSMAttack
from app.attacks.pgd import PGDAttack


class FakeModel:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, 3))
        self.model.eval()

    def pixel_to_model_input(self, image: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(image.unsqueeze(0), size=(16, 16), mode="bilinear", align_corners=False).squeeze(0)

    def predict_tensor(self, image: torch.Tensor):
        logits = self.model(image)
        values = torch.softmax(logits, dim=1)[0]
        index = int(values.argmax())
        return type("Prediction", (), {"class_id": index, "class_name": str(index), "confidence": float(values[index].detach()), "top_predictions": (), "inference_time_ms": 0.0})()

    def predict(self, image: np.ndarray):
        source = torch.from_numpy(image).permute(2, 0, 1).float().div(255)
        return self.predict_tensor(self.pixel_to_model_input(source).unsqueeze(0))


def test_fgsm_respects_budget_and_preserves_model():
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    model = FakeModel()
    before = [parameter.detach().clone() for parameter in model.model.parameters()]
    result = FGSMAttack().generate(image, model, epsilon=0.02)
    assert result.adversarial_image.shape == image.shape
    assert result.perturbation_magnitude <= 0.02 + 1 / 255
    assert all(torch.equal(old, new) for old, new in zip(before, model.model.parameters()))
    assert result.attack_parameters["epsilon"] == 0.02


def test_pgd_respects_budget_and_iteration_parameter():
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    result = PGDAttack().generate(image, FakeModel(), epsilon=0.03, step_size=0.01, iterations=2, random_start=False)
    assert result.adversarial_image.shape == image.shape
    assert result.perturbation_magnitude <= 0.03 + 1 / 255
    assert result.attack_parameters["iterations"] == 2


def test_patch_is_local_and_returns_mask():
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    result = AdversarialPatchAttack().generate(image, FakeModel(), patch_size=0.25, iterations=1)
    changed = np.any(result.adversarial_image != image, axis=2)
    assert result.adversarial_image.shape == image.shape
    assert result.patch_mask is not None
    assert np.all(~changed | (result.patch_mask == 1))


def test_attack_registry():
    assert isinstance(get_attack("fgsm"), FGSMAttack)
    assert isinstance(get_attack("pgd"), PGDAttack)
    assert isinstance(get_attack("adversarial_patch"), AdversarialPatchAttack)


def test_unknown_attack_fails_cleanly():
    try:
        get_attack("unknown")
    except ValueError as error:
        assert "Unknown attack" in str(error)
    else:
        raise AssertionError("Unknown attacks must fail")
