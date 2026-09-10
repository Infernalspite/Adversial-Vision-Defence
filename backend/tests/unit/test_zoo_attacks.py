import numpy as np
import pytest
import torch

from app.attacks import get_attack, supported_attacks
from app.attacks.pgd import PGDAttack
from app.attacks.zoo import CROP_SIZE, ZOO_ATTACK_NAMES, NormalizedModule, ZooAttack, torchattacks_pgd_success
from app.utils.image import normalize_tensor


class Crop224Model:
    """Fake vision model operating on 224x224 center crops like the zoo path."""

    device = torch.device("cpu")

    def __init__(self, seed: int = 3) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * CROP_SIZE * CROP_SIZE, 3))
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.copy_(torch.randn_like(parameter) * 0.001)
        self.model.eval()

    def pixel_to_model_input(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("Expected (3, height, width)")
        height, width = image.shape[1:]
        top = max(0, (height - CROP_SIZE) // 2)
        left = max(0, (width - CROP_SIZE) // 2)
        crop = image[:, top : top + CROP_SIZE, left : left + CROP_SIZE]
        return normalize_tensor(crop)

    def predict_tensor(self, image: torch.Tensor):
        logits = self.model(image)
        values = torch.softmax(logits, dim=1)[0]
        index = int(values.argmax())
        return type("Prediction", (), {"class_id": index, "class_name": str(index), "confidence": float(values[index].detach()), "top_predictions": (), "inference_time_ms": 0.0})()

    def predict(self, image: np.ndarray):
        source = torch.from_numpy(image).permute(2, 0, 1).float().div(255)
        return self.predict_tensor(self.pixel_to_model_input(source).unsqueeze(0))


def test_zoo_registry_and_supported_attacks():
    for name in ZOO_ATTACK_NAMES:
        assert isinstance(get_attack(name), ZooAttack)
    names = set(supported_attacks())
    assert set(ZOO_ATTACK_NAMES).issubset(names)
    with pytest.raises(ValueError):
        get_attack("not_an_attack")


def test_normalized_module_matches_model_normalization():
    module = NormalizedModule(torch.nn.Identity())
    constant = torch.full((1, 3, 8, 8), 0.5)
    expected = normalize_tensor(constant.squeeze(0))
    assert torch.allclose(module(constant)[0], expected, atol=1e-5)


@pytest.mark.parametrize("name", ZOO_ATTACK_NAMES)
def test_zoo_attack_returns_contract_compliant_result(name):
    image = np.full((CROP_SIZE, CROP_SIZE, 3), 128, dtype=np.uint8)
    model = Crop224Model()
    result = ZooAttack(name).generate(image, model, epsilon=0.05, iterations=3)
    assert result.adversarial_image.shape == image.shape
    assert result.original_image.shape == image.shape
    assert result.attack_type == name
    assert 0.0 <= result.perturbation_magnitude <= 1.0
    assert result.attack_parameters["library"] in {"torchattacks", "art"}
    assert result.adversarial_prediction.class_id in {0, 1, 2}


def test_native_pgd_and_torchattacks_pgd_agree_on_identical_input():
    image = np.full((CROP_SIZE, CROP_SIZE, 3), 128, dtype=np.uint8)
    epsilon, step_size, iterations = 0.05, 0.01, 10

    native_model = Crop224Model()
    native = PGDAttack().generate(
        image, native_model, epsilon=epsilon, step_size=step_size, iterations=iterations, random_start=False
    )

    torchattacks_model = Crop224Model()
    torchattacks_success, _ = torchattacks_pgd_success(
        torchattacks_model, image, epsilon=epsilon, step_size=step_size, iterations=iterations
    )

    assert native.attack_success == torchattacks_success
