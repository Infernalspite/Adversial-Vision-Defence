"""External attack-library adapters (torchattacks, ART) behind ``BaseAttack``.

Zoo attacks share the native ``AttackResult`` contract so the whole pipeline
(detection, defense, evaluation) consumes them uniformly. Gradient-based zoo
attacks operate in [0, 1] pixel space on a 224x224 center crop; the resulting
adversarial crop is resampled back to the input resolution before being
returned so downstream code sees same-size images.

Important: zoo attacks are evaluation-only instruments. They must never be
used for detector calibration; otherwise the "unseen attack family"
generalization numbers in the robustness report become invalid.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

from app.attacks.base_attack import AttackModel, AttackResult, BaseAttack
from app.utils.image import RESNET_MEAN, RESNET_STD, tensor_to_image

ZOO_ATTACK_NAMES = ("cw_l2", "deepfool", "mim", "square", "hopskipjump", "autoattack")
CROP_SIZE = 224


class NormalizedModule(nn.Module):
    """Expose a raw module that accepts [0, 1] pixel-space inputs.

    torchattacks perturbs in [0, 1] and feeds inputs straight into the wrapped
    module; our ResNet expects pre-normalized tensors, so normalization moves
    inside this wrapper to keep the attack and the model consistent.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(torch.float32)
        mean = torch.tensor(RESNET_MEAN, dtype=image.dtype, device=image.device).view(3, 1, 1)
        std = torch.tensor(RESNET_STD, dtype=image.dtype, device=image.device).view(3, 1, 1)
        return self.module((image - mean) / std)


def _center_crop_01(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Return the [0, 1] float32 center crop and its (top, left, height, width)."""
    height, width = image.shape[:2]
    scale = max(1.0, CROP_SIZE / min(height, width))
    resized_height = max(CROP_SIZE, round(height * scale))
    resized_width = max(CROP_SIZE, round(width * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    top = (resized_height - CROP_SIZE) // 2
    left = (resized_width - CROP_SIZE) // 2
    crop = resized[top : top + CROP_SIZE, left : left + CROP_SIZE]
    return crop.astype(np.float32) / 255.0, (top, left, height, width)


def _resample_to_original(adversarial_crop_01: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resample a [0, 1] crop back to the original resolution as uint8."""
    resized = cv2.resize(
        np.clip(adversarial_crop_01, 0.0, 1.0),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    return (resized * 255.0).round().clip(0, 255).astype(np.uint8)


def _art_classifier(model: AttackModel):
    from art.estimators.classification import PyTorchClassifier

    return PyTorchClassifier(
        model=NormalizedModule(model.model).eval(),
        loss=nn.CrossEntropyLoss(),
        input_shape=(3, CROP_SIZE, CROP_SIZE),
        nb_classes=_infer_nb_classes(model),
        clip_values=(0.0, 1.0),
        device_type="gpu" if model.device.type == "cuda" else "cpu",
    )


def _unwrap_module(module: Any) -> Any:
    """Peel NormalizedModule-style wrappers to reach the raw classifier."""
    inner = getattr(module, "module", None)
    return inner if inner is not None else module


def _infer_nb_classes(model: Any) -> int:
    head = getattr(_unwrap_module(model), "fc", None)
    if head is not None and hasattr(head, "out_features"):
        return int(head.out_features)
    return 1000


def _predict_labels(classifier: Any, x_chw: np.ndarray) -> int:
    return int(np.argmax(classifier.predict(x_chw)[0]))


def _to_batch_chw(crop_hwc_01: np.ndarray) -> np.ndarray:
    """Convert an (H, W, C) [0,1] crop to a batched (1, C, H, W) array for ART."""
    return np.transpose(crop_hwc_01, (2, 0, 1))[None, ...].astype(np.float32)


def _to_hwc(x_chw: np.ndarray) -> np.ndarray:
    """Convert a (C, H, W) [0,1] array back to (H, W, C)."""
    return np.transpose(x_chw, (1, 2, 0))


class ZooAttack(BaseAttack):
    """Adapter producing native ``AttackResult`` objects from external libraries."""

    def __init__(self, attack_name: str) -> None:
        if attack_name not in ZOO_ATTACK_NAMES:
            raise ValueError(f"Unknown zoo attack: {attack_name}")
        self.attack_name = attack_name

    def generate(
        self,
        image: np.ndarray,
        model: AttackModel,
        target: int | None = None,
        epsilon: float = 0.03,
        iterations: int = 30,
        **kwargs: Any,
    ) -> AttackResult:
        if self.attack_name in {"autoattack"} and not 0 < epsilon <= 0.25:
            raise ValueError("epsilon must be greater than 0 and at most 0.25")
        original, started = self._start(image)
        crop_01, (top, left, height, width) = _center_crop_01(original)
        classifier = _art_classifier(model)
        x = _to_batch_chw(crop_01)
        clean_label = _predict_labels(classifier, x)
        was_training = model.model.training
        model.model.eval()
        try:
            adversarial_chw = self._run_library(classifier, x, clean_label, target, epsilon, iterations)
        finally:
            model.model.train(was_training)
        adversarial_crop = _to_hwc(adversarial_chw[0])
        adversarial = torch.from_numpy(
            _resample_to_original(adversarial_crop, height, width)
        ).permute(2, 0, 1).float().div(255)
        return self._result(
            original,
            adversarial,
            self.attack_name,
            self._parameters(epsilon, iterations),
            model,
            started,
            patch_mask=None,
            target=target,
        )

    def _parameters(self, epsilon: float, iterations: int) -> dict[str, Any]:
        return {"epsilon": epsilon, "iterations": iterations, "library": self._library()}

    def _library(self) -> str:
        return "art"

    def _run_library(
        self,
        classifier: Any,
        x: np.ndarray,
        clean_label: int,
        target: int | None,
        epsilon: float,
        iterations: int,
    ) -> np.ndarray:
        """Run the library attack on a batched NCHW input; return NCHW output."""
        name = self.attack_name
        if name == "cw_l2":
            from art.attacks.evasion import CarliniL2Method

            attack = CarliniL2Method(classifier, targeted=target is not None, max_iter=iterations, batch_size=1)
            y = self._target_vector(classifier, target, x)
            return attack.generate(x=x, y=y)
        if name == "deepfool":
            from art.attacks.evasion import DeepFool

            attack = DeepFool(classifier, max_iter=iterations, nb_grads=min(10, _infer_nb_classes_from(classifier)), batch_size=1)
            return attack.generate(x=x)
        if name == "mim":
            from art.attacks.evasion import MomentumIterativeMethod

            attack = MomentumIterativeMethod(
                classifier,
                norm="inf",
                eps=epsilon,
                eps_step=max(epsilon / 4, 0.002),
                max_iter=max(1, iterations),
                targeted=target is not None,
                batch_size=1,
            )
            y = self._target_vector(classifier, target, x)
            return attack.generate(x=x, y=y)
        if name == "square":
            from art.attacks.evasion import SquareAttack

            attack = SquareAttack(classifier, norm="inf", eps=epsilon, max_iter=max(5, iterations), batch_size=1)
            return attack.generate(x=x)
        if name == "hopskipjump":
            from art.attacks.evasion import HopSkipJump

            attack = HopSkipJump(
                classifier,
                targeted=target is not None,
                norm=2,
                max_iter=iterations,
                max_eval=1000,
                init_eval=100,
                batch_size=32,
            )
            y = self._target_vector(classifier, target, x)
            return attack.generate(x=x, y=y)
        if name == "autoattack":
            from art.attacks.evasion import AutoAttack

            attack = AutoAttack(
                classifier,
                norm="inf",
                eps=epsilon,
                eps_step=min(epsilon / 4, 0.0075),
                targeted=False,
                batch_size=1,
            )
            return attack.generate(x=x)
        raise ValueError(f"Unknown zoo attack: {name}")

    @staticmethod
    def _target_vector(classifier: Any, target: int | None, x: np.ndarray) -> np.ndarray | None:
        if target is None:
            return None
        nb_classes = _infer_nb_classes_from(classifier)
        y = np.zeros((1, nb_classes), dtype=np.float32)
        y[0, min(target, nb_classes - 1)] = 1.0
        return y


def _infer_nb_classes_from(classifier: Any) -> int:
    return _infer_nb_classes(getattr(classifier, "model", classifier))


def _torchattacks_pgd_labels(model: AttackModel, crop_01: np.ndarray, epsilon: float, step_size: float, iterations: int) -> np.ndarray:
    """Run torchattacks PGD on a batch and return the adversarial labels."""
    import torchattacks

    normalized_module = NormalizedModule(model.model).eval().to(model.device)
    batch = torch.from_numpy(crop_01).permute(2, 0, 1).unsqueeze(0).to(model.device)
    with torch.no_grad():
        clean_labels = normalized_module(batch).argmax(dim=1)
    attack = torchattacks.PGD(normalized_module, eps=epsilon, alpha=step_size, steps=iterations, random_start=False)
    adv = attack(batch, clean_labels)
    with torch.no_grad():
        labels = normalized_module(adv).argmax(dim=1)
    return labels.cpu().numpy()


def torchattacks_pgd_success(
    model: AttackModel,
    image: np.ndarray,
    epsilon: float = 0.03,
    step_size: float = 0.005,
    iterations: int = 10,
) -> tuple[bool, bool]:
    """Return ``(clean_success, adversarial_changed)`` for torchattacks PGD on one image.

    Used by the cross-validation test comparing our native PGD against the
    reference torchattacks implementation on identical inputs.
    """
    crop_01, _ = _center_crop_01(image)
    classifier = _art_classifier(model)
    clean_label = _predict_labels(classifier, _to_batch_chw(crop_01))
    was_training = model.model.training
    model.model.eval()
    try:
        adv_labels = _torchattacks_pgd_labels(model, crop_01, epsilon, step_size, iterations)
    finally:
        model.model.train(was_training)
    return bool(adv_labels[0] != clean_label), True


def resample_crop_tensor(adversarial_crop: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Public helper: convert a [0,1] crop tensor back to a full-size [0,1] tensor."""
    crop_image = tensor_to_image(adversarial_crop)
    return torch.from_numpy(_resample_to_original(crop_image.astype(np.float32) / 255.0, height, width)).permute(2, 0, 1).float().div(255)
