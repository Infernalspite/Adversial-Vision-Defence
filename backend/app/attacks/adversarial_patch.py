"""Localized adversarial patch attack."""

from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from app.attacks.base_attack import AttackModel, AttackResult, BaseAttack


class AdversarialPatchAttack(BaseAttack):
    """Optimize a square patch over a bounded image region."""

    def generate(
        self,
        image: np.ndarray,
        model: AttackModel,
        target: int | None = None,
        patch_size: float = 0.2,
        location: str = "center",
        iterations: int = 20,
        learning_rate: float = 0.05,
        **kwargs: Any,
    ) -> AttackResult:
        """Optimize a square patch; patch_size is a fraction of the shorter side."""
        if not 0.01 <= patch_size <= 1:
            raise ValueError("patch_size must be between 0.01 and 1")
        if not 1 <= iterations <= 100:
            raise ValueError("iterations must be between 1 and 100")
        original, started = self._start(image)
        height, width = original.shape[:2]
        side = max(1, min(height, width, round(min(height, width) * patch_size)))
        top, left = self._location(location, height, width, side)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[top : top + side, left : left + side] = 1
        clean = torch.from_numpy(original).permute(2, 0, 1).float().div(255).to(model.device)
        patch = torch.rand((3, side, side), device=model.device, requires_grad=True)
        optimizer = torch.optim.Adam([patch], lr=learning_rate)
        was_training = model.model.training
        model.model.eval()
        try:
            with torch.no_grad():
                label_logits = model.model(model.pixel_to_model_input(clean).unsqueeze(0))
                label = int(label_logits.argmax(dim=1).item()) if target is None else target
            for _ in range(iterations):
                optimizer.zero_grad(set_to_none=True)
                candidate = clean.clone()
                candidate[:, top : top + side, left : left + side] = patch
                logits = model.model(model.pixel_to_model_input(candidate).unsqueeze(0))
                loss = F.cross_entropy(logits, torch.tensor([label], device=model.device))
                (-loss if target is None else loss).backward()
                optimizer.step()
                with torch.no_grad():
                    patch.clamp_(0, 1)
            with torch.no_grad():
                adversarial = clean.clone()
                adversarial[:, top : top + side, left : left + side] = patch
        finally:
            model.model.train(was_training)
        return self._result(
            original,
            adversarial,
            "adversarial_patch",
            {"patch_size": patch_size, "location": location, "iterations": iterations, "learning_rate": learning_rate},
            model,
            started,
            patch_mask=mask,
            target=target,
        )

    @staticmethod
    def _location(location: str, height: int, width: int, side: int) -> tuple[int, int]:
        positions = {
            "center": ((height - side) // 2, (width - side) // 2),
            "top_left": (0, 0),
            "top_right": (0, width - side),
            "bottom_left": (height - side, 0),
            "bottom_right": (height - side, width - side),
        }
        if location not in positions:
            raise ValueError(f"Unsupported patch location: {location}")
        return positions[location]
