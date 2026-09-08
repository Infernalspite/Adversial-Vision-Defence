"""Fast Gradient Sign Method attack."""

from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from app.attacks.base_attack import AttackModel, AttackResult, BaseAttack


class FGSMAttack(BaseAttack):
    """Single-step untargeted or targeted pixel-space FGSM."""

    def generate(
        self,
        image: np.ndarray,
        model: AttackModel,
        target: int | None = None,
        epsilon: float = 0.01,
        **kwargs: Any,
    ) -> AttackResult:
        """Generate an image with a pixel-wise infinity-norm budget."""
        if not 0 < epsilon <= 0.25:
            raise ValueError("epsilon must be greater than 0 and at most 0.25")
        original, started = self._start(image)
        source = torch.from_numpy(original).permute(2, 0, 1).float().div(255).to(model.device)
        source.requires_grad_(True)
        was_training = model.model.training
        model.model.eval()
        try:
            with torch.enable_grad():
                clean_input = model.pixel_to_model_input(source)
                with torch.no_grad():
                    clean_logits = model.model(clean_input.unsqueeze(0))
                    clean_label = int(clean_logits.argmax(dim=1).item()) if target is None else target
                logits = model.model(model.pixel_to_model_input(source).unsqueeze(0))
                loss = F.cross_entropy(logits, torch.tensor([clean_label], device=model.device))
                gradient = torch.autograd.grad(loss, source)[0]
                direction = -1 if target is not None else 1
                adversarial = (source + direction * epsilon * gradient.sign()).clamp(0, 1).detach()
        finally:
            model.model.train(was_training)
        return self._result(
            original, adversarial, "fgsm", {"epsilon": epsilon}, model, started, target=target
        )
