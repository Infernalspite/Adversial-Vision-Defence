"""Projected Gradient Descent attack."""

from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from app.attacks.base_attack import AttackModel, AttackResult, BaseAttack


class PGDAttack(BaseAttack):
    """Iterative pixel-space attack projected into an infinity-norm ball."""

    def generate(
        self,
        image: np.ndarray,
        model: AttackModel,
        target: int | None = None,
        epsilon: float = 0.03,
        step_size: float = 0.005,
        iterations: int = 10,
        random_start: bool = True,
        **kwargs: Any,
    ) -> AttackResult:
        """Generate a PGD image using epsilon, step_size, iterations, and random_start."""
        if not 0 < epsilon <= 0.25 or not 0 < step_size <= epsilon:
            raise ValueError("epsilon must be in (0, 0.25] and step_size must be in (0, epsilon]")
        if not 1 <= iterations <= 100:
            raise ValueError("iterations must be between 1 and 100")
        original, started = self._start(image)
        clean = torch.from_numpy(original).permute(2, 0, 1).float().div(255).to(model.device)
        adversarial = clean.detach().clone()
        if random_start:
            adversarial = (adversarial + torch.empty_like(adversarial).uniform_(-epsilon, epsilon)).clamp(0, 1)
        was_training = model.model.training
        model.model.eval()
        try:
            with torch.no_grad():
                label_logits = model.model(model.pixel_to_model_input(clean).unsqueeze(0))
                label = int(label_logits.argmax(dim=1).item()) if target is None else target
            for _ in range(iterations):
                adversarial.requires_grad_(True)
                with torch.enable_grad():
                    logits = model.model(model.pixel_to_model_input(adversarial).unsqueeze(0))
                    loss = F.cross_entropy(logits, torch.tensor([label], device=model.device))
                    gradient = torch.autograd.grad(loss, adversarial)[0]
                direction = -1 if target is not None else 1
                updated = adversarial.detach() + direction * step_size * gradient.sign()
                adversarial = torch.max(torch.min(updated, clean + epsilon), clean - epsilon).clamp(0, 1).detach()
        finally:
            model.model.train(was_training)
        return self._result(
            original,
            adversarial,
            "pgd",
            {"epsilon": epsilon, "step_size": step_size, "iterations": iterations, "random_start": random_start},
            model,
            started,
            target=target,
        )
