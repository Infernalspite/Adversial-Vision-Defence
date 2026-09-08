"""Adapter from AttackConfig to existing Phase 2 attacks."""

from typing import Any

import numpy as np

from app.attacks import get_attack
from app.learning.schemas import AttackConfig


class EvolutionAttackGenerator:
    """Generate attacks through the existing registry only."""

    def generate(self, image: np.ndarray, model: object, config: AttackConfig) -> Any:
        config.validate()
        attack_name = "adversarial_patch" if config.attack_type == "patch" else config.attack_type
        parameters: dict[str, Any] = {"epsilon": config.epsilon, "step_size": config.step_size, "iterations": config.iterations, "random_start": config.random_start, "patch_size": config.patch_size, "location": config.patch_location}
        return get_attack(attack_name).generate(image, model, target=config.target_class, **parameters)
