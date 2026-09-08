"""Seeded, bounded attack-parameter mutation operators."""

import random

from app.learning.schemas import AttackConfig


class AttackMutator:
    """Mutate attack configurations without exceeding configured bounds."""

    def __init__(self, seed: int = 7) -> None:
        self.seed = seed
        self.random = random.Random(seed)

    def mutate(self, config: AttackConfig, generation: int = 1) -> AttackConfig:
        config.validate()
        if config.attack_type == "fgsm":
            return AttackConfig("fgsm", epsilon=min(0.25, config.epsilon * self.random.choice((1.1, 1.25, 1.5))), seed=config.seed)
        if config.attack_type == "pgd":
            return AttackConfig("pgd", epsilon=min(0.25, config.epsilon * self.random.choice((1.1, 1.25, 1.5))), step_size=min(0.25, config.step_size * self.random.choice((1.0, 1.25))), iterations=min(100, max(1, int(config.iterations * self.random.choice((1.25, 1.5))))), random_start=bool(self.random.choice((True, False))), seed=config.seed)
        return AttackConfig("patch", patch_size=min(1.0, config.patch_size * self.random.choice((1.1, 1.25))), patch_location=self.random.choice(("center", "top_left", "top_right", "bottom_left", "bottom_right")), iterations=min(100, max(1, int(config.iterations * 1.25))), seed=config.seed)
