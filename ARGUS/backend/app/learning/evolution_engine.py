"""Bounded offline attack evolution engine."""

from typing import Any, Callable

import numpy as np

from app.learning.attack_generator import EvolutionAttackGenerator
from app.learning.difficulty import DifficultyScorer
from app.learning.experiment_manager import ExperimentManager
from app.learning.mutation import AttackMutator
from app.learning.schemas import AttackConfig


class EvolutionEngine:
    """Generate, evaluate, rank, and mutate attack populations offline."""

    def __init__(self, seed: int = 7, experiment_manager: ExperimentManager | None = None) -> None:
        self.mutator = AttackMutator(seed); self.difficulty = DifficultyScorer(); self.generator = EvolutionAttackGenerator(); self.experiments = experiment_manager or ExperimentManager()

    def run(self, image: np.ndarray, model: object, attack_type: str, generations: int = 3, population_size: int = 20, evaluator: Callable[[Any], dict[str, Any]] | None = None) -> dict[str, Any]:
        if not 1 <= generations <= 10 or not 1 <= population_size <= 100: raise ValueError("Evolution bounds exceeded")
        experiment_id, path = self.experiments.create({"attack_type": attack_type, "generations": generations, "population_size": population_size, "seed": self.mutator.seed})
        config = AttackConfig(attack_type)
        generation_reports = []
        population = [config for _ in range(population_size)]
        for generation in range(generations):
            results = []
            for item in population:
                attack = self.generator.generate(image, model, item)
                result = evaluator(attack) if evaluator else {"attack_success": attack.attack_success, "adversarial_confidence": attack.adversarial_prediction.confidence, "attack_score": 0.0, "defense_recovered": False, "verification_score": 0.0, "attack_parameters": item.as_dict()}
                result = {**result, "difficulty_score": self.difficulty.score(result), "attack_parameters": item.as_dict()}
                results.append(result)
            results.sort(key=lambda item: item["difficulty_score"], reverse=True)
            self.experiments.save_generation(path, generation, results)
            generation_reports.append({"generation": generation, "best_difficulty": results[0]["difficulty_score"] if results else 0.0, "results": results})
            population = [self.mutator.mutate(config, generation) for _ in range(population_size)]
        report = {"experiment_id": experiment_id, "generations": generation_reports, "status": "CANDIDATE", "production_modified": False}
        self.experiments.save_report(path, report)
        return report
