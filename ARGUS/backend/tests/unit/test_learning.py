import json

import pytest

from app.learning.difficulty import DifficultyScorer
from app.learning.experiment_manager import ExperimentManager
from app.learning.failure_analyzer import FailureAnalyzer
from app.learning.hard_negative_mining import HardNegativeMiner
from app.learning.model_validation import ModelValidationGate
from app.learning.mutation import AttackMutator
from app.learning.schemas import AttackConfig


def test_hard_negative_mining_categories():
    record = {"request_id": "r1", "stage": "decision", "details": {"final_state": "TRUSTED", "attack_score": 0.9, "verification_score": 0.2}}
    mined = HardNegativeMiner().mine([record])
    assert mined[0].category == "unsafe_acceptance"
    assert "TRUSTED" in mined[0].explanation


def test_attack_config_and_bounded_mutation():
    config = AttackConfig("pgd", epsilon=0.02, iterations=5)
    mutated = AttackMutator(42).mutate(config)
    mutated.validate()
    assert mutated.epsilon <= 0.25
    assert mutated.iterations <= 100
    assert AttackMutator(42).mutate(config) == mutated
    with pytest.raises(ValueError): AttackConfig("bad").validate()


def test_difficulty_scoring_and_validation_gate():
    score = DifficultyScorer().score({"attack_success": True, "adversarial_confidence": 0.8, "attack_score": 0.1, "defense_recovered": False, "verification_score": 0.2})
    assert 0 <= score <= 1
    rejected = ModelValidationGate().validate("candidate", {"unsafe_acceptance_rate": 0.2, "clean_rejection_rate": 0.1, "tpr": 0.8, "defense_recovery_rate": 0.2})
    assert rejected.status == "REJECTED"
    validated = ModelValidationGate().validate("candidate", {"unsafe_acceptance_rate": 0.0, "clean_rejection_rate": 0.1, "tpr": 0.8, "defense_recovery_rate": 0.2})
    assert validated.status == "VALIDATED"


def test_experiment_manager_persists(tmp_path):
    manager = ExperimentManager(tmp_path)
    experiment_id, path = manager.create({"attack_type": "pgd", "generations": 2})
    manager.save_generation(path, 0, [{"difficulty_score": 0.8}])
    manager.save_report(path, {"status": "CANDIDATE"})
    assert manager.get(experiment_id)["report"]["status"] == "CANDIDATE"
    assert len(manager.list()) == 1
