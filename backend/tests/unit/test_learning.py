import json

import pytest
from PIL import Image

from app.learning.difficulty import DifficultyScorer
from app.learning.experiment_manager import ExperimentManager
from app.learning.failure_analyzer import FailureAnalyzer
from app.learning.hard_negative_mining import HardNegativeMiner
from app.learning.model_validation import ModelValidationGate
from app.learning.mutation import AttackMutator
from app.learning.robust_training import RobustVisionTrainer, build_robust_training_manifest
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


def test_build_robust_training_manifest(tmp_path):
    clean_root = tmp_path / "clean"
    adv_root = tmp_path / "adversarial"
    (clean_root / "tench").mkdir(parents=True)
    (adv_root / "tench").mkdir(parents=True)
    (clean_root / "tench" / "a.png").write_bytes(b"fake")
    (adv_root / "tench" / "b.png").write_bytes(b"fake")
    manifest = build_robust_training_manifest(clean_root, adv_root, tmp_path / "manifest.json")
    assert manifest["label_to_index"]["tench"] == 0
    assert len(manifest["samples"]) == 2
    assert manifest["samples"][0]["label"] == 0
    assert manifest["samples"][0]["is_adversarial"] == 0


def test_robust_vision_trainer_metrics(tmp_path):
    train_root = tmp_path / "train"
    val_root = tmp_path / "val"
    for root in (train_root, val_root):
        (root / "class_a").mkdir(parents=True)
        (root / "class_b").mkdir(parents=True)
        for class_name in ("class_a", "class_b"):
            image = Image.new("RGB", (32, 32), color=(255, 0, 0))
            image.save(root / class_name / f"{class_name}.png")

    trainer = RobustVisionTrainer({"output_dir": str(tmp_path / "models"), "epochs": 1, "batch_size": 1, "learning_rate": 1e-4})
    manifest = build_robust_training_manifest(train_root, None, tmp_path / "manifest.json")
    summary = trainer.train_from_manifest(manifest)
    assert summary["best_val_accuracy"] >= 0.0
    assert "history" in summary
