import numpy as np
import pytest

from app.evaluation.attack_metrics import attack_success_rate
from app.evaluation.defense_metrics import defense_metrics
from app.evaluation.decision_metrics import decision_metrics
from app.evaluation.detection_metrics import detection_metrics
from app.evaluation.image_metrics import image_metrics
from app.evaluation.localization_metrics import localization_metrics
from app.evaluation.performance_metrics import latency_metrics


def base_records():
    return [
        {"is_adversarial": True, "clean_correct": True, "prediction_changed": True, "attack_score": 0.9, "attack_detected": True, "attack_success": True, "defense_recovered": True, "defense_acceptable": True, "clean_confidence": 0.9, "adversarial_confidence": 0.4, "defended_confidence": 0.8, "final_state": "DEFENDED"},
        {"is_adversarial": True, "clean_correct": True, "prediction_changed": False, "attack_score": 0.4, "attack_detected": False, "attack_success": False, "defense_recovered": False, "defense_acceptable": False, "clean_confidence": 0.9, "adversarial_confidence": 0.8, "defended_confidence": 0.8, "final_state": "ABSTAIN"},
        {"is_adversarial": False, "clean_correct": True, "prediction_changed": False, "attack_score": 0.1, "attack_detected": False, "attack_success": False, "defense_recovered": False, "defense_acceptable": True, "clean_confidence": 0.9, "adversarial_confidence": 0.9, "defended_confidence": 0.9, "final_state": "TRUSTED"},
        {"is_adversarial": False, "clean_correct": True, "prediction_changed": False, "attack_score": 0.8, "attack_detected": True, "attack_success": False, "defense_recovered": False, "defense_acceptable": True, "clean_confidence": 0.9, "adversarial_confidence": 0.9, "defended_confidence": 0.9, "final_state": "ABSTAIN"},
    ]


def test_attack_success_excludes_ineligible_samples():
    assert attack_success_rate(base_records()) == pytest.approx(0.5)


def test_detection_metrics_known_confusion_matrix():
    metrics = detection_metrics(base_records())
    assert metrics["tp"] == 1 and metrics["fp"] == 1 and metrics["tn"] == 1 and metrics["fn"] == 1
    assert metrics["tpr"] == pytest.approx(0.5)
    assert metrics["fpr"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)


def test_defense_and_decision_metrics():
    records = base_records()
    assert defense_metrics(records)["defense_recovery_rate"] == pytest.approx(1.0)
    decisions = decision_metrics(records)
    assert decisions["trusted_rate"] == pytest.approx(0.25)
    assert decisions["unsafe_acceptance_rate"] == 0.0
    assert decisions["clean_rejection_rate"] == 0.5


def test_localization_iou():
    truth = np.zeros((4, 4), dtype=np.uint8); truth[1:3, 1:3] = 1
    metrics = localization_metrics([{"ground_truth_mask": truth, "predicted_mask": truth.copy()}])
    assert metrics["average_iou"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_image_and_latency_metrics():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    assert image_metrics(image, image)["mae"] == 0
    latency = latency_metrics([10, 20, 30, 40])
    assert latency["mean_ms"] == 25
    assert latency["median_ms"] == 25
    assert latency["p95_ms"] >= 30
