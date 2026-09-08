import numpy as np
import pytest

from app.defense import get_defense
from app.defense.evaluation import evaluate_defense
from app.defense.orchestrator import DefenseOrchestrator
from app.defense.policy import DefensePolicy, DefensePolicyConfig


def image() -> np.ndarray:
    value = np.zeros((32, 32, 3), dtype=np.uint8)
    value[:, :, 1] = 220
    return value


def test_transform_preserves_dimensions_and_valid_range():
    result = get_defense("transform").defend(image(), {"parameters": {"method": "gaussian_blur", "kernel_size": 3}})
    assert result.defended_image.shape == image().shape
    assert result.defended_image.dtype == np.uint8
    assert result.parameters["method"] == "gaussian_blur"


def test_mask_modifies_suspicious_region_and_handles_empty_regions():
    source = image()
    regions = [{"x": 10, "y": 10, "width": 8, "height": 8, "area_pixels": 64}]
    result = get_defense("mask").defend(source, {"suspicious_regions": regions, "parameters": {"mask_padding": 0}})
    assert result.defended_image.shape == source.shape
    assert result.defense_applied is True
    empty = get_defense("mask").defend(source, {"suspicious_regions": []})
    assert empty.defense_applied is False
    assert np.array_equal(empty.defended_image, source)


def test_purification_executes():
    result = get_defense("purification").defend(image(), {"parameters": {"strength": 1}})
    assert result.defense_applied is True
    assert result.defended_image.shape == image().shape


def test_certified_is_explicitly_inactive():
    result = get_defense("certified").defend(image())
    assert result.defense_applied is False
    assert result.metadata["status"] == "NOT_IMPLEMENTED"


def test_policy_selects_expected_strategies():
    policy = DefensePolicy()
    assert policy.decide(0.1, None, {}, 0.9, []).selected_defense == "none"
    assert policy.decide(0.45, None, {}, 0.6, []).selected_defense == "transform"
    assert policy.decide(0.8, None, {}, 0.2, [{"x": 1}], 5).selected_defense == "mask"
    assert policy.decide(0.8, None, {}, 0.2, [], 50).selected_defense == "purification"
    assert policy.decide(0.4, "unknown", {}, 0.5, []).selected_defense == "transform"


def test_policy_thresholds_are_configurable():
    policy = DefensePolicy(DefensePolicyConfig(no_defense_threshold=0.1, transform_threshold=0.2, extreme_threshold=0.9))
    assert policy.decide(0.15, None, {}, 0.5, []).selected_defense == "transform"
    with pytest.raises(ValueError):
        DefensePolicyConfig(no_defense_threshold=0.8, transform_threshold=0.2)


def test_orchestrator_returns_trace_and_result():
    result = DefenseOrchestrator().execute(
        image(),
        {"attack_score": 0.4, "global_trust_score": 0.6, "suspicious_regions": [], "suspicious_area_percentage": 0},
    )
    assert result.decision.selected_defense == "transform"
    assert result.defense.defended_image.shape == image().shape
    assert result.trace.success is True
    assert result.trace.attack_score == 0.4


def test_defense_evaluation_metrics():
    source = image()
    sample = {
        "clean_prediction": {"class_id": 1, "confidence": 0.9},
        "adversarial_prediction": {"class_id": 2, "confidence": 0.7},
        "defended_prediction": {"class_id": 1, "confidence": 0.8},
        "original_image": source,
        "defended_image": source.copy(),
    }
    metrics = evaluate_defense([sample])
    assert metrics.attack_success_rate == 1
    assert metrics.defense_recovery_rate == 1
    assert metrics.mean_absolute_error == 0
