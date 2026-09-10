import pytest

from app.pipeline.decision_engine import DecisionContext, DecisionPolicyConfig, FinalDecisionEngine
from app.pipeline.fallback import abstain_fallback
from app.pipeline.states import FinalState


def context(**overrides):
    values = {
        "attack_score": 0.1,
        "attack_detected": False,
        "global_trust_score": 0.9,
        "suspicious_regions": [],
        "defense_applied": False,
        "defense_method": "none",
        "original_prediction": "dog",
        "original_confidence": 0.9,
        "defended_prediction": "dog",
        "defended_confidence": 0.9,
        "verification_score": 0.9,
        "object_consistency": 0.9,
        "geometry_consistency": 0.9,
        "scene_consistency": 0.9,
    }
    values.update(overrides)
    return DecisionContext(**values)


def test_trusted_requires_independent_evidence():
    result = FinalDecisionEngine().decide(context())
    assert result.final_state is FinalState.TRUSTED
    assert result.final_prediction == "dog"


def test_prediction_alone_cannot_produce_trusted():
    result = FinalDecisionEngine().decide(context(global_trust_score=0.1, verification_score=0.1, object_consistency=0.1, geometry_consistency=0.1, scene_consistency=0.1))
    assert result.final_state is FinalState.ABSTAIN
    assert result.final_prediction is None


def test_defended_requires_attack_defense_and_verification():
    result = FinalDecisionEngine().decide(context(
        attack_score=0.9,
        attack_detected=True,
        defense_applied=True,
        defense_method="purification",
        original_confidence=0.2,
        defended_prediction="cat",
        defended_confidence=0.88,
        defended_stability=1.0,
        verification_score=0.86,
        global_trust_score=0.55,
    ))
    assert result.final_state is FinalState.DEFENDED
    assert result.final_prediction == "cat"


def test_unchanged_label_after_defense_cannot_be_defended():
    """A defense that changes nothing must not certify the suspicious label."""
    result = FinalDecisionEngine().decide(context(
        attack_score=0.9,
        attack_detected=True,
        defense_applied=True,
        defense_method="purification",
        original_prediction="gar",
        defended_prediction="gar",
        defended_confidence=0.99,
        defended_stability=1.0,
        verification_score=0.95,
    ))
    assert result.final_state is FinalState.ABSTAIN
    assert any("did not change" in reason for reason in result.decision_reasons)


def test_unstable_recovery_cannot_be_defended():
    """A changed label that still flips under benign probes is not a recovery."""
    result = FinalDecisionEngine().decide(context(
        attack_score=0.9,
        attack_detected=True,
        defense_applied=True,
        defense_method="purification",
        original_prediction="gar",
        defended_prediction="cat",
        defended_stability=0.0,
        verification_score=0.95,
    ))
    assert result.final_state is FinalState.ABSTAIN
    assert any("unstable" in reason for reason in result.decision_reasons)


def test_attack_detection_alone_cannot_produce_defended():
    result = FinalDecisionEngine().decide(context(attack_score=0.9, attack_detected=True, defense_applied=False, verification_score=0.9))
    assert result.final_state is FinalState.ABSTAIN
    assert result.final_prediction is None


@pytest.mark.parametrize("overrides", [
    {"attack_score": 0.9, "attack_detected": True, "defense_applied": True, "verification_score": 0.2, "defended_confidence": 0.2},
    {"attack_score": 0.9, "attack_detected": True, "defense_applied": True, "verification_score": 0.9, "defended_confidence": 0.2},
])
def test_failed_defense_abstains(overrides):
    assert FinalDecisionEngine().decide(context(**overrides)).final_state is FinalState.ABSTAIN


def test_thresholds_are_configurable_and_fallback_is_no_action():
    engine = FinalDecisionEngine(DecisionPolicyConfig(attack_threshold=0.2, high_trust_threshold=0.8))
    assert engine.config.attack_threshold == 0.2
    fallback = abstain_fallback()
    assert fallback.action == "NO_ACTION"
