import numpy as np

from app.calibration import (
    calibrate,
    evaluate_threshold,
    generate_weight_grid,
    logistic_scores,
    save_calibration,
)
from app.detectors.attack_scorer import DETECTOR_FEATURE_ORDER, valid_logistic_payload

NAMES = list(DETECTOR_FEATURE_ORDER)


def make_records(n_clean: int, n_adv: int, saliency_inverted: bool = False):
    generator = np.random.default_rng(11)
    records = []
    for _ in range(n_clean):
        scores = {
            "feature_squeezing": float(generator.uniform(0.0, 0.35)),
            "frequency_analysis": float(generator.uniform(0.0, 0.4)),
            "confidence_instability": float(generator.uniform(0.0, 0.35)),
            "saliency_analysis": float(generator.uniform(0.6, 1.0)) if saliency_inverted else float(generator.uniform(0.0, 0.5)),
        }
        records.append({"is_adversarial": False, "detector_scores": scores})
    for _ in range(n_adv):
        scores = {
            "feature_squeezing": float(generator.uniform(0.65, 1.0)),
            "frequency_analysis": float(generator.uniform(0.5, 0.95)),
            "confidence_instability": float(generator.uniform(0.6, 1.0)),
            "saliency_analysis": float(generator.uniform(0.0, 0.4)) if saliency_inverted else float(generator.uniform(0.6, 1.0)),
        }
        records.append({"is_adversarial": True, "detector_scores": scores})
    return records


def test_weight_grid_is_complete_and_normalized():
    grid = generate_weight_grid()
    assert len(grid) == 35  # compositions of 1.0 in quarters over 4 detectors
    for weights in grid:
        assert abs(sum(weights.values()) - 1.0) < 1e-9
        assert all(weight >= 0.0 for weight in weights.values())
        assert set(weights) == set(NAMES)


def test_evaluate_threshold_confusion_counts():
    fused = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 0, 1, 0])
    evaluation = evaluate_threshold(fused, labels, 0.5)
    assert (evaluation.tp, evaluation.fp, evaluation.tn, evaluation.fn) == (1, 1, 1, 1)
    assert evaluation.fpr == 0.5 and evaluation.recall == 0.5


def test_calibrate_respects_fpr_constraint_and_maximizes_objective():
    records = make_records(n_clean=60, n_adv=40)
    features = np.asarray([[record["detector_scores"][name] for name in NAMES] for record in records])
    labels = np.asarray([1 if record["is_adversarial"] else 0 for record in records])
    result = calibrate(features, labels, use_logistic=True)
    assert result.evaluation.fpr <= 0.20 + 1e-9
    assert result.evaluation.recall >= 0.9  # separable data must be caught
    assert result.candidates_evaluated > 0
    payload = result.to_calibration_payload()
    assert payload["selected_weights"]
    assert 0.05 <= payload["selected_threshold"] <= 0.95
    # Logistic fusion runs but may legitimately lose to weighted fusion;
    # when present it must match the deployment schema exactly.
    assert payload["logistic_regression"] is None or valid_logistic_payload(payload["logistic_regression"])


def test_golden_saliency_inverted_loses_fusion_weight_but_stays_reported():
    """Saliency that is anti-correlated must not win fused-score weight.

    The trust/localization stage keeps the saliency map regardless; this
    test pins the score-fusion side of that contract.
    """
    records = make_records(n_clean=60, n_adv=40, saliency_inverted=True)
    features = np.asarray([[record["detector_scores"][name] for name in NAMES] for record in records])
    labels = np.asarray([1 if record["is_adversarial"] else 0 for record in records])
    result = calibrate(features, labels, use_logistic=False)
    assert result.weights["saliency_analysis"] == 0.0
    # The per-detector evidence is retained even at weight zero.
    assert "saliency_analysis" in result.to_calibration_payload()["selected_weights"]
    assert result.evaluation.f1 >= 0.9


def test_calibrate_requires_both_classes():
    features = np.zeros((5, 4))
    labels = np.ones(5, dtype=int)
    try:
        calibrate(features, labels)
    except ValueError as error:
        assert "both" in str(error)
    else:
        raise AssertionError("Single-class calibration must fail")


def test_save_calibration_round_trip(tmp_path):
    records = make_records(n_clean=30, n_adv=30)
    features = np.asarray([[record["detector_scores"][name] for name in NAMES] for record in records])
    labels = np.asarray([1 if record["is_adversarial"] else 0 for record in records])
    result = calibrate(features, labels)
    path = save_calibration(result, tmp_path / "calibration" / "detector_calibration.json")
    from app.calibration import load_calibration

    payload = load_calibration(path)
    assert payload["schema"] == "argus.detector_calibration"
    assert abs(sum(payload["selected_weights"].values()) - 1.0) < 1e-9
    # When logistic fusion won, scores derived from the saved payload must
    # be valid probabilities; otherwise the payload carries no logistic model.
    if payload["logistic_regression"] is not None:
        scores = logistic_scores(features, payload["logistic_regression"])
        assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0
