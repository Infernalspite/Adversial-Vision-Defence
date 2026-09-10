from app.evaluation.report import render_robustness_report


def sample_summary() -> dict:
    return {
        "meta": {
            "generated": "2026-09-09 12:00 UTC",
            "model": "robust tier model",
            "calibration": "data/models/detector_calibration.json",
            "clean_count": 250,
            "attacked_count": 480,
        },
        "clean": {"clean_accuracy": 0.912, "fpr": 0.04, "abstain_rate": 0.08},
        "robust_accuracy_before_defense": 0.61,
        "robust_accuracy_after_defense": 0.68,
        "detection": {"f1": 0.77, "fpr": 0.05, "recall": 0.72},
        "decisions": {"unsafe_acceptance_rate": 0.12},
        "abstention_precision": 0.81,
        "latency": {"mean_ms": 690.0, "p95_ms": 810.0},
        "by_attack_family": {
            "fgsm": [
                {"strength": "fgsm_e2", "samples": 80, "attack_success_rate": 0.31, "robust_accuracy": 0.69,
                 "detection": {"recall": 0.55}, "defense": {"defense_recovery_rate": 0.2},
                 "decisions": {"unsafe_acceptance_rate": 0.18}},
                {"strength": "fgsm_e16", "samples": 80, "attack_success_rate": 0.9, "robust_accuracy": 0.1,
                 "detection": {"recall": 0.95}, "defense": {"defense_recovery_rate": 0.1},
                 "decisions": {"unsafe_acceptance_rate": 0.02}},
            ],
        },
        "localization": {"samples": 40, "average_iou": 0.42, "precision": 0.51, "recall": 0.66},
        "unseen_attack_families": {
            "deepfool": {"samples": 20, "attack_success_rate": 0.85, "robust_accuracy": 0.15,
                          "detection": {"recall": 0.4}, "defense": {"defense_recovery_rate": 0.1}},
        },
    }


def test_report_includes_headline_and_discipline_sections():
    text = render_robustness_report(sample_summary())
    assert "# ARGUS-AEGIS Robustness Report" in text
    assert "Clean accuracy" in text and "Robust accuracy" in text
    assert "fgsm" in text and "fgsm_e2" in text and "fgsm_e16" in text
    assert "Unseen attack families" in text and "deepfool" in text
    assert "Threat-model limits" in text
    assert "91.2%" in text and "61.0%" in text
    assert "test only" in text


def test_report_handles_missing_sections():
    summary = sample_summary()
    summary["unseen_attack_families"] = {}
    summary["by_attack_family"] = {}
    summary["localization"] = {"samples": 0}
    text = render_robustness_report(summary)
    assert "Threat-model limits" in text
    assert "Per attack family and strength" not in text
