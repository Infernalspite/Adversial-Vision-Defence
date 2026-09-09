"""Calibrate detector threshold and interpretable weights on validation results only."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DETECTORS = ("feature_squeezing", "frequency_analysis", "confidence_instability", "saliency_analysis")
LOGISTIC_SCHEMA = "argus.logistic_regression"


def records(results_root: Path) -> list[dict]:
    output = []
    for path in results_root.glob("*_results.jsonl"):
        output.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return output


def metrics(rows: list[dict], threshold: float, weights: tuple[float, ...]) -> dict[str, float]:
    scores = [sum(weight * float(row.get("detector_scores", {}).get(name, 0.0)) for name, weight in zip(DETECTORS, weights)) for row in rows]
    actual = [bool(row.get("is_adversarial")) for row in rows]
    predicted = [score >= threshold for score in scores]
    tp = sum(a and p for a, p in zip(actual, predicted)); fp = sum(not a and p for a, p in zip(actual, predicted))
    tn = sum(not a and not p for a, p in zip(actual, predicted)); fn = sum(a and not p for a, p in zip(actual, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": threshold, "tpr": recall, "fpr": fp / (fp + tn) if fp + tn else 0.0, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def auc(rows: list[dict], weights: tuple[float, ...]) -> float:
    scores = [sum(weight * float(row.get("detector_scores", {}).get(name, 0.0)) for name, weight in zip(DETECTORS, weights)) for row in rows]
    positives = [score for score, row in zip(scores, rows) if row.get("is_adversarial")]
    negatives = [score for score, row in zip(scores, rows) if not row.get("is_adversarial")]
    if not positives or not negatives:
        return 0.0
    return sum(1.0 if positive > negative else 0.5 if positive == negative else 0.0 for positive in positives for negative in negatives) / (len(positives) * len(negatives))


def pr_auc(rows: list[dict], weights: tuple[float, ...]) -> float:
    scored = sorted(((sum(weight * float(row.get("detector_scores", {}).get(name, 0.0)) for name, weight in zip(DETECTORS, weights)), bool(row.get("is_adversarial"))) for row in rows), reverse=True)
    positives = sum(actual for _, actual in scored)
    if not positives:
        return 0.0
    precision_recall = [(1.0, 0.0)]
    true_positive = false_positive = 0
    for _, actual in scored:
        true_positive += int(actual); false_positive += int(not actual)
        precision_recall.append((true_positive / (true_positive + false_positive), true_positive / positives))
    return sum((recall - previous_recall) * precision for (precision, recall), (_, previous_recall) in zip(precision_recall[1:], precision_recall))


def logistic_fit(rows: list[dict], iterations: int = 4000, learning_rate: float = 0.05, l2: float = 0.01) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    features = np.asarray([[float(row["detector_scores"][name]) for name in DETECTORS] for row in rows], dtype=float)
    labels = np.asarray([bool(row["is_adversarial"]) for row in rows], dtype=float)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (features - mean) / scale
    coefficients = np.zeros(len(DETECTORS), dtype=float)
    intercept = 0.0
    for _ in range(iterations):
        logits = np.clip(standardized @ coefficients + intercept, -60.0, 60.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        error = probabilities - labels
        coefficients -= learning_rate * (standardized.T @ error / len(rows) + l2 * coefficients)
        intercept -= learning_rate * float(error.mean())
    return coefficients, intercept, mean, scale


def logistic_metrics(rows: list[dict], scores: np.ndarray, threshold: float) -> dict[str, float]:
    actual = np.asarray([bool(row["is_adversarial"]) for row in rows])
    predicted = scores >= threshold
    tp = int(np.sum(actual & predicted)); fp = int(np.sum(~actual & predicted))
    tn = int(np.sum(~actual & ~predicted)); fn = int(np.sum(actual & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": float(threshold), "tpr": recall, "fpr": fp / (fp + tn) if fp + tn else 0.0, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--results", type=Path, default=ROOT / "data" / "evaluation_validation" / "results"); parser.add_argument("--output", type=Path, default=ROOT / "data" / "evaluation_validation" / "calibration.json"); args = parser.parse_args()
    rows = [row for row in records(args.results) if row.get("source_split", "validation") == "validation"]
    if not rows or not all(row.get("detector_scores") and "is_adversarial" in row for row in rows):
        raise SystemExit("Validation results with detector_scores are required")
    if any(not all(name in row["detector_scores"] for name in DETECTORS) for row in rows):
        raise SystemExit("Validation results must contain all canonical detector scores")
    distributions = {}
    for name in DETECTORS + ("attack_score",):
        values = np.asarray([float(row.get("detector_scores", {}).get(name, row.get(name, 0.0))) for row in rows], dtype=float)
        distributions[name] = {"mean": float(values.mean()), "median": float(np.median(values)), "std": float(values.std()), "min": float(values.min()), "max": float(values.max()), "percentiles": {str(p): float(np.percentile(values, p)) for p in (5, 25, 75, 95)}}
    candidates = []
    weight_values = (0.0, 0.25, 0.5, 0.75, 1.0)
    weight_sets = [weights for weights in product(weight_values, repeat=4) if abs(sum(weights) - 1.0) < 1e-9 and sum(weight > 0 for weight in weights) >= 2]
    for weights in weight_sets:
        for threshold in np.arange(0.05, 1.0, 0.05):
            result = metrics(rows, float(round(threshold, 2)), weights)
            if result["fpr"] <= 0.20:
                candidates.append((result["f1"] - 0.25 * result["fpr"], result, weights))
    _, best, best_weights = max(candidates, key=lambda item: item[0])
    coefficients, intercept, mean, scale = logistic_fit(rows)
    logistic_features = np.asarray([[float(row["detector_scores"][name]) for name in DETECTORS] for row in rows])
    logistic_scores = 1.0 / (1.0 + np.exp(-np.clip((logistic_features - mean) / scale @ coefficients + intercept, -60.0, 60.0)))
    logistic_candidates = [logistic_metrics(rows, logistic_scores, float(threshold)) for threshold in np.unique(np.r_[0.0, logistic_scores, 1.0])]
    eligible_logistic = [item for item in logistic_candidates if item["fpr"] <= 0.20]
    if not eligible_logistic:
        raise SystemExit("Unable to select a logistic threshold under FPR <= 0.20")
    best_logistic = max(eligible_logistic, key=lambda item: (item["f1"] - 0.25 * item["fpr"], item["threshold"]))
    by_category = {category: [row for row in rows if row.get("category") == category] for category in ("clean", "fgsm", "pgd", "patch")}
    category_distributions = {category: {name: {"mean": float(np.mean([float(row.get("detector_scores", {}).get(name, 0.0)) for row in category_rows])) if category_rows else 0.0, "median": float(np.median([float(row.get("detector_scores", {}).get(name, 0.0)) for row in category_rows])) if category_rows else 0.0} for name in DETECTORS} for category, category_rows in by_category.items()}
    output = {"dataset": "validation", "detectors": DETECTORS, "distributions": distributions, "distributions_by_category": category_distributions, "candidate_thresholds": [metrics(rows, float(round(value, 2)), best_weights) for value in np.arange(0.05, 1.0, 0.05)], "selected_threshold": best["threshold"], "selected_weights": dict(zip(DETECTORS, best_weights)), "selected_metrics": {**best, "roc_auc": auc(rows, best_weights), "pr_auc": pr_auc(rows, best_weights)}, "weight_candidates": len(weight_sets), "logistic_regression": {"schema": LOGISTIC_SCHEMA, "version": 1, "feature_order": list(DETECTORS), "mean": mean.tolist(), "scale": scale.tolist(), "coefficients": coefficients.tolist(), "intercept": float(intercept), "threshold": best_logistic["threshold"], "fit_metadata": {"rows": len(rows), "iterations": 4000, "learning_rate": 0.05, "l2": 0.01, "metrics": best_logistic}}, "criterion": "maximum F1 - 0.25*FPR subject to FPR <= 0.20; validation only"}
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["selected_metrics"], indent=2))


if __name__ == "__main__":
    main()