"""Calibrate detector threshold and interpretable weights on validation results only."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DETECTORS = ("feature_squeezing", "frequency_analysis", "confidence_instability", "saliency_analysis")


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


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--results", type=Path, default=ROOT / "data" / "evaluation_validation" / "results"); parser.add_argument("--output", type=Path, default=ROOT / "data" / "evaluation_validation" / "calibration.json"); args = parser.parse_args()
    rows = records(args.results)
    if not rows or not all(row.get("detector_scores") for row in rows):
        raise SystemExit("Validation results with detector_scores are required")
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
    by_category = {category: [row for row in rows if row.get("category") == category] for category in ("clean", "fgsm", "pgd", "patch")}
    category_distributions = {category: {name: {"mean": float(np.mean([float(row.get("detector_scores", {}).get(name, 0.0)) for row in category_rows])) if category_rows else 0.0, "median": float(np.median([float(row.get("detector_scores", {}).get(name, 0.0)) for row in category_rows])) if category_rows else 0.0} for name in DETECTORS} for category, category_rows in by_category.items()}
    output = {"dataset": "validation", "detectors": DETECTORS, "distributions": distributions, "distributions_by_category": category_distributions, "candidate_thresholds": [metrics(rows, float(round(value, 2)), best_weights) for value in np.arange(0.05, 1.0, 0.05)], "selected_threshold": best["threshold"], "selected_weights": dict(zip(DETECTORS, best_weights)), "selected_metrics": {**best, "roc_auc": auc(rows, best_weights), "pr_auc": pr_auc(rows, best_weights)}, "weight_candidates": len(weight_sets), "criterion": "maximum F1 - 0.25*FPR subject to FPR <= 0.20; validation only"}
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["selected_metrics"], indent=2))


if __name__ == "__main__":
    main()