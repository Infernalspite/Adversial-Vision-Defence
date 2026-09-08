"""Thresholded and score-based detection metrics."""

from typing import Any


def confusion(records: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    """Return true-positive, false-positive, true-negative, false-negative counts."""
    tp = fp = tn = fn = 0
    for record in records:
        actual = bool(record.get("is_adversarial"))
        predicted = bool(record.get("attack_detected"))
        if actual and predicted: tp += 1
        elif actual: fn += 1
        elif predicted: fp += 1
        else: tn += 1
    return tp, fp, tn, fn


def detection_metrics(records: list[dict[str, Any]]) -> dict[str, float | int]:
    """Calculate TPR, FPR, precision, recall, F1, and ROC-AUC."""
    tp, fp, tn, fn = confusion(records)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "tpr": recall, "recall": recall, "fpr": fp / (fp + tn) if fp + tn else 0.0, "precision": precision, "f1": f1, "roc_auc": roc_auc(records)}


def roc_auc(records: list[dict[str, Any]]) -> float:
    """Compute rank-based ROC-AUC from raw attack scores."""
    positives = [float(record.get("attack_score", 0.0)) for record in records if record.get("is_adversarial")]
    negatives = [float(record.get("attack_score", 0.0)) for record in records if not record.get("is_adversarial")]
    if not positives or not negatives:
        return 0.0
    wins = sum(1.0 if positive > negative else 0.5 if positive == negative else 0.0 for positive in positives for negative in negatives)
    return wins / (len(positives) * len(negatives))
