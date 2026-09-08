"""Ground-truth-gated localization metrics."""

from typing import Any

import numpy as np


def localization_metrics(records: list[dict[str, Any]]) -> dict[str, float | int]:
    """Calculate IoU/precision/recall only for records with valid masks."""
    valid = [record for record in records if record.get("ground_truth_mask") is not None]
    if not valid:
        return {"samples": 0, "average_iou": 0.0, "precision": 0.0, "recall": 0.0, "suspicious_area_percentage": 0.0}
    ious: list[float] = []; precisions: list[float] = []; recalls: list[float] = []; areas: list[float] = []
    for record in valid:
        predicted = np.asarray(record.get("predicted_mask"), dtype=bool)
        truth = np.asarray(record["ground_truth_mask"], dtype=bool)
        if predicted.shape != truth.shape: continue
        intersection = np.logical_and(predicted, truth).sum(); union = np.logical_or(predicted, truth).sum()
        ious.append(float(intersection / union) if union else 1.0)
        precisions.append(float(intersection / predicted.sum()) if predicted.sum() else 0.0)
        recalls.append(float(intersection / truth.sum()) if truth.sum() else 0.0)
        areas.append(float(predicted.mean() * 100))
    return {"samples": len(ious), "average_iou": float(np.mean(ious)) if ious else 0.0, "precision": float(np.mean(precisions)) if precisions else 0.0, "recall": float(np.mean(recalls)) if recalls else 0.0, "suspicious_area_percentage": float(np.mean(areas)) if areas else 0.0}
