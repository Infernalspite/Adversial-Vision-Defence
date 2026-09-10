"""Detector-fusion calibration via constrained grid search.

Implements the calibration spec: threshold grid 0.05-0.95, non-negative
detector weights in 0.25 increments summing to 1, hard constraint
FPR <= 0.20 on clean val images, objective max(F1 - 0.25 * FPR).

Calibration must run ONLY on the val split. The output file follows the
schema that ``DetectionPipeline._load_calibration`` already consumes, so
deployment is just the file appearing at ``data/models/detector_calibration.json``.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.detectors.attack_scorer import DETECTOR_FEATURE_ORDER, valid_logistic_payload

THRESHOLD_GRID = [round(0.05 * step, 2) for step in range(1, 20)]  # 0.05..0.95
WEIGHT_STEPS = (0.0, 0.25, 0.5, 0.75, 1.0)
FPR_CONSTRAINT = 0.20
FPR_PENALTY = 0.25


@dataclass(slots=True)
class DetectionEvaluation:
    """Confusion-derived detection metrics at one operating point."""

    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def fpr(self) -> float:
        denominator = self.fp + self.tn
        return self.fp / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else 0.0

    @property
    def precision(self) -> float:
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    @property
    def objective(self) -> float:
        """F1 penalized by false positives: max(F1 - 0.25 * FPR)."""
        return self.f1 - FPR_PENALTY * self.fpr

    def as_dict(self) -> dict[str, float | int]:
        return {
            "threshold": self.threshold,
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "tpr": self.recall, "fpr": self.fpr,
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
            "objective": self.objective,
        }


def generate_weight_grid() -> list[dict[str, float]]:
    """All non-negative weights in 0.25 increments summing to exactly 1."""
    combos: list[dict[str, float]] = []
    for values in itertools.product(WEIGHT_STEPS, repeat=len(DETECTOR_FEATURE_ORDER)):
        if abs(sum(values) - 1.0) < 1e-9:
            combos.append({name: float(weight) for name, weight in zip(DETECTOR_FEATURE_ORDER, values)})
    return combos


def build_feature_matrix(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Extract (N, 4) detector-score matrix and adversarial labels from records."""
    features: list[list[float]] = []
    labels: list[int] = []
    for record in records:
        scores = record.get("detector_scores") or {}
        if any(name not in scores for name in DETECTOR_FEATURE_ORDER):
            continue
        features.append([float(scores[name]) for name in DETECTOR_FEATURE_ORDER])
        labels.append(1 if record.get("is_adversarial") else 0)
    if not features:
        raise ValueError("No records with complete detector scores")
    return np.asarray(features, dtype=float), np.asarray(labels, dtype=int)


def evaluate_threshold(fused: np.ndarray, labels: np.ndarray, threshold: float) -> DetectionEvaluation:
    detected = fused >= threshold
    adversarial = labels == 1
    return DetectionEvaluation(
        threshold=threshold,
        tp=int((detected & adversarial).sum()),
        fp=int((detected & ~adversarial).sum()),
        tn=int((~detected & ~adversarial).sum()),
        fn=int((~detected & adversarial).sum()),
    )


def fit_logistic(features: np.ndarray, labels: np.ndarray, epochs: int = 800, learning_rate: float = 0.5, l2: float = 1e-3) -> dict[str, Any]:
    """Deterministic standardized logistic regression for the fused score.

    Returns the payload in the ``argus.logistic_regression`` schema consumed
    by ``UnifiedAttackScorer``.
    """
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (features - mean) / scale
    coefficients = np.zeros(features.shape[1], dtype=float)
    intercept = 0.0
    for _ in range(epochs):
        logits = standardized @ coefficients + intercept
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        error = probabilities - labels
        gradient_w = standardized.T @ error / len(labels) + l2 * coefficients
        gradient_b = float(error.mean())
        coefficients -= learning_rate * gradient_w
        intercept -= learning_rate * gradient_b
    return {
        "schema": "argus.logistic_regression",
        "version": 1,
        "feature_order": list(DETECTOR_FEATURE_ORDER),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "intercept": float(intercept),
        "threshold": 0.5,
    }


def logistic_scores(features: np.ndarray, payload: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(payload["mean"], dtype=float)
    scale = np.asarray(payload["scale"], dtype=float)
    coefficients = np.asarray(payload["coefficients"], dtype=float)
    standardized = (features - mean) / scale
    logits = standardized @ coefficients + float(payload["intercept"])
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))


@dataclass(slots=True)
class CalibrationResult:
    """Best constraint-satisfying operating point plus search diagnostics."""

    weights: dict[str, float]
    threshold: float
    evaluation: DetectionEvaluation
    logistic_regression: dict[str, Any] | None
    logistic_evaluation: DetectionEvaluation | None
    candidates_evaluated: int = 0
    constraint_violations: int = 0
    search_space: dict[str, int] = field(default_factory=dict)

    def to_calibration_payload(self) -> dict[str, Any]:
        """Schema consumed by ``DetectionPipeline._load_calibration``."""
        logistic = self.logistic_regression
        threshold = self.threshold
        if logistic is not None:
            logistic = dict(logistic)
            logistic["threshold"] = threshold
            if not valid_logistic_payload(logistic):
                logistic = None
        return {
            "schema": "argus.detector_calibration",
            "version": 1,
            "selected_weights": dict(self.weights),
            "selected_threshold": threshold,
            "logistic_regression": logistic,
            "selected_metrics": self.evaluation.as_dict(),
            "logistic_metrics": self.logistic_evaluation.as_dict() if self.logistic_evaluation else None,
            "search": {
                "objective": "max(f1 - 0.25 * fpr)",
                "fpr_constraint": FPR_CONSTRAINT,
                "candidates_evaluated": self.candidates_evaluated,
                "constraint_violations": self.constraint_violations,
                "search_space": self.search_space,
                "feature_order": list(DETECTOR_FEATURE_ORDER),
            },
        }


def calibrate(
    features: np.ndarray,
    labels: np.ndarray,
    use_logistic: bool = True,
) -> CalibrationResult:
    """Grid search thresholds x weights (and optional logistic fusion)."""
    if len(np.unique(labels)) < 2:
        raise ValueError("Calibration needs both clean and adversarial samples")
    weight_grid = generate_weight_grid()
    best: CalibrationResult | None = None
    candidates = 0
    violations = 0

    def consider(evaluation: DetectionEvaluation, weights: dict[str, float], logistic_payload: dict[str, Any] | None, logistic_eval: DetectionEvaluation | None) -> None:
        nonlocal best, violations
        if evaluation.fpr > FPR_CONSTRAINT + 1e-9:
            violations += 1
            return
        candidate = CalibrationResult(
            weights=weights,
            threshold=evaluation.threshold,
            evaluation=evaluation,
            logistic_regression=logistic_payload,
            logistic_evaluation=logistic_eval,
        )
        if best is None or (candidate.evaluation.objective, -candidate.evaluation.fpr, candidate.evaluation.recall) > (
            best.evaluation.objective, -best.evaluation.fpr, best.evaluation.recall
        ):
            best = candidate

    # Weighted-mean fusion branch.
    matrix = np.asarray(features, dtype=float)
    for weights in weight_grid:
        weight_vector = np.asarray([weights[name] for name in DETECTOR_FEATURE_ORDER], dtype=float)
        fused = matrix @ weight_vector
        for threshold in THRESHOLD_GRID:
            candidates += 1
            consider(evaluate_threshold(fused, labels, threshold), weights, None, None)

    # Logistic fusion branch (weights kept for schema compatibility and
    # per-detector explainability; fusion itself uses the fitted model).
    logistic_payload: dict[str, Any] | None = None
    if use_logistic:
        logistic_payload = fit_logistic(matrix, labels)
        fused_logistic = logistic_scores(matrix, logistic_payload)
        for threshold in THRESHOLD_GRID:
            candidates += 1
            consider(evaluate_threshold(fused_logistic, labels, threshold), generate_weight_grid()[0], logistic_payload, evaluate_threshold(fused_logistic, labels, threshold))

    if best is None:
        raise RuntimeError(
            f"No operating point satisfied FPR <= {FPR_CONSTRAINT:.0%}; "
            "collect more clean val samples or revisit the detectors"
        )
    best.candidates_evaluated = candidates
    best.constraint_violations = violations
    best.search_space = {
        "thresholds": len(THRESHOLD_GRID),
        "weight_combinations": len(weight_grid),
        "logistic_enabled": use_logistic,
    }
    return best


def save_calibration(result: CalibrationResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_calibration_payload(), indent=2), encoding="utf-8")
    return path


def load_calibration(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
