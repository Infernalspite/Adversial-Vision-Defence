"""Defense effectiveness metrics separate from detection metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DefenseMetrics:
    """Aggregate attack, recovery, confidence, and distortion measurements."""

    sample_count: int
    attack_success_rate: float
    defense_recovery_rate: float
    average_clean_confidence: float
    average_adversarial_confidence: float
    average_defended_confidence: float
    mean_absolute_error: float
    mean_squared_error: float
    psnr_db: float


def evaluate_defense(samples: list[dict[str, object]]) -> DefenseMetrics:
    """Evaluate records containing clean, adversarial, and defended predictions/images."""
    if not samples:
        return DefenseMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("inf"))
    attack_successes = 0
    recoveries = 0
    mae: list[float] = []
    mse: list[float] = []
    clean_conf = []
    adversarial_conf = []
    defended_conf = []
    for sample in samples:
        clean = sample["clean_prediction"]
        adversarial = sample["adversarial_prediction"]
        defended = sample["defended_prediction"]
        clean_id = int(clean["class_id"])
        attack_succeeded = clean_id != int(adversarial["class_id"])
        attack_successes += int(attack_succeeded)
        recoveries += int(attack_succeeded and clean_id == int(defended["class_id"]))
        clean_conf.append(float(clean["confidence"]))
        adversarial_conf.append(float(adversarial["confidence"]))
        defended_conf.append(float(defended["confidence"]))
        original = np.asarray(sample["original_image"], dtype=np.float32)
        output = np.asarray(sample["defended_image"], dtype=np.float32)
        difference = output - original
        mae.append(float(np.abs(difference).mean() / 255.0))
        mse.append(float(np.mean(difference**2) / (255.0**2)))
    mean_mse = float(np.mean(mse))
    psnr = float("inf") if mean_mse == 0 else float(10 * np.log10(1.0 / mean_mse))
    count = len(samples)
    return DefenseMetrics(
        sample_count=count,
        attack_success_rate=attack_successes / count,
        defense_recovery_rate=recoveries / attack_successes if attack_successes else 0.0,
        average_clean_confidence=float(np.mean(clean_conf)),
        average_adversarial_confidence=float(np.mean(adversarial_conf)),
        average_defended_confidence=float(np.mean(defended_conf)),
        mean_absolute_error=float(np.mean(mae)),
        mean_squared_error=mean_mse,
        psnr_db=psnr,
    )