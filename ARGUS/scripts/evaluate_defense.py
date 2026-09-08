"""Run a small defense-recovery metric demonstration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.attacks import get_attack
from app.defense.evaluation import evaluate_defense
from app.models.model_registry import get_model
from app.pipeline.defense_pipeline import ArgusDefensePipeline


def main() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 1] = 220
    model = get_model("resnet18")
    pipeline = ArgusDefensePipeline()
    records = []
    for name, attack in (
        ("fgsm", get_attack("fgsm")),
        ("pgd", get_attack("pgd")),
        ("patch", get_attack("adversarial_patch")),
    ):
        parameters = {"epsilon": 0.01} if name == "fgsm" else {}
        if name == "pgd":
            parameters = {"epsilon": 0.03, "step_size": 0.005, "iterations": 3, "random_start": False}
        if name == "patch":
            parameters = {"patch_size": 0.2, "iterations": 3}
        attack_result = attack.generate(image, model, **parameters)
        defended = pipeline.run(attack_result.adversarial_image, model, patch_mask=attack_result.patch_mask)
        records.append({
            "clean_prediction": {"class_id": attack_result.original_prediction.class_id, "confidence": attack_result.original_prediction.confidence},
            "adversarial_prediction": {"class_id": attack_result.adversarial_prediction.class_id, "confidence": attack_result.adversarial_prediction.confidence},
            "defended_prediction": {"class_id": defended.defended_prediction.class_id, "confidence": defended.defended_prediction.confidence},
            "original_image": attack_result.original_image,
            "defended_image": defended.orchestration.defense.defended_image,
        })
        print(f"{name}: defense={defended.orchestration.decision.selected_defense}")
    metrics = evaluate_defense(records)
    print("\nARGUS-AEGIS Defense Evaluation")
    print(f"Attack Success Rate: {metrics.attack_success_rate:.3f}")
    print(f"Defense Recovery Rate: {metrics.defense_recovery_rate:.3f}")
    print(f"Mean Absolute Error: {metrics.mean_absolute_error:.4f}")
    print(f"Mean Squared Error: {metrics.mean_squared_error:.4f}")
    print(f"PSNR (dB): {metrics.psnr_db:.2f}")


if __name__ == "__main__":
    main()
