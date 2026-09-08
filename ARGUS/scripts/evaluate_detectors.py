"""Run a small clean-versus-adversarial detector comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.attacks import get_attack
from app.detectors import get_detector, supported_detectors
from app.detectors.detection_pipeline import DetectionPipeline
from app.detectors.evaluation import evaluate_detector
from app.models.model_registry import get_model


def main() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 1] = 220
    model = get_model("resnet18")
    attacks = {
        "FGSM": get_attack("fgsm").generate(image, model, epsilon=0.01),
        "PGD": get_attack("pgd").generate(image, model, epsilon=0.03, step_size=0.005, iterations=3, random_start=False),
        "Patch": get_attack("adversarial_patch").generate(image, model, patch_size=0.2, iterations=3),
    }
    samples = [("Clean", image)] + [(name, result.adversarial_image) for name, result in attacks.items()]
    context = {"model": model}

    print("ARGUS-AEGIS Detector Evaluation")
    print("================================")
    print("\nDataset:")
    print("Clean: 1")
    print("FGSM: 1")
    print("PGD: 1")
    print("Patch: 1")
    print("\n-------------------------------")
    print("Detector                 Clean FP")
    print("-------------------------------")
    for name in supported_detectors():
        detector = get_detector(name)
        clean = detector.detect(image, context)
        print(f"{name:<25} {int(clean.detected)}")

    pipeline = DetectionPipeline()
    print("\n-------------------------------")
    print("Unified Attack Scorer")
    print("-------------------------------")
    clean_result = pipeline.analyze(image, model)
    print(f"Clean Average Score:       {clean_result.attack_score:.3f}")
    for name, sample in samples[1:]:
        result = pipeline.analyze(sample, model)
        print(f"{name} Average Score:         {result.attack_score:.3f}")
    print(f"\nDetection Threshold:       {clean_result.detection_threshold:.2f}")
    evaluation = evaluate_detector(
        [(image, False)] + [(result.adversarial_image, True) for result in attacks.values()],
        model,
        pipeline,
    )
    print(f"False Positive Rate:       {evaluation.false_positive_rate:.3f}")
    print(f"True Positive Rate:        {evaluation.true_positive_rate:.3f}")
    print(f"False Negative Rate:       {evaluation.false_negative_rate:.3f}")
    print("\nNote: these are demonstration samples, not a calibrated benchmark dataset.")


if __name__ == "__main__":
    main()
