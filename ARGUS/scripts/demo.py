"""One-click bounded ARGUS-AEGIS integration demo."""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "backend"))
from app.attacks import get_attack
from app.models.model_registry import get_model
from app.pipeline.pipeline import ArgusPipeline, PipelineContext
from app.utils.image import decode_image


def sample_image() -> np.ndarray:
    candidates = sorted((ROOT / "data" / "samples").glob("*.png")) + sorted((ROOT / "data" / "evaluation" / "clean").glob("*.png"))
    if candidates:
        return decode_image(candidates[0].read_bytes())
    image = np.zeros((64, 64, 3), dtype=np.uint8); image[:, :, 1] = 220
    return image


def run_case(name: str, image: np.ndarray, model: object) -> None:
    result = ArgusPipeline().run(PipelineContext(name, image, {"model": model, "mode": "demo", "attack_type": name if name != "clean" else None}))
    verification = result.verification.verification
    print(f"\n{name.upper()}")
    print(f"Attack Score: {result.defense.detection.attack_score:.2f}")
    print(f"Trust Score: {result.defense.trust.trust_map.values.mean():.2f}")
    print(f"Defense: {result.defense.orchestration.decision.selected_defense}")
    print(f"Verification: {verification.verification_score:.2f}")
    print(f"FINAL: {result.decision.final_state.value}")


def main() -> None:
    print("=====================================")
    print("ARGUS-AEGIS DEMO")
    print("=====================================")
    model = get_model("resnet18"); clean = sample_image()
    print(f"\nCLEAN IMAGE\nPrediction: {model.predict(clean).class_name}\nConfidence: {model.predict(clean).confidence:.2f}")
    run_case("clean", clean, model)
    for name, attack_name, parameters in (("fgsm", "fgsm", {"epsilon": 0.01}), ("pgd", "pgd", {"epsilon": 0.03, "step_size": 0.005, "iterations": 3, "random_start": False}), ("patch", "adversarial_patch", {"patch_size": 0.2, "location": "center", "iterations": 3})):
        attack = get_attack(attack_name).generate(clean, model, **parameters)
        run_case(name, attack.adversarial_image, model)
    print("\n=====================================\nDEMO COMPLETE\n=====================================")


if __name__ == "__main__": main()
