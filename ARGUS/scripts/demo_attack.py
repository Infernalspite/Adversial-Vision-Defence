"""Run a local ARGUS-AEGIS adversarial attack demonstration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.attacks import get_attack
from app.models.model_registry import get_model
from app.utils.image import decode_image, encode_image


def main() -> None:
    parser = argparse.ArgumentParser(description="ARGUS-AEGIS adversarial attack demo")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--attack", choices=["fgsm", "pgd", "adversarial_patch"], required=True)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--random-start", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--patch-size", type=float, default=None)
    parser.add_argument("--location", default=None)
    args = parser.parse_args()

    image = decode_image(args.image.read_bytes())
    model = get_model("resnet18")
    attack = get_attack(args.attack)
    parameters = {
        key: value
        for key, value in {
            "epsilon": args.epsilon,
            "step_size": args.step_size,
            "iterations": args.iterations,
            "random_start": args.random_start,
            "patch_size": args.patch_size,
            "location": args.location,
        }.items()
        if value is not None
    }
    result = attack.generate(image, model, **parameters)

    output_dir = ROOT / "data" / "adversarial"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.image.stem}_{args.attack}.png"
    output_path.write_bytes(encode_image(result.adversarial_image))

    print("ARGUS-AEGIS Adversarial Attack Demo")
    print()
    print("Original:")
    print(f"Prediction: {result.original_prediction.class_name}")
    print(f"Confidence: {result.original_prediction.confidence:.2f}")
    print()
    print("Attack:")
    print(args.attack.upper())
    for name, value in result.attack_parameters.items():
        print(f"{name}: {value}")
    print()
    print("Adversarial:")
    print(f"Prediction: {result.adversarial_prediction.class_name}")
    print(f"Confidence: {result.adversarial_prediction.confidence:.2f}")
    print()
    print(f"Prediction Changed: {'YES' if result.adversarial_prediction.class_id != result.original_prediction.class_id else 'NO'}")
    print(f"Attack Success: {'YES' if result.attack_success else 'NO'}")
    print(f"Generation Time: {result.generation_time_ms:.2f} ms")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
