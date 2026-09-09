"""Generate adversarial examples from a clean class-structured dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.attacks import get_attack
from app.models.model_registry import get_model
from app.utils.image import decode_image, encode_image


def iter_images(root: Path):
    for image_path in sorted(root.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            yield image_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate adversarial datasets from a clean class directory.")
    parser.add_argument("--input", type=Path, required=True, help="Root directory containing class folders with clean images")
    parser.add_argument("--output", type=Path, required=True, help="Output root (e.g. data/evaluation_validation)")
    parser.add_argument("--attack", choices=["fgsm", "pgd", "adversarial_patch"], default="fgsm")
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--step-size", type=float, default=0.005)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--patch-size", type=float, default=0.2)
    parser.add_argument("--patch-location", type=str, default="center")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    model = get_model("resnet18")
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    target_dir = output_root / args.attack
    target_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for class_dir in sorted(args.input.iterdir()):
        if not class_dir.is_dir():
            continue
        target_class_dir = target_dir / class_dir.name
        target_class_dir.mkdir(parents=True, exist_ok=True)
        for image_path in iter_images(class_dir):
            if args.max_images is not None and count >= args.max_images:
                return
            image = decode_image(image_path.read_bytes())
            attack_kwargs = {
                "epsilon": args.epsilon,
                "step_size": args.step_size,
                "iterations": args.iterations,
                "patch_size": args.patch_size,
                "location": args.patch_location,
                "random_start": False,
            }
            if args.attack == "adversarial_patch":
                attack_kwargs.pop("epsilon", None)
                attack_kwargs.pop("step_size", None)
                attack_kwargs.pop("iterations", None)
                attack_kwargs["iterations"] = args.iterations
            attack = get_attack(args.attack).generate(image, model, **attack_kwargs)
            out_path = target_class_dir / f"{image_path.stem}_{args.attack}.png"
            out_path.write_bytes(encode_image(attack.adversarial_image, ".png"))
            count += 1
            print(f"Saved {out_path}")

    print(f"Generated {count} adversarial images to {target_dir}")


if __name__ == "__main__":
    main()
