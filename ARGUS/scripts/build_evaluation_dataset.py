"""Build clean/FGSM/PGD/patch evaluation samples without overwriting existing files."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.attacks import get_attack
from app.models.model_registry import get_model
from app.utils.image import decode_image, encode_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "evaluation")
    parser.add_argument("--number", type=int, default=None)
    parser.add_argument("--attack-type", choices=["all", "fgsm", "pgd", "patch"], default="all")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    random.seed(args.seed)
    images = sorted(path for path in args.input.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    random.shuffle(images)
    images = images[: args.number] if args.number else images
    model = get_model("resnet18")
    attacks = {"fgsm": {"epsilon": 0.01}, "pgd": {"epsilon": 0.03, "step_size": 0.005, "iterations": 10, "random_start": False}, "patch": {"patch_size": 0.2, "location": "center", "iterations": 10}}
    for category in ("clean", "fgsm", "pgd", "patch"):
        (args.output / category).mkdir(parents=True, exist_ok=True)
    for source in images:
        image = decode_image(source.read_bytes())
        clean_prediction = model.predict(image)
        clean_name = f"{source.stem}.png"
        clean_path = args.output / "clean" / clean_name
        if not clean_path.exists(): clean_path.write_bytes(encode_image(image))
        clean_metadata = {"image": clean_name, "category": "clean", "source": "evaluation", "ground_truth_class": clean_prediction.class_name, "attack_type": None}
        clean_path.with_suffix(".json").write_text(json.dumps(clean_metadata, indent=2), encoding="utf-8")
        selected = attacks if args.attack_type == "all" else {args.attack_type: attacks[args.attack_type]}
        for attack_name, parameters in selected.items():
            if (args.output / attack_name / clean_name).exists(): continue
            result = get_attack("adversarial_patch" if attack_name == "patch" else attack_name).generate(image, model, **parameters)
            target_path = args.output / attack_name / clean_name
            target_path.write_bytes(encode_image(result.adversarial_image))
            metadata = {"image": clean_name, "category": "adversarial", "attack_type": attack_name, "ground_truth_class": clean_prediction.class_name, "source_image": str(Path("clean") / clean_name), "parameters": parameters}
            target_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            if result.patch_mask is not None:
                cv2.imwrite(str(target_path.with_name(f"{target_path.stem}_mask.png")), result.patch_mask * 255)
    print(f"Built evaluation samples under {args.output}")


if __name__ == "__main__": main()
