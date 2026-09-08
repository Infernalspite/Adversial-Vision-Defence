"""Build clean/FGSM/PGD/patch evaluation samples without overwriting existing files."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.attacks import get_attack
from app.models.model_registry import get_model
from app.utils.image import decode_image, encode_image


IMAGENETTE_CLASSES = {
    "n01440764": (0, "tench"),
    "n02102040": (1, "English springer"),
    "n02979186": (2, "cassette player"),
    "n03000684": (3, "chain saw"),
    "n03028079": (4, "church"),
    "n03394916": (5, "French horn"),
    "n03417042": (6, "garbage truck"),
    "n03425413": (7, "gas pump"),
    "n03445777": (8, "golf ball"),
    "n03888257": (9, "parachute"),
}


def _class_info(source: Path) -> tuple[int | None, str | None]:
    for token in source.stem.split("_"):
        if token in IMAGENETTE_CLASSES:
            return IMAGENETTE_CLASSES[token]
    return None, None


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "evaluation")
    parser.add_argument("--number", type=int, default=None)
    parser.add_argument("--attack-type", choices=["all", "fgsm", "pgd", "patch"], default="all")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    images = sorted(
        path for path in args.input.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        and not path.stem.endswith("_mask")
    )
    random.shuffle(images)
    images = images[: args.number] if args.number else images
    model = get_model("resnet18")
    attacks = {"fgsm": {"epsilon": 0.01}, "pgd": {"epsilon": 0.03, "step_size": 0.005, "iterations": 10, "random_start": False}, "patch": {"patch_size": 0.2, "location": "center", "iterations": 10}}
    for category in ("clean", "fgsm", "pgd", "patch"):
        (args.output / category).mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    source_split = args.input.name
    for source in images:
        image = decode_image(source.read_bytes())
        clean_prediction = model.predict(image)
        class_id, class_name = _class_info(source)
        clean_name = f"{source.stem}.png"
        clean_path = args.output / "clean" / clean_name
        if not clean_path.exists(): clean_path.write_bytes(encode_image(image))
        clean_metadata = {
            "sample_id": source.stem,
            "image": clean_name,
            "category": "clean",
            "source_image": _relative_path(source),
            "source_split": source_split,
            "class_name": class_name,
            "class_id": class_id,
            "ground_truth_class": class_name,
            "attack_type": None,
            "is_adversarial": False,
            "seed": args.seed,
        }
        clean_path.with_suffix(".json").write_text(json.dumps(clean_metadata, indent=2), encoding="utf-8")
        manifest.append(clean_metadata)
        selected = attacks if args.attack_type == "all" else {args.attack_type: attacks[args.attack_type]}
        for attack_name, parameters in selected.items():
            target_path = args.output / attack_name / clean_name
            if not target_path.exists():
                result = get_attack("adversarial_patch" if attack_name == "patch" else attack_name).generate(image, model, **parameters)
                target_path.write_bytes(encode_image(result.adversarial_image))
                if result.patch_mask is not None:
                    cv2.imwrite(str(target_path.with_name(f"{target_path.stem}_mask.png")), result.patch_mask * 255)
            metadata = {
                "sample_id": source.stem,
                "image": clean_name,
                "category": "adversarial",
                "source_image": _relative_path(source),
                "source_split": source_split,
                "clean_image": str(Path("clean") / clean_name),
                "class_name": class_name,
                "class_id": class_id,
                "ground_truth_class": class_name,
                "attack_type": attack_name,
                "is_adversarial": True,
                "adversarial_image": str(Path(attack_name) / clean_name),
                "mask_path": str(Path(attack_name) / f"{target_path.stem}_mask.png") if attack_name == "patch" else None,
                "generation_parameters": parameters,
                "seed": args.seed,
            }
            target_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            manifest.append(metadata)
    (args.output / "manifest.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in manifest), encoding="utf-8"
    )
    print(f"Built evaluation samples under {args.output}")


if __name__ == "__main__": main()
