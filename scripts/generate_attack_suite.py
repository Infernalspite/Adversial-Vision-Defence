"""Generate adversarial datasets from a split-manifest class structure.

Design (per the approved plan):
- Attacks are always generated against the model they will be used to test
  (`--model-tag baseline` for round A, `--model-tag robust` for round B).
- Every attempt is recorded in the run manifest with its success flag, so
  robust accuracy (1 - attack success rate) remains computable.
- Only successful attacks are written as image files: a failed attack is a
  clean-ish image and must not be labeled as adversarial data.
- Zoo attacks (cw_l2, deepfool, mim, square, hopskipjump, autoattack) are
  evaluation-only and must never be fed to detector calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.attacks import get_attack, supported_attacks  # noqa: E402
from app.attacks.zoo import ZOO_ATTACK_NAMES  # noqa: E402
from app.learning.splits import load_split_manifest  # noqa: E402
from app.models.model_registry import model_registry  # noqa: E402
from app.utils.image import decode_image, encode_image  # noqa: E402

NATIVE_ATTACKS = ("fgsm", "pgd", "patch")


def stable_seed(*parts: str) -> int:
    """Deterministic 31-bit seed from arbitrary strings."""
    import hashlib

    digest = hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def config_name(attack: str, epsilon_255: int | None, patch_size: float | None) -> str:
    if attack == "adversarial_patch":
        return f"patch_s{patch_size}"
    if epsilon_255 is not None and attack in {"fgsm", "pgd"}:
        return f"{attack}_e{epsilon_255}"
    return attack


def parse_epsilon(spec: str) -> float:
    """Epsilon given in 255ths (e.g. '8' -> 8/255)."""
    value = float(spec)
    if not 0 < value <= 64:
        raise ValueError("Epsilon must be given in 255ths, e.g. 8 meaning 8/255")
    return value / 255.0


def iter_split_images(splits_root: Path, split: str, per_class_limit: int | None):
    """Yield (class_name, image_path) for a split directory, sorted for determinism."""
    split_root = splits_root / split
    if not split_root.exists():
        raise FileNotFoundError(f"Split directory not found: {split_root}")
    for class_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
        images = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if per_class_limit is not None:
            images = images[:per_class_limit]
        for image_path in images:
            yield class_dir.name, image_path


def attack_kwargs(attack: str, epsilon: float, patch_size: float, seed: int) -> dict[str, Any]:
    if attack == "fgsm":
        return {"epsilon": epsilon}
    if attack == "pgd":
        return {"epsilon": epsilon, "step_size": epsilon / 4.0, "iterations": 10, "random_start": True}
    if attack == "adversarial_patch":
        return {"patch_size": patch_size, "location": "random", "location_seed": seed, "iterations": 20}
    if attack in ZOO_ATTACK_NAMES:
        return {"epsilon": epsilon, "iterations": 30}
    raise ValueError(f"Unsupported attack: {attack}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the adversarial attack suite for a model tag.")
    parser.add_argument("--model-tag", choices=["baseline", "robust"], required=True)
    parser.add_argument("--splits", default="val,test", help="Comma-separated splits to attack")
    parser.add_argument("--attacks", default="fgsm,pgd,patch", help="Comma-separated attack names")
    parser.add_argument("--epsilons", default="2,4,8,16", help="Epsilons in 255ths, comma-separated")
    parser.add_argument("--patch-sizes", default="0.15,0.30")
    parser.add_argument("--max-per-class", type=int, default=None, help="Cap native-attack images per class")
    parser.add_argument("--zoo-max-per-class", type=int, default=2, help="Cap zoo-attack images per class")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "robust" / "attacks")
    parser.add_argument("--splits-root", type=Path, default=ROOT / "data" / "robust" / "splits")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "robust" / "splits" / "manifest.json")
    args = parser.parse_args()

    attacks = [name.strip() for name in args.attacks.split(",") if name.strip()]
    attacks = ["adversarial_patch" if name == "patch" else name for name in attacks]
    unknown = [name for name in attacks if name not in supported_attacks()]
    if unknown:
        raise SystemExit(f"Unknown attacks: {unknown}")
    epsilons = [parse_epsilon(spec) for spec in args.epsilons.split(",")]
    patch_sizes = [float(value) for value in args.patch_sizes.split(",")]
    splits = [split.strip() for split in args.splits.split(",")]

    model = model_registry.get("resnet18" if args.model_tag == "baseline" else "robust")
    manifest = load_split_manifest(args.manifest)
    source_split_by_class_stem = {
        (entry["class_name"], Path(entry["source_path"]).stem): entry["split"]
        for entry in manifest["assignments"]
    }

    run_manifest_path = args.output_root / args.model_tag / "run_manifest.json"
    run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    run_entries: list[dict[str, Any]] = (
        json.loads(run_manifest_path.read_text(encoding="utf-8"))["entries"]
        if run_manifest_path.exists() else []
    )
    seen_keys = {(entry["attack_config"], entry["image"]) for entry in run_entries}

    stats: dict[str, dict[str, int]] = {}
    started_all = time.perf_counter()
    for split in splits:
        for attack in attacks:
            is_zoo = attack in ZOO_ATTACK_NAMES
            per_class_cap = args.zoo_max_per_class if is_zoo else args.max_per_class
            grid = (
                [(None, size) for size in patch_sizes] if attack == "adversarial_patch"
                else [(eps, None) for eps in epsilons] if attack in {"fgsm", "pgd"}
                else [(None, None)]
            )
            for epsilon, patch_size in grid:
                name = config_name(attack, None if epsilon is None else round(epsilon * 255), patch_size)
                target_dir = args.output_root / args.model_tag / split / name
                target_dir.mkdir(parents=True, exist_ok=True)
                attempted = succeeded = 0
                class_counts: dict[str, int] = {}
                for class_name, image_path in iter_split_images(args.splits_root, split, per_class_cap):
                    if per_class_cap is not None and class_counts.get(class_name, 0) >= per_class_cap:
                        continue
                    seed = stable_seed(args.model_tag, split, name, str(image_path))
                    torch.manual_seed(seed)
                    np.random.seed(seed % (2**32))
                    image = decode_image(image_path.read_bytes())
                    kwargs = attack_kwargs(attack, epsilon if epsilon is not None else 0.03, patch_size or 0.2, seed)
                    started = time.perf_counter()
                    try:
                        result = get_attack(attack).generate(image, model, **kwargs)
                    except Exception as error:  # noqa: BLE001 - record and continue
                        print(f"[warn] {name} {image_path.name}: {error}", flush=True)
                        continue
                    attempted += 1
                    entry = {
                        "attack_config": f"{args.model_tag}/{split}/{name}",
                        "attack": attack,
                        "parameters": result.attack_parameters,
                        "model_tag": args.model_tag,
                        "source_split": source_split_by_class_stem.get((class_name, image_path.stem), split),
                        "class_name": class_name,
                        "image": str(image_path),
                        "success": bool(result.attack_success),
                        "clean_prediction": {
                            "class_id": result.original_prediction.class_id,
                            "class_name": result.original_prediction.class_name,
                            "confidence": result.original_prediction.confidence,
                        },
                        "adversarial_prediction": {
                            "class_id": result.adversarial_prediction.class_id,
                            "class_name": result.adversarial_prediction.class_name,
                            "confidence": result.adversarial_prediction.confidence,
                        },
                        "perturbation_l_inf": result.perturbation_magnitude,
                        "generation_time_ms": (time.perf_counter() - started) * 1000,
                        "saved_image": None,
                        "mask": None,
                        "sidecar": None,
                    }
                    if result.attack_success:
                        out_path = target_dir / class_name / f"{image_path.stem}.png"
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        out_path.write_bytes(encode_image(result.adversarial_image, ".png"))
                        entry["saved_image"] = str(out_path.relative_to(args.output_root))
                        sidecar = {
                            "schema": "argus.attack_sample",
                            "version": 1,
                            "attack": attack,
                            "attack_config": name,
                            "model_tag": args.model_tag,
                            "source_split": entry["source_split"],
                            "class_name": class_name,
                            "parameters": result.attack_parameters,
                            "success": True,
                            "clean_prediction": entry["clean_prediction"],
                            "adversarial_prediction": entry["adversarial_prediction"],
                        }
                        sidecar_path = out_path.with_suffix(".json")
                        sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
                        entry["sidecar"] = str(sidecar_path.relative_to(args.output_root))
                        if result.patch_mask is not None:
                            mask_path = target_dir / class_name / f"{image_path.stem}_mask.png"
                            cv2.imwrite(str(mask_path), (result.patch_mask * 255).astype(np.uint8))
                            entry["mask"] = str(mask_path.relative_to(args.output_root))
                        succeeded += 1
                    key = (entry["attack_config"], entry["image"])
                    if key not in seen_keys:
                        run_entries.append(entry)
                        seen_keys.add(key)
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                stats[name] = {"attempted": stats.get(name, {}).get("attempted", 0) + attempted,
                               "succeeded": stats.get(name, {}).get("succeeded", 0) + succeeded}
                rate = succeeded / attempted if attempted else 0.0
                print(f"[{args.model_tag}/{split}] {name}: {succeeded}/{attempted} succeeded ({rate:.1%})", flush=True)

    run_manifest_path.write_text(json.dumps({
        "schema": "argus.attack_run",
        "version": 1,
        "model_tag": args.model_tag,
        "entries": run_entries,
    }, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - started_all
    print(f"Wrote {run_manifest_path} ({len(run_entries)} entries) in {elapsed / 60:.1f} min")
    for name, counts in sorted(stats.items()):
        print(f"TOTAL {name}: {counts['succeeded']}/{counts['attempted']}")


if __name__ == "__main__":
    main()
