"""Collect detector scores on val clean/adversarial images and calibrate fusion.

Phase 1 collects per-image detector scores for the val split (clean images
plus the attack suite generated against the model being deployed). Phase 2
runs the constrained grid search from ``app.calibration`` and writes
``data/models/detector_calibration.json`` in the schema that
``DetectionPipeline`` consumes.

Calibration never touches the test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.calibration import calibrate  # noqa: E402
from app.detectors import get_detector, supported_detectors  # noqa: E402
from app.detectors.attack_scorer import DETECTOR_FEATURE_ORDER  # noqa: E402
from app.models.model_registry import get_model  # noqa: E402
from app.utils.image import decode_image  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def iter_images(root: Path):
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for image_path in sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
            yield class_dir.name, image_path


def collect_scores(
    model_tag: str,
    splits_root: Path,
    attacks_root: Path,
    output_path: Path,
    max_per_class: int,
) -> list[dict[str, Any]]:
    """Run all four detectors on val clean images plus the attacked set."""
    model = get_model("resnet18" if model_tag == "baseline" else "robust")
    names = supported_detectors()
    rows: list[dict[str, Any]] = []

    def add_row(class_name: str, image_path: Path, is_adversarial: bool, attack_config: str, source_split: str) -> None:
        image = decode_image(image_path.read_bytes())
        scores = {name: get_detector(name).detect(image, {"model": model}).score for name in names}
        rows.append({
            "sample_id": image_path.stem,
            "class_name": class_name,
            "attack_config": attack_config,
            "source_split": source_split,
            "is_adversarial": is_adversarial,
            "detector_scores": scores,
        })

    clean_count: dict[str, int] = {}
    for class_name, image_path in iter_images(splits_root / "val"):
        if clean_count.get(class_name, 0) >= max_per_class:
            continue
        add_row(class_name, image_path, False, "clean", "val")
        clean_count[class_name] = clean_count.get(class_name, 0) + 1
        if sum(clean_count.values()) % 50 == 0:
            print(f"clean: {sum(clean_count.values())} scored", flush=True)

    attack_root = attacks_root / model_tag / "val"
    if not attack_root.exists():
        raise SystemExit(f"Missing attack suite for model tag '{model_tag}' at {attack_root}")
    attacked_count = 0
    for config_dir in sorted(p for p in attack_root.iterdir() if p.is_dir()):
        for class_name, image_path in iter_images(config_dir):
            sidecar = image_path.with_suffix(".json")
            source_split = "val"
            if sidecar.exists():
                source_split = json.loads(sidecar.read_text(encoding="utf-8")).get("source_split", "val")
            add_row(class_name, image_path, True, config_dir.name, source_split)
            attacked_count += 1
            if attacked_count % 50 == 0:
                print(f"attacked: {attacked_count} scored", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(f"Collected {len(rows)} detector records at {output_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate detector fusion on the val split.")
    parser.add_argument("--model-tag", choices=["baseline", "robust"], default="baseline")
    parser.add_argument("--splits-root", type=Path, default=ROOT / "data" / "robust" / "splits")
    parser.add_argument("--attacks-root", type=Path, default=ROOT / "data" / "robust" / "attacks")
    parser.add_argument("--scores", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "models" / "detector_calibration.json")
    parser.add_argument("--max-per-class", type=int, default=25, help="Clean val images per class for the FPR side")
    parser.add_argument("--reuse-scores", action="store_true", help="Skip collection and reuse the saved scores file")
    args = parser.parse_args()

    scores_path = args.scores or (ROOT / "data" / "robust" / "calibration" / f"detector_scores_{args.model_tag}.jsonl")
    if args.reuse_scores and scores_path.exists():
        rows = [json.loads(line) for line in scores_path.read_text(encoding="utf-8").splitlines() if line]
        print(f"Reused {len(rows)} records from {scores_path}")
    else:
        rows = collect_scores(args.model_tag, args.splits_root, args.attacks_root, scores_path, args.max_per_class)

    clean_rows = [row for row in rows if not row["is_adversarial"]]
    attack_rows = [row for row in rows if row["is_adversarial"]]
    print(f"clean={len(clean_rows)} attacked={len(attack_rows)}")

    features = np.asarray(
        [[float(row["detector_scores"][name]) for name in DETECTOR_FEATURE_ORDER] for row in rows], dtype=float
    )
    labels = np.asarray([1 if row["is_adversarial"] else 0 for row in rows], dtype=int)
    result = calibrate(features, labels, use_logistic=True)
    payload = result.to_calibration_payload()
    payload["dataset"] = {
        "model_tag": args.model_tag,
        "clean": len(clean_rows),
        "attacked": len(attack_rows),
        "attack_configs": sorted({row["attack_config"] for row in attack_rows}),
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output}")
    print(json.dumps(payload["selected_metrics"], indent=2))


if __name__ == "__main__":
    main()
