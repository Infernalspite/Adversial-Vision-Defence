"""Print detector, perturbation, and localization evidence for paired samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "data" / "evaluation_test" / "results")
    parser.add_argument("--sample-id", required=True)
    args = parser.parse_args()
    matches = []
    for path in sorted(args.results.glob("*_results.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            record = json.loads(line)
            if record.get("sample_id") == args.sample_id or Path(record.get("sample", "")).stem == args.sample_id:
                matches.append(record)
    if not matches:
        raise SystemExit(f"No result found for sample {args.sample_id}")
    for record in matches:
        print(json.dumps({
            "category": record.get("category"),
            "sample_id": record.get("sample_id"),
            "clean_prediction": record.get("clean_prediction"),
            "original_prediction": record.get("original_prediction"),
            "attack_success": record.get("attack_success"),
            "attack_score": record.get("attack_score"),
            "attack_detected": record.get("attack_detected"),
            "detector_scores": record.get("detector_scores"),
            "perturbation_l_inf": record.get("perturbation_l_inf"),
            "perturbation_mean_abs": record.get("perturbation_mean_abs"),
            "changed_pixel_percentage": record.get("changed_pixel_percentage"),
            "patch_mask_area_ratio": record.get("patch_mask_area_ratio"),
            "patch_iou": record.get("patch_iou"),
            "patch_localization_precision": record.get("patch_localization_precision"),
            "patch_localization_recall": record.get("patch_localization_recall"),
        }, indent=2))


if __name__ == "__main__":
    main()
