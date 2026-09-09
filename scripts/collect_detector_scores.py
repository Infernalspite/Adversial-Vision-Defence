"""Collect fresh detector scores for validation calibration only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.detectors import get_detector, supported_detectors
from app.models.model_registry import get_model
from app.utils.image import decode_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "evaluation_validation")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "evaluation_validation" / "detector_scores.jsonl")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    model = get_model("resnet18")
    names = supported_detectors()
    rows = []
    for category in ("clean", "fgsm", "pgd", "patch"):
        paths = sorted(path for path in (args.dataset / category).iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and not path.stem.endswith("_mask"))
        if args.max_samples:
            paths = paths[:args.max_samples]
        for path in paths:
            metadata_path = path.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
            image = decode_image(path.read_bytes())
            scores = {name: get_detector(name).detect(image, {"model": model}).score for name in names}
            rows.append({"sample_id": metadata.get("sample_id", path.stem), "category": category, "source_split": metadata.get("source_split", "validation"), "is_adversarial": category != "clean", "detector_scores": scores})
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(f"Collected {len(rows)} detector records at {args.output}")


if __name__ == "__main__":
    main()
