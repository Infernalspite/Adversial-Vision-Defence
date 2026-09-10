"""Build the pinned class set and stratified train/val/test splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.learning.classes import write_classes_file  # noqa: E402
from app.learning.splits import assert_disjoint, build_split_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Pin classes and build stratified splits.")
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "imagenette" / "imagenette2-320" / "train")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "robust" / "splits")
    parser.add_argument("--train-cap", type=int, default=250)
    parser.add_argument("--val-cap", type=int, default=75)
    parser.add_argument("--test-cap", type=int, default=75)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    classes_path = write_classes_file(ROOT / "data" / "robust" / "classes.json")
    print(f"Wrote {classes_path}")

    manifest = build_split_manifest(
        source_root=args.source,
        output_root=args.output,
        per_class={"train": args.train_cap, "val": args.val_cap, "test": args.test_cap},
        seed=args.seed,
    )
    assert_disjoint(manifest)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {args.output / 'manifest.json'}")
    for split, counts in manifest["counts"].items():
        print(f"{split}: {counts['total']} images")


if __name__ == "__main__":
    main()
