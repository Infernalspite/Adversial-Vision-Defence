"""Create provenance manifests and verify source-pool disjointness."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLASS_INFO = {
    "n01440764": (0, "tench"), "n02102040": (1, "English springer"),
    "n02979186": (2, "cassette player"), "n03000684": (3, "chain saw"),
    "n03028079": (4, "church"), "n03394916": (5, "French horn"),
    "n03417042": (6, "garbage truck"), "n03425413": (7, "gas pump"),
    "n03445777": (8, "golf ball"), "n03888257": (9, "parachute"),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def class_info(path: Path) -> tuple[int | None, str | None]:
    match = re.search(r"n\d{8}", path.stem)
    return CLASS_INFO.get(match.group(0), (None, None)) if match else (None, None)


def source_records(source_root: Path) -> list[dict[str, object]]:
    records = []
    for split in ("baseline", "validation", "test"):
        for path in sorted((source_root / split).iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES or path.stem.endswith("_mask"):
                continue
            class_id, class_name = class_info(path)
            records.append({
                "sample_id": path.stem,
                "source_image": path.relative_to(ROOT).as_posix(),
                "source_split": split,
                "class_id": class_id,
                "class_name": class_name,
                "attack_type": None,
                "is_adversarial": False,
            })
    return records


def evaluation_records(evaluation_root: Path) -> list[dict[str, object]]:
    records = []
    for category in ("clean", "fgsm", "pgd", "patch"):
        directory = evaluation_root / category
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES or path.stem.endswith("_mask"):
                continue
            metadata_path = path.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
            records.append({**metadata, "sample_id": metadata.get("sample_id", path.stem), "image": path.relative_to(evaluation_root).as_posix(), "category": category, "is_adversarial": category != "clean"})
    return records


def write_records(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=ROOT / "data" / "sources")
    parser.add_argument("--evaluation-root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "manifests")
    args = parser.parse_args()

    sources = source_records(args.source_root)
    by_split = {split: {record["sample_id"] for record in sources if record["source_split"] == split} for split in ("baseline", "validation", "test")}
    overlaps = {f"{left}:{right}": sorted(by_split[left] & by_split[right]) for left in by_split for right in by_split if left < right}
    if any(overlaps.values()):
        raise SystemExit(f"Source split overlap detected: {overlaps}")
    for split in by_split:
        write_records([record for record in sources if record["source_split"] == split], args.output / f"{split}.jsonl")
    write_records(sources, args.output / "source_manifest.jsonl")
    for evaluation_root in args.evaluation_root:
        write_records(evaluation_records(evaluation_root), args.output / f"{evaluation_root.name}.jsonl")
    (args.output / "audit.json").write_text(json.dumps({"source_counts": {key: len(value) for key, value in by_split.items()}, "source_overlaps": overlaps}, indent=2), encoding="utf-8")
    print(json.dumps({"source_counts": {key: len(value) for key, value in by_split.items()}, "source_overlaps": overlaps}, indent=2))


if __name__ == "__main__":
    main()