"""Deterministic stratified dataset splitting for the robust tier.

Splitting happens on the clean image pool once; attacked variants inherit the
split of their source image so no image (or any of its attacked variants)
appears in more than one split.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
from pathlib import Path
from typing import Any

from app.learning.classes import CLASS_NAMES, WNID_TO_NAME

SPLIT_NAMES = ("train", "val", "test")
WNID_PATTERN = re.compile(r"^n\d{8}$")
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".webp"}


def md5_of_file(path: Path) -> str:
    """Stream an md5 of file contents (cheap enough for ~4k files)."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_class_dirs(source_root: Path) -> list[Path]:
    candidates = sorted(p for p in source_root.iterdir() if p.is_dir())
    wnid_dirs = [p for p in candidates if WNID_PATTERN.match(p.name)]
    if wnid_dirs:
        return wnid_dirs
    friendly = [p for p in candidates if p.name in set(CLASS_NAMES)]
    if friendly:
        return friendly
    raise ValueError(f"No class directories found under {source_root}")


def _collect_images(class_dir: Path) -> list[Path]:
    return sorted(
        path for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def build_split_manifest(
    source_root: str | Path,
    output_root: str | Path,
    per_class: dict[str, int] | None = None,
    seed: int = 7,
    class_names: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Assign every clean image to exactly one split; return the manifest dict.

    Symlinks into split directories are created under ``output_root`` so no
    image is duplicated on disk. ``class_names`` overrides the pinned class
    table (wnid -> name) for tests and derived datasets; production must use
    the default.
    """
    source_root = Path(source_root)
    output_root = Path(output_root)
    wnid_to_name = dict(class_names) if class_names is not None else dict(WNID_TO_NAME)
    caps = per_class or {"train": 250, "val": 75, "test": 75}
    if set(caps) != set(SPLIT_NAMES):
        raise ValueError(f"per_class must define exactly {SPLIT_NAMES}")

    class_dirs = _iter_class_dirs(source_root)
    unknown = {name for name in (p.name for p in class_dirs) if name not in wnid_to_name}
    if unknown:
        raise ValueError(f"Unknown classes present: {sorted(unknown)}")

    assignments: list[dict[str, Any]] = []
    per_split_per_class: dict[str, dict[str, int]] = {split: {} for split in SPLIT_NAMES}
    for class_dir in class_dirs:
        wnid = class_dir.name
        class_name = wnid_to_name[wnid]
        images = _collect_images(class_dir)
        if len(images) < sum(caps.values()):
            raise ValueError(
                f"Class {wnid} has {len(images)} images; need at least {sum(caps.values())}"
            )
        rng = random.Random(f"{seed}-{wnid}")
        rng.shuffle(images)
        boundaries = {"train": caps["train"], "val": caps["val"], "test": caps["test"]}
        offset = 0
        for split in SPLIT_NAMES:
            count = boundaries[split]
            chosen = images[offset : offset + count]
            offset += count
            per_split_per_class[split].setdefault(class_name, 0)
            per_split_per_class[split][class_name] += len(chosen)
            for image_path in chosen:
                assignments.append({
                    "class_wnid": wnid,
                    "class_name": class_name,
                    "split": split,
                    "source_path": str(image_path),
                    "md5": md5_of_file(image_path),
                })

    # Deterministic global ordering.
    split_rank = {name: index for index, name in enumerate(SPLIT_NAMES)}
    assignments.sort(key=lambda entry: (split_rank[entry["split"]], entry["class_wnid"], entry["source_path"]))

    manifest: dict[str, Any] = {
        "schema": "argus.dataset_splits",
        "version": 1,
        "seed": seed,
        "per_class_cap": caps,
        "source_root": str(source_root),
        "num_classes": len(class_dirs),
        "counts": {
            split: {
                "total": sum(per_split_per_class[split].values()),
                "per_class": dict(sorted(per_split_per_class[split].items())),
            }
            for split in SPLIT_NAMES
        },
        "assignments": assignments,
    }

    # Materialize split directories via relative symlinks (no image copies).
    materialized_names = sorted(wnid_to_name[wnid] for wnid in {entry["class_wnid"] for entry in assignments})
    for split in SPLIT_NAMES:
        for class_name in materialized_names:
            (output_root / split / class_name).mkdir(parents=True, exist_ok=True)
    for entry in assignments:
        target_dir = output_root / entry["split"] / entry["class_name"]
        link_path = target_dir / Path(entry["source_path"]).name
        source = Path(entry["source_path"])
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        try:
            link_path.symlink_to(relative_path_between(link_path, source))
        except OSError:
            # Windows without developer mode may refuse symlinks; copy instead.
            shutil.copyfile(source, link_path)

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def relative_path_between(link_path: Path, target: Path) -> str:
    """Relative path from a link's directory to a target file."""
    return os.path.relpath(target.resolve(), start=link_path.parent.resolve())


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    """Load a saved splits manifest."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assert_disjoint(manifest: dict[str, Any]) -> None:
    """Raise if any source image is assigned to more than one split."""
    seen: dict[str, str] = {}
    for entry in manifest["assignments"]:
        key = entry["md5"]
        split = entry["split"]
        previous = seen.get(key)
        if previous is not None and previous != split:
            raise ValueError(f"Image md5 {key} appears in splits {previous} and {split}")
        seen[key] = split
