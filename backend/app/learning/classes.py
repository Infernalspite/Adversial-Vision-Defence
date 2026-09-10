"""Fixed class set for the robust specialist tier.

This module is the single source of truth for the defended class list.
Every dataset build, attack generation, training run, and evaluation must
resolve classes through it so the list cannot drift between steps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# WordNet IDs are Imagenette's exact 10 classes. Changing this table changes
# the defended scope of the whole system; do it deliberately, then re-run
# splits, training, attack generation, and calibration.
IMAGENETTE_CLASSES: tuple[dict[str, str], ...] = (
    {"wnid": "n01440764", "name": "tench"},
    {"wnid": "n02102040", "name": "English springer spaniel"},
    {"wnid": "n02979186", "name": "cassette player"},
    {"wnid": "n03000684", "name": "chain saw"},
    {"wnid": "n03028079", "name": "church"},
    {"wnid": "n03394916", "name": "French horn"},
    {"wnid": "n03417042", "name": "garbage truck"},
    {"wnid": "n03425413", "name": "gas pump"},
    {"wnid": "n03445777", "name": "golf ball"},
    {"wnid": "n03888257", "name": "parachute"},
)

CLASS_NAMES: tuple[str, ...] = tuple(entry["name"] for entry in IMAGENETTE_CLASSES)
WNIDS: tuple[str, ...] = tuple(entry["wnid"] for entry in IMAGENETTE_CLASSES)
WNID_TO_NAME: dict[str, str] = {entry["wnid"]: entry["name"] for entry in IMAGENETTE_CLASSES}
NAME_TO_WNID: dict[str, str] = {entry["name"]: entry["wnid"] for entry in IMAGENETTE_CLASSES}


def classes_manifest() -> dict[str, Any]:
    """Return a serializable description of the fixed class set."""
    return {
        "schema": "argus.defended_classes",
        "version": 1,
        "num_classes": len(IMAGENETTE_CLASSES),
        "classes": [dict(entry) for entry in IMAGENETTE_CLASSES],
    }


def write_classes_file(output_path: str | Path) -> Path:
    """Persist the pinned class set so artifacts can reference one file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(classes_manifest(), indent=2), encoding="utf-8")
    return path
