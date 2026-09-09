"""End-to-end robust-training pipeline using clean and adversarial datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.learning.robust_training import RobustVisionTrainer, build_robust_training_manifest


def train_robust_model(
    clean_root: str | Path,
    adversarial_root: str | Path | None,
    train_root: str | Path | None = None,
    val_root: str | Path | None = None,
    output_dir: str | Path = "data/models",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
) -> dict[str, Any]:
    """Run a complete manifest-to-training workflow and return the serializable model summary."""
    manifest_path = Path(output_dir) / "robust_training_manifest.json"
    manifest = build_robust_training_manifest(clean_root, adversarial_root, manifest_path)
    trainer = RobustVisionTrainer({
        "output_dir": str(output_dir),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_root": str(train_root) if train_root is not None else None,
        "val_root": str(val_root) if val_root is not None else None,
        "manifest_root": str(Path(manifest_path).parent),
    })
    summary = trainer.train_from_manifest(manifest)
    (Path(output_dir) / "training_config.json").write_text(json.dumps({
        "clean_root": str(clean_root),
        "adversarial_root": str(adversarial_root) if adversarial_root is not None else None,
        "output_dir": str(output_dir),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }, indent=2), encoding="utf-8")
    return summary
