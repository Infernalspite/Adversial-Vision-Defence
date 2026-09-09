"""CLI entry point for training a robust vision model from clean/adversarial data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.learning.training_pipeline import train_robust_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a robust object-recognition model from clean and adversarial image folders.")
    parser.add_argument("--clean-root", type=Path, required=True, help="Folder containing clean class directories")
    parser.add_argument("--adversarial-root", type=Path, help="Optional folder containing adversarial class directories")
    parser.add_argument("--train-root", type=Path, default=None)
    parser.add_argument("--val-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "models")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()

    summary = train_robust_model(
        clean_root=args.clean_root,
        adversarial_root=args.adversarial_root,
        train_root=args.train_root,
        val_root=args.val_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    print(summary)


if __name__ == "__main__":
    main()
