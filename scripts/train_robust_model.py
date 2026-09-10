"""CLI entry point for training a robust vision model with adversarial training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.learning.training_pipeline import train_robust_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a robust object-recognition model with clean + on-the-fly adversarial examples.")
    parser.add_argument("--clean-root", type=Path, required=True, help="Folder containing clean class directories (train split)")
    parser.add_argument("--val-root", type=Path, required=True, help="Folder containing val-split class directories")
    parser.add_argument("--adversarial-root", type=Path, default=None, help="Optional folder containing adversarial class directories")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "models")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--attack-mode", choices=["fgsm", "pgd2", "none"], default="fgsm")
    parser.add_argument("--train-epsilon", type=float, default=8 / 255)
    parser.add_argument("--adv-mix", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-per-class-limit", type=int, default=0, help="Cap training images per class (0 = all)")
    parser.add_argument("--robust-val-limit", type=int, default=10, help="Val images per class for PGD robust evaluation")
    parser.add_argument("--robust-val-steps", type=int, default=5, help="PGD steps for robust validation")
    args = parser.parse_args()

    summary = train_robust_model(
        clean_root=args.clean_root,
        adversarial_root=args.adversarial_root,
        train_root=args.clean_root,
        val_root=args.val_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        attack_mode=args.attack_mode,
        train_epsilon=args.train_epsilon,
        adv_mix=args.adv_mix,
        image_size=args.image_size,
        train_per_class_limit=args.train_per_class_limit,
        robust_val_limit=args.robust_val_limit,
        robust_val_steps=args.robust_val_steps,
    )
    print(f"best_val_accuracy={summary['best_val_accuracy']:.4f}")
    print(f"best_robust_val_accuracy={summary.get('best_robust_val_accuracy', 0.0):.4f}")
    print(f"model_path={summary['model_path']}")


if __name__ == "__main__":
    main()
