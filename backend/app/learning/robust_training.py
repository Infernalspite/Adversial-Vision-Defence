"""Utilities for building and running a real robust image-classification training workflow."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


@dataclass(slots=True)
class TrainingMetrics:
    """Serializable metrics for a training run."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float | int]] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    timestamp: str = ""


class ImageFolderDataset(Dataset):
    """Simple image dataset for clean/adversarial samples stored by class folder."""

    def __init__(self, root: str | Path, image_size: int = 224, transform: transforms.Compose | None = None) -> None:
        self.root = Path(root)
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.samples: list[tuple[Path, int]] = []
        self.classes = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        self.class_to_index = {name: idx for idx, name in enumerate(self.classes)}

        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                    self.samples.append((image_path, self.class_to_index[class_dir.name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, label


def build_robust_training_manifest(clean_root: str | Path, adversarial_root: str | Path | None, output_path: str | Path) -> dict[str, Any]:
    """Build a manifest that includes clean labels and optional adversarial variants for the same classes."""
    clean_root = Path(clean_root)
    adversarial_root = Path(adversarial_root) if adversarial_root is not None else None
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    class_names: set[str] = set()
    if clean_root.exists():
        class_names.update(p.name for p in clean_root.iterdir() if p.is_dir())
    if adversarial_root is not None and adversarial_root.exists():
        class_names.update(p.name for p in adversarial_root.iterdir() if p.is_dir())

    label_to_index = {name: idx for idx, name in enumerate(sorted(class_names))}
    samples: list[dict[str, Any]] = []

    def add_root(root: Path | None, is_adversarial: int) -> None:
        if root is None or not root.exists():
            return
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            label_idx = label_to_index.get(class_dir.name, len(label_to_index))
            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                    samples.append({
                        "image_path": str(image_path),
                        "label": label_idx,
                        "class_name": class_dir.name,
                        "is_adversarial": is_adversarial,
                    })

    add_root(clean_root, 0)
    add_root(adversarial_root, 1)

    manifest = {
        "label_to_index": label_to_index,
        "num_classes": len(label_to_index),
        "samples": samples,
        "training_goal": "robust_object_recognition_and_adversarial_detection",
    }
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _compute_classification_metrics(y_true: list[int], y_pred: list[int], class_names: list[str]) -> tuple[float, float, dict[str, dict[str, float | int]], list[list[int]]]:
    """Compute accuracy, macro-F1, per-class metrics, and confusion matrix from integer labels."""
    total = len(y_true)
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    accuracy = correct / total if total else 0.0

    confusion = np.zeros((len(class_names), len(class_names)), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        confusion[actual, predicted] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for idx, class_name in enumerate(class_names):
        tp = int(confusion[idx, idx])
        fp = int(confusion[:, idx].sum() - tp)
        fn = int(confusion[idx, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        support = int(confusion[idx, :].sum())
        per_class[class_name] = {
            "support": support,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    return accuracy, macro_f1, per_class, confusion.tolist()


class RobustVisionTrainer:
    """Fine-tune a pretrained ResNet-18 for robust multi-class recognition."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.device = torch.device(self.config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_dir = Path(self.config.get("output_dir", "data/models"))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = int(self.config.get("batch_size", 16))
        self.learning_rate = float(self.config.get("learning_rate", 3e-4))
        self.weight_decay = float(self.config.get("weight_decay", 1e-4))
        self.epochs = int(self.config.get("epochs", 3))
        self.image_size = int(self.config.get("image_size", 224))

    def _build_model(self, num_classes: int) -> nn.Module:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model.to(self.device)

    def train_from_manifest(self, manifest_path: str | Path | dict[str, Any], limit_train: int | None = None, limit_val: int | None = None) -> dict[str, Any]:
        """Train from a manifest object or file and return real metrics, state dict, and metadata."""
        manifest = manifest_path if isinstance(manifest_path, dict) else json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        label_to_index = manifest["label_to_index"]
        class_names = [name for name, _ in sorted(label_to_index.items(), key=lambda item: item[1])]
        samples = manifest["samples"]

        if not samples:
            raise ValueError(f"Manifest {manifest_path} contains no training samples.")

        train_samples = [sample for sample in samples if sample.get("is_adversarial") in (0, 1)]
        if len(train_samples) == 0:
            raise ValueError("No usable samples present in the manifest for training.")

        manifest_root = Path(self.config.get("manifest_root", ".")) if not isinstance(manifest_path, dict) else Path(self.config.get("manifest_root", "."))
        if not isinstance(manifest_path, dict):
            manifest_root = Path(manifest_path).parent

        train_root = Path(self.config.get("train_root", str(manifest_root / "train")))
        val_root = Path(self.config.get("val_root", str(manifest_root / "val")))

        train_ds = ImageFolderDataset(train_root, image_size=self.image_size) if train_root.exists() else _ManifestDataset(samples, self.image_size)
        val_ds = ImageFolderDataset(val_root, image_size=self.image_size) if val_root.exists() else _ManifestDataset(samples, self.image_size, split_seed=42)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

        model = self._build_model(len(class_names))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        best_state = None
        best_val_accuracy = -1.0
        history: list[TrainingMetrics] = []

        for epoch in range(1, self.epochs + 1):
            model.train()
            running_loss = 0.0
            running_correct = 0
            total_seen = 0
            train_predictions: list[int] = []
            train_targets: list[int] = []

            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                logits = model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                predictions = logits.argmax(dim=1)
                running_correct += (predictions == targets).sum().item()
                total_seen += inputs.size(0)
                train_predictions.extend(predictions.cpu().tolist())
                train_targets.extend(targets.cpu().tolist())

            train_accuracy = running_correct / max(total_seen, 1)
            train_loss = running_loss / max(total_seen, 1)

            model.eval()
            val_loss_total = 0.0
            val_correct = 0
            val_total = 0
            val_predictions: list[int] = []
            val_targets: list[int] = []
            with torch.inference_mode():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    logits = model(inputs)
                    loss = criterion(logits, targets)
                    val_loss_total += loss.item() * targets.size(0)
                    predictions = logits.argmax(dim=1)
                    val_correct += (predictions == targets).sum().item()
                    val_total += targets.size(0)
                    val_predictions.extend(predictions.cpu().tolist())
                    val_targets.extend(targets.cpu().tolist())

            val_accuracy = val_correct / max(val_total, 1)
            val_loss = val_loss_total / max(val_total, 1)
            accuracy, macro_f1, per_class, confusion_matrix = _compute_classification_metrics(val_targets, val_predictions, class_names)
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=float(train_loss),
                train_accuracy=float(train_accuracy),
                val_loss=float(val_loss),
                val_accuracy=float(accuracy),
                macro_f1=float(macro_f1),
                per_class=per_class,
                confusion_matrix=confusion_matrix,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            history.append(metrics)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_state = {"model_state": model.state_dict(), "class_names": class_names, "val_accuracy": val_accuracy}

        if best_state is None:
            raise RuntimeError("Training did not produce a valid checkpoint.")

        checkpoint_path = self.model_dir / "robust_model.pt"
        torch.save(best_state, checkpoint_path)
        summary = {
            "model_path": str(checkpoint_path),
            "device": str(self.device),
            "num_classes": len(class_names),
            "class_names": class_names,
            "best_val_accuracy": float(best_val_accuracy),
            "history": [
                {
                    "epoch": entry.epoch,
                    "train_loss": entry.train_loss,
                    "train_accuracy": entry.train_accuracy,
                    "val_loss": entry.val_loss,
                    "val_accuracy": entry.val_accuracy,
                    "macro_f1": entry.macro_f1,
                    "per_class": entry.per_class,
                    "confusion_matrix": entry.confusion_matrix,
                }
                for entry in history
            ],
        }
        (self.model_dir / "robust_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


class _ManifestDataset(Dataset):
    """Minimal dataset implementation for manifest-only training runs."""

    def __init__(self, samples: list[dict[str, Any]], image_size: int = 224, split_seed: int = 7) -> None:
        self.samples = samples
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = self.samples[idx]
        path = Path(item["image_path"])
        image = Image.open(path).convert("RGB")
        return self.transform(image), int(item["label"])
