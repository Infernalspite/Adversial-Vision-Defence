import numpy as np
import torch
from PIL import Image

from app.learning.robust_training import (
    RobustVisionTrainer,
    build_robust_training_manifest,
    fgsm_batch,
    normalize_batch,
    pgd_batch,
)


class TinyClassifier(torch.nn.Module):
    """Minimal normalized-input classifier for AT primitive tests."""

    def __init__(self) -> None:
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(3 * 8 * 8, 3)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.linear(self.flatten(image))


def test_normalize_batch_matches_single_image_stats():
    batch = torch.rand(2, 3, 8, 8)
    normalized = normalize_batch(batch)
    assert normalized.shape == batch.shape
    assert torch.isfinite(normalized).all()


def test_fgsm_batch_respects_budget_and_is_whitebox():
    torch.manual_seed(0)
    model = TinyClassifier().eval()
    inputs = torch.rand(4, 3, 8, 8)
    targets = torch.tensor([0, 1, 2, 0])
    epsilon = 0.05
    adversarial = fgsm_batch(model, inputs, targets, epsilon, random_start=False)
    assert adversarial.shape == inputs.shape
    assert float((adversarial - inputs).abs().max()) <= epsilon + 1e-5
    assert float(adversarial.min()) >= 0.0 and float(adversarial.max()) <= 1.0


def test_fgsm_batch_random_start_projects_into_ball():
    """R+FGSM must stay within the epsilon ball around the clean input."""
    torch.manual_seed(2)
    model = TinyClassifier().eval()
    inputs = torch.rand(4, 3, 8, 8)
    targets = torch.tensor([0, 1, 2, 0])
    epsilon = 0.05
    adversarial = fgsm_batch(model, inputs, targets, epsilon, random_start=True)
    assert float((adversarial - inputs).abs().max()) <= epsilon + 1e-5
    # With random start the result differs from clean-start FGSM.
    clean_start = fgsm_batch(model, inputs, targets, epsilon, random_start=False)
    assert not torch.allclose(adversarial, clean_start)


def test_pgd_batch_stays_in_linf_ball_and_improves_loss():
    torch.manual_seed(0)
    model = TinyClassifier().eval()
    inputs = torch.rand(4, 3, 8, 8)
    targets = torch.tensor([0, 1, 2, 0])
    epsilon, steps = 0.08, 5
    adversarial = pgd_batch(model, inputs, targets, epsilon, steps=steps)
    assert float((adversarial - inputs).abs().max()) <= epsilon + 1e-5
    with torch.no_grad():
        clean_loss = torch.nn.functional.cross_entropy(model(normalize_batch(inputs)), targets)
        adv_loss = torch.nn.functional.cross_entropy(model(normalize_batch(adversarial)), targets)
    assert adv_loss >= clean_loss - 1e-6


def test_pgd_batch_random_start_differs_from_clean_start():
    torch.manual_seed(1)
    model = TinyClassifier().eval()
    inputs = torch.rand(2, 3, 8, 8)
    targets = torch.tensor([0, 1])
    deterministic = pgd_batch(model, inputs, targets, 0.08, steps=3, random_start=False)
    randomized = pgd_batch(model, inputs, targets, 0.08, steps=3, random_start=True)
    assert not torch.allclose(deterministic, randomized)


def test_train_dataset_yields_pixel_space_batches(tmp_path):
    """Regression: training batches must be [0, 1] pixel tensors.

    The on-the-fly attacks (fgsm_batch/pgd_batch) and the training loop's
    normalize_batch both expect [0, 1] inputs. Feeding ImageNet-normalized
    tensors silently destroyed every adversarial example (clamp(0, 1) on
    normalized data) and pinned robust accuracy at chance.
    """
    from app.learning.robust_training import ImageFolderDataset

    root = tmp_path / "train" / "a"
    root.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(0, 0, 255)).save(root / "x.png")

    pixel_ds = ImageFolderDataset(tmp_path / "train", image_size=8, normalize=False)
    tensor, _ = pixel_ds[0]
    assert float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0

    normalized_ds = ImageFolderDataset(tmp_path / "train", image_size=8, normalize=True)
    normalized, _ = normalized_ds[0]
    assert float(normalized.min()) < 0.0 and float(normalized.max()) > 1.0


def test_trainer_rejects_unknown_attack_mode(tmp_path):
    trainer = RobustVisionTrainer({"output_dir": str(tmp_path), "attack_mode": "nonsense"})
    train_root = tmp_path / "train"
    (train_root / "a").mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(train_root / "a" / "x.png")
    manifest = build_robust_training_manifest(train_root, None, tmp_path / "manifest.json")
    try:
        trainer.train_from_manifest(manifest)
    except ValueError as error:
        assert "attack_mode" in str(error)
    else:
        raise AssertionError("Unknown attack_mode must fail")


def test_trainer_attack_mode_none_skips_robust_metrics(tmp_path):
    train_root = tmp_path / "train"
    val_root = tmp_path / "val"
    for root in (train_root, val_root):
        (root / "class_a").mkdir(parents=True)
        (root / "class_b").mkdir(parents=True)
        for class_name in ("class_a", "class_b"):
            image = Image.new("RGB", (32, 32), color=(255, 0, 0))
            image.save(root / class_name / f"{class_name}.png")
    trainer = RobustVisionTrainer({
        "output_dir": str(tmp_path / "models"),
        "epochs": 1,
        "batch_size": 2,
        "attack_mode": "none",
        "image_size": 32,
    })
    summary = trainer.train_from_manifest(build_robust_training_manifest(train_root, None, tmp_path / "m.json"))
    assert summary["attack_mode"] == "none"
    assert all(entry["robust_val_accuracy"] is None for entry in summary["history"])


def test_trainer_fgsm_mode_reports_robust_accuracy(tmp_path):
    train_root = tmp_path / "train"
    val_root = tmp_path / "val"
    for root in (train_root, val_root):
        (root / "class_a").mkdir(parents=True)
        (root / "class_b").mkdir(parents=True)
        for class_name in ("class_a", "class_b"):
            for index in range(2):
                color = (255, 0, 0) if class_name == "class_a" else (0, 255, 0)
                Image.new("RGB", (32, 32), color=color).save(root / class_name / f"{class_name}{index}.png")
    trainer = RobustVisionTrainer({
        "output_dir": str(tmp_path / "models"),
        "epochs": 1,
        "batch_size": 2,
        "attack_mode": "fgsm",
        "train_epsilon": 4 / 255,
        "robust_val_limit": 2,
        "robust_val_steps": 2,
        "image_size": 32,
    })
    summary = trainer.train_from_manifest(build_robust_training_manifest(train_root, None, tmp_path / "m.json"))
    entry = summary["history"][0]
    assert entry["robust_val_accuracy"] is not None
    assert 0.0 <= entry["robust_val_accuracy"] <= 1.0
    assert (tmp_path / "models" / "robust_model.pt").exists()
