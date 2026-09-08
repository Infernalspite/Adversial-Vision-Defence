"""Central configuration for reproducible evaluation runs."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Dataset, calibration, attack, and output settings."""

    dataset_root: Path = Path("data/evaluation")
    results_root: Path = Path("data/evaluation/results")
    plots_root: Path = Path("data/evaluation/plots")
    seed: int = 7
    max_samples_per_category: int | None = None
    detector_weights: dict[str, float] | None = None
    detection_threshold: float = 0.70
    attack_parameters: dict[str, dict[str, object]] = field(default_factory=lambda: {
        "fgsm": {"epsilon": 0.01},
        "pgd": {"epsilon": 0.03, "step_size": 0.005, "iterations": 10, "random_start": False},
        "patch": {"patch_size": 0.2, "location": "center", "iterations": 10},
    })

    def ensure_directories(self) -> None:
        for path in (self.dataset_root, self.results_root, self.plots_root):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def categories(self) -> tuple[str, ...]:
        return ("clean", "fgsm", "pgd", "patch")
