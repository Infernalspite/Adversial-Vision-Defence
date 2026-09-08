from pathlib import Path

from app.evaluation.config import EvaluationConfig
from app.evaluation.evaluator import EvaluationRunner


def test_evaluator_handles_tiny_empty_dataset_without_fabricating_samples(tmp_path: Path):
    config = EvaluationConfig(dataset_root=tmp_path / "evaluation", results_root=tmp_path / "results", plots_root=tmp_path / "plots")
    summary = EvaluationRunner(config).run()
    assert summary["status"] == "no_samples"
    assert summary["dataset"] == {"clean": 0, "fgsm": 0, "pgd": 0, "patch": 0}
    assert (config.results_root / "summary.json").exists()
