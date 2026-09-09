"""Run the final evaluation and generate machine-readable summaries."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.evaluation.config import EvaluationConfig
from app.evaluation.evaluator import EvaluationRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "evaluation")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--calibration", type=Path, default=None)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8")) if args.calibration else {}
    config = EvaluationConfig(
        dataset_root=args.dataset,
        results_root=args.dataset / "results",
        plots_root=args.dataset / "plots",
        max_samples_per_category=args.max_samples,
        detector_weights=calibration.get("selected_weights"),
        detection_threshold=float(calibration.get("logistic_regression", {}).get("threshold", calibration.get("selected_threshold", 0.70))),
        logistic_regression=calibration.get("logistic_regression"),
    )
    summary = EvaluationRunner(config).run()
    metadata = {"timestamp": datetime.now(timezone.utc).isoformat(), "dataset_root": str(config.dataset_root), "seed": config.seed, "attack_parameters": config.attack_parameters, "thresholds": {"attack": config.detection_threshold, "trust": 0.70, "verification": 0.70}, "detector_weights": config.detector_weights, "calibration": str(args.calibration) if args.calibration else None, "git_commit": _git_commit()}
    (config.results_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    _write_csv(summary, config.results_root / "summary.csv")
    print_report(summary)


def _git_commit() -> str | None:
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError): return None


def _write_csv(summary: dict, path: Path) -> None:
    rows = []
    for category, values in summary.get("categories", {}).items():
        row = {"category": category}; row.update({f"attack_{key}": value for key, value in values.get("attack", {}).items()}); row.update({f"detection_{key}": value for key, value in values.get("detection", {}).items()}); row.update({f"defense_{key}": value for key, value in values.get("defense", {}).items()}); row.update({f"decision_{key}": value for key, value in values.get("decisions", {}).items()}); rows.append(row)
    if not rows: path.write_text("category\n", encoding="utf-8"); return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def print_report(summary: dict) -> None:
    print("=" * 56); print("ARGUS-AEGIS EVALUATION"); print("=" * 56)
    if summary.get("status") == "no_samples":
        print("\nNo evaluation samples found. Populate data/evaluation before interpreting metrics.\n")
    print("\nDataset")
    for category, count in summary.get("dataset", {}).items(): print(f"{category.title():<12} {count}")
    for category, values in summary.get("categories", {}).items():
        print(f"\n{category.upper()}")
        print(f"Attack success: {values['attack']['attack_success_rate']:.1%}")
        print(f"Detection TPR:  {values['detection']['tpr']:.1%}")
        print(f"Detection FPR:  {values['detection']['fpr']:.1%}")
        print(f"Precision/F1:    {values['detection']['precision']:.1%} / {values['detection']['f1']:.1%}")
        print(f"Recovery:       {values['defense']['defense_recovery_rate']:.1%}")
        print(f"Trusted/Defended/Abstain: {values['decisions']['trusted_rate']:.1%} / {values['decisions']['defended_rate']:.1%} / {values['decisions']['abstain_rate']:.1%}")
    print("\nCombined safety")
    for key in ("unsafe_acceptance_rate", "safe_rejection_rate", "clean_rejection_rate"): print(f"{key.replace('_', ' ').title():<25} {summary['combined']['decisions'][key]:.1%}")
    print("\nRaw results and CSV are under data/evaluation/results/")


if __name__ == "__main__": main()
