"""Report validation-only threshold tradeoffs without changing production config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.evaluation.detection_metrics import detection_metrics


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--results", type=Path, default=ROOT / "data/evaluation/results"); args = parser.parse_args()
    records = []
    for path in args.results.glob("*_results.jsonl"):
        if path.name.startswith("clean"): continue
        records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    if not records: print("No validation results found."); return
    best = None
    for threshold_index in range(10, 91):
        threshold = threshold_index / 100
        candidate = [dict(record, attack_detected=float(record.get("attack_score", 0)) >= threshold) for record in records]
        metrics = detection_metrics(candidate)
        unsafe = sum(not record["attack_detected"] for record in candidate) / len(candidate)
        objective = metrics["f1"] - unsafe * 0.5
        if best is None or objective > best[0]: best = (objective, threshold, metrics, unsafe)
    _, threshold, metrics, unsafe = best
    print(f"Validation threshold: {threshold:.2f}")
    print(f"Validation TPR: {metrics['tpr']:.1%}")
    print(f"Validation FPR: {metrics['fpr']:.1%}")
    print(f"Validation F1: {metrics['f1']:.1%}")
    print(f"Validation unsafe acceptance proxy: {unsafe:.1%}")
    print("This report does not modify production thresholds or use a final test set.")


if __name__ == "__main__": main()
