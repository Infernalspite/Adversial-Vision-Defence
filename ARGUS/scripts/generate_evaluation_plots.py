"""Generate presentation-ready plots from raw evaluation JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--results", type=Path, default=ROOT / "data/evaluation/results"); parser.add_argument("--output", type=Path, default=ROOT / "data/evaluation/plots"); args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for path in args.results.glob("*_results.jsonl"):
        datasets[path.stem.replace("_results", "")] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    categories = list(datasets)
    plt.figure(figsize=(8, 4)); plt.bar(categories, [sum(item["attack_score"] for item in datasets[name]) / len(datasets[name]) if datasets[name] else 0 for name in categories], color="#4cc9c0"); plt.ylabel("Mean attack score"); plt.title("Attack score by category"); plt.tight_layout(); plt.savefig(args.output / "attack_score_distribution.png", dpi=160); plt.close()
    plt.figure(figsize=(8, 4)); plt.bar(categories, [sum(item["attack_detected"] for item in datasets[name]) / len(datasets[name]) if datasets[name] else 0 for name in categories], color="#efaa68"); plt.ylabel("Detection rate"); plt.title("Detection rate by category"); plt.ylim(0, 1); plt.tight_layout(); plt.savefig(args.output / "detection_rate.png", dpi=160); plt.close()
    plt.figure(figsize=(8, 4)); states = ["TRUSTED", "DEFENDED", "ABSTAIN"]; width = .24; positions = range(len(categories));
    for index, state in enumerate(states): plt.bar([position + (index - 1) * width for position in positions], [sum(item["final_state"] == state for item in datasets[name]) / len(datasets[name]) if datasets[name] else 0 for name in categories], width=width, label=state)
    plt.xticks(list(positions), categories); plt.ylabel("Rate"); plt.ylim(0, 1); plt.title("Final decision distribution"); plt.legend(); plt.tight_layout(); plt.savefig(args.output / "decision_distribution.png", dpi=160); plt.close()
    print(f"Plots saved to {args.output}")


if __name__ == "__main__": main()
