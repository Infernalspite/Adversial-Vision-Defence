"""Compare recorded configurations without cherry-picking metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--results", type=Path, default=Path("data/evaluation/results")); parser.add_argument("--config", action="append", metavar="NAME=PATH", help="Recorded configuration result directory; repeatable."); args = parser.parse_args()
    configurations = {"Full ARGUS-AEGIS": args.results}
    for value in args.config or []:
        name, separator, path = value.partition("=")
        if separator and name and path: configurations[name] = Path(path)
    print("ARGUS-AEGIS ABLATION SUMMARY")
    print("Configuration              Unsafe Acceptance   Recovery   Abstain")
    for name, directory in configurations.items():
        records = []
        for path in directory.glob("*_results.jsonl"):
            records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
        total = len(records); adversarial = [record for record in records if record.get("is_adversarial")]
        if not records:
            print(f"{name:<26} no recorded samples")
            continue
        unsafe = sum(record.get("final_state") == "TRUSTED" for record in adversarial) / len(adversarial) if adversarial else 0
        recovery = sum(bool(record.get("defense_recovered")) for record in adversarial) / len(adversarial) if adversarial else 0
        abstain = sum(record.get("final_state") == "ABSTAIN" for record in records) / total
        print(f"{name:<26} {unsafe:.1%}              {recovery:.1%}     {abstain:.1%}")
    print("Only separately recorded configurations are compared; no baseline values are fabricated.")


if __name__ == "__main__": main()
