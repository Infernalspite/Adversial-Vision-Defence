"""Run the bounded audit-to-candidate learning cycle."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "backend"))
from app.learning.learning_pipeline import run_learning_cycle

parser = argparse.ArgumentParser(); parser.add_argument("--audit", type=Path, default=ROOT / "data/audit/audit_records.json"); parser.add_argument("--output", type=Path, default=ROOT / "data/learning"); args = parser.parse_args(); print(run_learning_cycle(args.audit, args.output))
