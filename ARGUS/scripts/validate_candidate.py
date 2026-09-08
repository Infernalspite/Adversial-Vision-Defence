"""Validate persisted candidate metrics without deploying anything."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "backend"))
from app.learning.model_validation import ModelValidationGate

parser = argparse.ArgumentParser(); parser.add_argument("--metrics", type=Path, required=True); parser.add_argument("--candidate-id", default="candidate"); parser.add_argument("--baseline", type=Path, default=None); args = parser.parse_args(); metrics = json.loads(args.metrics.read_text(encoding="utf-8")); baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else {}; print(json.dumps(asdict(ModelValidationGate().validate(args.candidate_id, metrics, baseline)), indent=2, default=str))
