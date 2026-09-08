"""Mine offline hard negatives from persisted audit records."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "backend"))
from app.audit.audit_logger import AuditLogger
from app.learning.dataset_builder import CandidateDatasetBuilder
from app.learning.hard_negative_mining import HardNegativeMiner

parser = argparse.ArgumentParser(); parser.add_argument("--audit", type=Path, default=ROOT / "data/audit/audit_records.json"); parser.add_argument("--output", type=Path, default=ROOT / "data/learning/hard_negatives.json"); args = parser.parse_args()
miner = HardNegativeMiner(); mined = miner.mine(AuditLogger(args.audit).recent(1000)); CandidateDatasetBuilder().build(mined, args.output); print(json.dumps({"hard_negative_count": len(mined), "output": str(args.output)}, indent=2))
