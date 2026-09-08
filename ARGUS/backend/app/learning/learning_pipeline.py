"""High-level offline self-healing cycle."""

import json
from pathlib import Path
from typing import Any

from app.audit.audit_logger import AuditLogger
from app.learning.dataset_builder import CandidateDatasetBuilder
from app.learning.hard_negative_mining import HardNegativeMiner
from app.learning.model_validation import ModelValidationGate


def run_learning_cycle(audit_path: str | Path = "data/audit/audit_records.json", output_root: str | Path = "data/learning") -> dict[str, Any]:
    """Run audit -> failure analysis -> hard negatives -> candidate artifact."""
    logger = AuditLogger(audit_path)
    hard_negatives = HardNegativeMiner().mine([event for event in logger.recent(1000)])
    output = Path(output_root); output.mkdir(parents=True, exist_ok=True)
    candidate_path = CandidateDatasetBuilder().build(hard_negatives, output / "hard_negatives.json")
    validation = ModelValidationGate().validate("candidate_pending", {"unsafe_acceptance_rate": 0.0, "clean_rejection_rate": 0.0, "tpr": 0.0, "defense_recovery_rate": 0.0})
    summary = {"hard_negative_count": len(hard_negatives), "candidate_dataset": str(candidate_path), "validation": {"status": validation.status, "reasons": validation.reasons}, "production_modified": False}
    (output / "learning_cycle.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
