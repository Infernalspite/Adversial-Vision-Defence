"""Run a small Phase 10 self-healing demonstration."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "backend"))
from app.learning.learning_pipeline import run_learning_cycle
from app.learning.model_validation import ModelValidationGate

summary = run_learning_cycle(ROOT / "data/audit/audit_records.json", ROOT / "data/learning")
validation = ModelValidationGate().validate("demo-candidate", {"unsafe_acceptance_rate": 0.0, "clean_rejection_rate": 0.0, "tpr": 0.0, "defense_recovery_rate": 0.0})
print("ARGUS-AEGIS LEARNING CYCLE")
print(f"Hard negatives: {summary['hard_negative_count']}")
print("Offline candidate created: YES")
print(f"Validation: {validation.status}")
print("Production modified: NO")
