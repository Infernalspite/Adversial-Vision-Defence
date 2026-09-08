"""Defense recovery and confidence metrics."""

from typing import Any


def defense_metrics(records: list[dict[str, Any]]) -> dict[str, float | int]:
    """Calculate recovery, confidence movement, and defense failure."""
    attacked = [record for record in records if record.get("attack_success")]
    recovered = [record for record in attacked if record.get("defense_recovered")]
    failures = [record for record in records if not record.get("defense_acceptable", False)]
    confidence_drop = [float(record["adversarial_confidence"]) - float(record["clean_confidence"]) for record in records if record.get("clean_confidence") is not None]
    confidence_restore = [float(record["defended_confidence"]) - float(record["adversarial_confidence"]) for record in records if record.get("defended_confidence") is not None]
    return {"successfully_attacked": len(attacked), "recovered": len(recovered), "defense_recovery_rate": len(recovered) / len(attacked) if attacked else 0.0, "defense_failure_rate": len(failures) / len(records) if records else 0.0, "average_confidence_drop": sum(confidence_drop) / len(confidence_drop) if confidence_drop else 0.0, "average_confidence_restored": sum(confidence_restore) / len(confidence_restore) if confidence_restore else 0.0}
