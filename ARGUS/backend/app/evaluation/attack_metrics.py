"""Attack effectiveness metrics."""

from typing import Any


def attack_success_rate(records: list[dict[str, Any]]) -> float:
    """Measure prediction changes only among correctly classified clean sources."""
    eligible = [record for record in records if record.get("is_adversarial") and record.get("clean_correct") is True]
    if not eligible:
        return 0.0
    return sum(bool(record.get("prediction_changed")) for record in eligible) / len(eligible)
