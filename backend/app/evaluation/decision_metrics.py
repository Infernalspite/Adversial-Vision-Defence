"""Final-state and safety metrics."""

from collections import Counter
from typing import Any


def decision_metrics(records: list[dict[str, Any]]) -> dict[str, float | int]:
    """Calculate state distribution and safety outcomes."""
    count = Counter(record.get("final_state") for record in records)
    total = len(records)
    adversarial = [record for record in records if record.get("is_adversarial")]
    clean = [record for record in records if not record.get("is_adversarial")]
    trusted_adversarial = sum(record.get("final_state") == "TRUSTED" for record in adversarial)
    safe_rejected = sum(record.get("final_state") in {"DEFENDED", "ABSTAIN"} for record in adversarial)
    clean_rejected = sum(record.get("final_state") == "ABSTAIN" for record in clean)
    return {"trusted_count": count["TRUSTED"], "defended_count": count["DEFENDED"], "abstain_count": count["ABSTAIN"], "trusted_rate": count["TRUSTED"] / total if total else 0.0, "defended_rate": count["DEFENDED"] / total if total else 0.0, "abstain_rate": count["ABSTAIN"] / total if total else 0.0, "unsafe_acceptance_rate": trusted_adversarial / len(adversarial) if adversarial else 0.0, "safe_rejection_rate": safe_rejected / len(adversarial) if adversarial else 0.0, "clean_rejection_rate": clean_rejected / len(clean) if clean else 0.0}
