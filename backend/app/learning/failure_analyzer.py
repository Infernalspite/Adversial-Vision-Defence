"""Categorize audit-record failures for offline review."""

from typing import Any

from app.learning.schemas import FailureAnalysis


class FailureAnalyzer:
    """Apply transparent evidence-based failure categories."""

    def analyze(self, record: dict[str, Any]) -> FailureAnalysis | None:
        details = record.get("details", record)
        final_state = details.get("final_state")
        attack_score = float(details.get("attack_score", 0.0) or 0.0)
        verification = float(details.get("verification_score", 1.0) or 0.0)
        trust = float(details.get("trust_map", {}).get("global_trust_score", 1.0) or 0.0)
        if final_state == "TRUSTED" and attack_score >= 0.7:
            return FailureAnalysis("unsafe_acceptance", "Adversarial evidence exceeded threshold but the input was accepted as TRUSTED.", 1.0)
        if final_state == "ABSTAIN" and attack_score < 0.3:
            return FailureAnalysis("clean_rejection", "Low attack evidence resulted in ABSTAIN, increasing clean-input usability cost.", 0.5)
        if details.get("defense_applied") and verification < 0.7:
            return FailureAnalysis("verification_failure", "A defense was applied but semantic verification remained weak.", 0.8)
        if details.get("defense_applied") and details.get("defended_prediction") != details.get("original_prediction"):
            return FailureAnalysis("defense_failure", "The defended interpretation did not recover the original interpretation.", 0.8)
        if attack_score >= 0.7 and trust >= 0.7 and verification < 0.5:
            return FailureAnalysis("localization_failure", "Global evidence and trust disagreed with weak verification evidence.", 0.7)
        detectors = details.get("detectors", {})
        scores = [float(value.get("score", 0)) for value in detectors.values() if isinstance(value, dict)]
        if scores and max(scores) - min(scores) > 0.6:
            return FailureAnalysis("detector_disagreement", "Detector outputs strongly disagree and require offline review.", 0.6)
        return None
