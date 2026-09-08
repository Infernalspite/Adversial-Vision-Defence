"""Mine hard negatives from persisted audit records."""

from typing import Any, Iterable

from app.learning.failure_analyzer import FailureAnalyzer
from app.learning.schemas import HardNegative


class HardNegativeMiner:
    """Create reviewable hard-negative records without changing live decisions."""

    def __init__(self, analyzer: FailureAnalyzer | None = None) -> None:
        self.analyzer = analyzer or FailureAnalyzer()

    def mine(self, records: Iterable[dict[str, Any]]) -> list[HardNegative]:
        mined: list[HardNegative] = []
        for record in records:
            failure = self.analyzer.analyze(record)
            if failure is None:
                continue
            details = record.get("details", record)
            mined.append(HardNegative(
                request_id=str(record.get("request_id", "unknown")),
                category=failure.category,
                explanation=failure.explanation,
                timestamp=record.get("timestamp"),
                attack_type=details.get("attack_type"),
                attack_score=details.get("attack_score"),
                trust_score=(details.get("trust_map") or {}).get("global_trust_score"),
                verification_score=details.get("verification_score"),
                final_state=details.get("final_state"),
                original_prediction=details.get("original_prediction"),
                defended_prediction=details.get("defended_prediction"),
                expected_behavior="ABSTAIN_OR_DEFEND" if failure.category == "unsafe_acceptance" else "REVIEW",
                image_reference=details.get("image_reference"),
                metadata={"stage": record.get("stage"), "severity": failure.severity},
            ))
        return mined
