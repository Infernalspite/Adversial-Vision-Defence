"""Conservative final decision engine for Phase 7."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.pipeline.states import FinalState


@dataclass(frozen=True, slots=True)
class DecisionPolicyConfig:
    """Configurable MVP thresholds for the final decision policy."""

    attack_threshold: float = 0.25
    high_trust_threshold: float = 0.70
    verification_threshold: float = 0.70
    minimum_confidence: float = 0.50
    recovery_threshold: float = 0.50

    def __post_init__(self) -> None:
        values = (self.attack_threshold, self.high_trust_threshold, self.verification_threshold, self.minimum_confidence, self.recovery_threshold)
        if any(value < 0 or value > 1 for value in values):
            raise ValueError("decision thresholds must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """Evidence passed from Phases 3-6 into the final decision engine."""

    attack_score: float
    attack_detected: bool
    global_trust_score: float
    suspicious_regions: list[dict[str, Any]]
    defense_applied: bool
    defense_method: str | None
    original_prediction: str | None
    original_confidence: float | None
    defended_prediction: str | None
    defended_confidence: float | None
    verification_score: float
    object_consistency: float
    geometry_consistency: float
    scene_consistency: float
    defended_stability: float = 0.0
    """Fraction of benign probes on the *defended* image that kept the same label (0 when no defense ran)."""


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """Final externally meaningful result and explainable evidence."""

    final_state: FinalState
    final_prediction: str | None
    final_confidence: float | None
    decision_score: float
    explanation: str
    decision_reasons: tuple[str, ...]
    evidence_summary: dict[str, Any] = field(default_factory=dict)


class FinalDecisionEngine:
    """Map existing evidence to TRUSTED, DEFENDED, or ABSTAIN only."""

    def __init__(self, config: DecisionPolicyConfig | None = None) -> None:
        self.config = config or DecisionPolicyConfig()

    def decide(self, context: DecisionContext) -> DecisionResult:
        """Apply conservative trusted/defended/abstain rules."""
        attack_score = _bounded(context.attack_score)
        trust = _bounded(context.global_trust_score)
        verification = _bounded(context.verification_score)
        original_confidence = _bounded_optional(context.original_confidence)
        defended_confidence = _bounded_optional(context.defended_confidence)
        component_floor = min(_bounded(context.object_consistency), _bounded(context.geometry_consistency), _bounded(context.scene_consistency))
        selected_confidence = defended_confidence if context.defense_applied else original_confidence
        decision_score = _bounded(0.25 * (1 - attack_score) + 0.25 * trust + 0.25 * verification + 0.25 * (selected_confidence or 0.0))
        suspicious = len(context.suspicious_regions)

        trusted = (
            not context.attack_detected
            and attack_score < self.config.attack_threshold
            and trust >= self.config.high_trust_threshold
            and verification >= self.config.verification_threshold
            and (original_confidence or 0.0) >= self.config.minimum_confidence
            and component_floor >= self.config.verification_threshold
        )
        if trusted:
            reasons = (
                "Attack evidence remained below the detection threshold.",
                "Global trust exceeded the configured high-trust threshold.",
                "Object, geometry, and scene verification were strong.",
                "Original prediction confidence was sufficient.",
            )
            return DecisionResult(FinalState.TRUSTED, context.original_prediction, original_confidence, decision_score, "The input remained below attack risk thresholds and its interpretation was strongly verified.", reasons, _evidence(context, decision_score, suspicious))

        # A "defended" outcome must mean the defense *recovered* the input:
        # the defense changed the suspicious pre-defense interpretation, and
        # the post-defense interpretation is stable under benign probes (a
        # decision still sitting on an adversarially manufactured boundary
        # would flip). Requiring merely "defended == original" would certify
        # the adversarial label itself whenever a defense changes nothing.
        semantic_recovery = (
            context.defended_prediction is not None
            and context.original_prediction is not None
            and context.defended_prediction != context.original_prediction
            and context.defended_stability >= 0.5
        )
        defended = (
            context.attack_detected
            and context.defense_applied
            and semantic_recovery
            and attack_score >= self.config.attack_threshold
            and verification >= self.config.verification_threshold
            and (defended_confidence or 0.0) >= self.config.recovery_threshold
            and component_floor >= self.config.verification_threshold
            and context.defended_prediction is not None
        )
        if defended:
            reasons = [
                "Adversarial evidence exceeded the detection threshold.",
                f"A {context.defense_method or 'configured'} defense was applied.",
                "Object, geometry, and scene verification exceeded the configured threshold.",
                "The defended prediction retained sufficient confidence.",
            ]
            if suspicious:
                reasons.append(f"{suspicious} suspicious region(s) were included in the evidence chain.")
            return DecisionResult(FinalState.DEFENDED, context.defended_prediction, defended_confidence, decision_score, "Adversarial evidence was detected, but the defended interpretation met the configured verification requirements.", tuple(reasons), _evidence(context, decision_score, suspicious))

        reasons = []
        if context.attack_detected or attack_score >= self.config.attack_threshold:
            reasons.append("Adversarial evidence remained high.")
        if not context.defense_applied:
            reasons.append("No defense produced a verified recovery result.")
        elif context.defended_prediction == context.original_prediction:
            reasons.append("The applied defense did not change the suspicious interpretation.")
        elif context.defended_stability < 0.5:
            reasons.append("The post-defense interpretation remained unstable under benign probes.")
        elif (defended_confidence or 0.0) < self.config.recovery_threshold:
            reasons.append("Defended confidence remained below the recovery threshold.")
        if verification < self.config.verification_threshold or component_floor < self.config.verification_threshold:
            reasons.append("Semantic verification remained inconclusive or contradictory.")
        if not reasons:
            reasons.append("Available evidence was insufficient to safely trust an interpretation.")
        return DecisionResult(FinalState.ABSTAIN, None, None, decision_score, "The system abstained because available evidence was insufficient to safely select an interpretation.", tuple(reasons), _evidence(context, decision_score, suspicious))


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _bounded_optional(value: float | None) -> float | None:
    return None if value is None else _bounded(value)


def _evidence(context: DecisionContext, decision_score: float, suspicious: int) -> dict[str, Any]:
    return {
        "attack_score": _bounded(context.attack_score),
        "global_trust_score": _bounded(context.global_trust_score),
        "verification_score": _bounded(context.verification_score),
        "object_consistency": _bounded(context.object_consistency),
        "geometry_consistency": _bounded(context.geometry_consistency),
        "scene_consistency": _bounded(context.scene_consistency),
        "suspicious_region_count": suspicious,
        "decision_score": decision_score,
    }
