"""Full ARGUS-AEGIS pipeline through the Phase 7 final decision."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.audit.audit_logger import AuditEvent, AuditLogger
from app.pipeline.defense_pipeline import ArgusDefensePipeline, DefensePipelineResult
from app.pipeline.decision_engine import DecisionContext, DecisionResult, FinalDecisionEngine
from app.pipeline.fallback import FallbackResponse, abstain_fallback
from app.verification.verification_pipeline import VerificationPipeline, VerificationPipelineResult


@dataclass(slots=True)
class PipelineContext:
    """Per-request values shared across pipeline stages."""

    request_id: str
    image: Any
    metadata: dict[str, Any]


class ArgusPipeline:
    """Coordinate detection, trust, defense, re-inference, and verification."""

    def __init__(self, defense_pipeline: ArgusDefensePipeline | None = None, verification_pipeline: VerificationPipeline | None = None, decision_engine: FinalDecisionEngine | None = None, audit_logger: AuditLogger | None = None) -> None:
        self.defense_pipeline = defense_pipeline or ArgusDefensePipeline()
        self.verification_pipeline = verification_pipeline or VerificationPipeline()
        self.decision_engine = decision_engine or FinalDecisionEngine()
        self.audit_logger = audit_logger or AuditLogger()

    def run(self, context: PipelineContext) -> "FullPipelineResult":
        """Run detection through final decision and return no raw image data."""
        model = context.metadata.get("model")
        if model is None:
            raise ValueError("Pipeline context metadata must include a model")
        defense_result = self.defense_pipeline.run(context.image, model, context.metadata.get("patch_mask"))
        verification_result = self.verification_pipeline.run(
            context.image,
            defense_result.orchestration.defense.defended_image,
            defense_result.original_prediction,
            defense_result.defended_prediction,
            defense_result.trust.trust_map,
            defense_result.trust.localization.regions,
            defense_result.detection,
        )
        verification = verification_result.verification
        decision = self.decision_engine.decide(DecisionContext(
            attack_score=defense_result.detection.attack_score,
            attack_detected=defense_result.detection.attack_detected,
            global_trust_score=float(defense_result.trust.trust_map.values.mean()),
            suspicious_regions=defense_result.trust.localization.regions,
            defense_applied=defense_result.orchestration.defense.defense_applied,
            defense_method=defense_result.orchestration.decision.selected_defense,
            original_prediction=defense_result.original_prediction.class_name,
            original_confidence=defense_result.original_prediction.confidence,
            defended_prediction=defense_result.defended_prediction.class_name,
            defended_confidence=defense_result.defended_prediction.confidence,
            defended_stability=defense_result.defended_stability,
            verification_score=verification.verification_score,
            object_consistency=verification.object_consistency["object_consistency_score"],
            geometry_consistency=verification.geometry_consistency["structural_similarity"],
            scene_consistency=verification.scene_consistency["scene_similarity_score"],
        ))
        fallback = abstain_fallback() if decision.final_state.value == "ABSTAIN" else None
        self.audit_logger.record(AuditEvent(
            request_id=context.request_id,
            stage="decision",
            message=decision.explanation,
            timestamp=datetime.now(timezone.utc),
            details={
                "original_prediction": defense_result.original_prediction.class_name,
                "defended_prediction": defense_result.defended_prediction.class_name,
                "attack_score": defense_result.detection.attack_score,
                "trust_map": {"global_trust_score": float(defense_result.trust.trust_map.values.mean())},
                "suspicious_regions": defense_result.trust.localization.regions,
                "selected_defense": defense_result.orchestration.decision.selected_defense,
                "verification_score": verification.verification_score,
                "final_state": decision.final_state.value,
                "final_prediction": decision.final_prediction,
                "decision_reasons": decision.decision_reasons,
                "processing_time_ms": defense_result.processing_time_ms + verification_result.processing_time_ms,
            },
        ))
        return FullPipelineResult(defense_result, verification_result, decision, fallback, self.audit_logger.for_request(context.request_id))


@dataclass(frozen=True, slots=True)
class FullPipelineResult:
    """Complete Phase 1-7 output with exactly one final state."""

    defense: DefensePipelineResult
    verification: VerificationPipelineResult
    decision: DecisionResult
    fallback: FallbackResponse | None
    audit: list[dict[str, Any]]
