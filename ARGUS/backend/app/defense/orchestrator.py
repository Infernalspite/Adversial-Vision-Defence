"""Autonomous defense orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.defense.base_defense import BaseDefense, DefenseResult
from app.defense.policy import DefenseDecision, DefensePolicy


@dataclass(frozen=True, slots=True)
class DefenseTrace:
    """Explainable record of policy selection and execution."""

    timestamp: str
    attack_score: float
    selected_defense: str
    reason: str
    input_trust_score: float
    suspicious_regions: int
    defense_parameters: dict[str, Any]
    processing_time_ms: float
    success: bool
    output_image_reference: str | None = None


@dataclass(frozen=True, slots=True)
class OrchestrationResult:
    """Defense result plus the policy decision and trace."""

    defense: DefenseResult
    decision: DefenseDecision
    trace: DefenseTrace


class DefenseOrchestrator:
    """Select and execute one appropriate MVP defense strategy."""

    def __init__(self, policy: DefensePolicy | None = None) -> None:
        self.policy = policy or DefensePolicy()

    def execute(self, image: np.ndarray, context: dict[str, Any]) -> OrchestrationResult:
        """Apply policy-selected defense and preserve an explainable trace."""
        started = time.perf_counter()
        decision = self.policy.decide(
            attack_score=float(context.get("attack_score", 0.0)),
            attack_type=context.get("attack_type"),
            detector_evidence=context.get("detector_evidence", {}),
            global_trust_score=float(context.get("global_trust_score", 1.0)),
            suspicious_regions=context.get("suspicious_regions", []),
            suspicious_area_percentage=float(context.get("suspicious_area_percentage", 0.0)),
        )
        from app.defense import get_defense

        strategy: BaseDefense = get_defense(decision.selected_defense)
        defense = strategy.defend(image, {**context, "parameters": decision.parameters})
        elapsed = (time.perf_counter() - started) * 1000
        trace = DefenseTrace(
            timestamp=datetime.now(timezone.utc).isoformat(),
            attack_score=float(context.get("attack_score", 0.0)),
            selected_defense=decision.selected_defense,
            reason=decision.reason,
            input_trust_score=float(context.get("global_trust_score", 1.0)),
            suspicious_regions=len(context.get("suspicious_regions", [])),
            defense_parameters=decision.parameters,
            processing_time_ms=elapsed,
            success=True,
        )
        return OrchestrationResult(defense, decision, trace)

    def choose(self, attack_score: float, attack_type: str | None, localization: Any, confidence: float, evidence: list[Any]) -> BaseDefense:
        """Compatibility adapter returning the policy-selected strategy."""
        decision = self.policy.decide(attack_score, attack_type, evidence, 1 - confidence, localization)
        from app.defense import get_defense
        return get_defense(decision.selected_defense)
