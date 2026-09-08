"""Certified inference roadmap placeholder."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from app.defense.base_defense import BaseDefense, DefenseResult


class CertifiedInferenceDefense(BaseDefense):
    """Explicitly inactive; no fake certification is exposed."""

    def defend(self, image: np.ndarray, context: dict[str, Any] | None = None) -> DefenseResult:
        """Return a non-applied roadmap result."""
        return DefenseResult(
            defense_name="certified",
            defense_applied=False,
            defended_image=image.copy(),
            parameters={},
            affected_regions=[],
            change_magnitude=0.0,
            processing_time_ms=0.0,
            explanation="Certified inference is a future production/roadmap capability.",
            metadata={"status": "NOT_IMPLEMENTED", "reason": "Certified inference is a future production/roadmap capability."},
        )
