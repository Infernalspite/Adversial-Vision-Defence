"""Lightweight image purification defense."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from app.defense.base_defense import BaseDefense, DefenseResult, change_magnitude


class PurificationDefense(BaseDefense):
    """Apply conservative denoising and mild reconstruction."""

    def defend(self, image: np.ndarray, context: dict[str, Any] | None = None) -> DefenseResult:
        """Run a small bilateral denoising pass without claiming certification."""
        started = time.perf_counter()
        strength = int((context or {}).get("parameters", {}).get("strength", 1))
        strength = max(1, min(3, strength))
        defended = cv2.bilateralFilter(image, 5 + 2 * strength, 20 * strength, 20 * strength)
        return DefenseResult(
            defense_name="purification",
            defense_applied=True,
            defended_image=defended.astype(np.uint8),
            parameters={"strength": strength},
            affected_regions=[],
            change_magnitude=change_magnitude(image, defended),
            processing_time_ms=(time.perf_counter() - started) * 1000,
            explanation="Applied lightweight denoising purification.",
        )
