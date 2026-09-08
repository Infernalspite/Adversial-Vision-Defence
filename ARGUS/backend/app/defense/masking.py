"""Suspicious-region masking defense."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from app.defense.base_defense import BaseDefense, DefenseResult, change_magnitude


class MaskingDefense(BaseDefense):
    """Inpaint suspicious bounding boxes using surrounding pixels."""

    def defend(self, image: np.ndarray, context: dict[str, Any] | None = None) -> DefenseResult:
        """Replace localized suspicious regions; leave clean inputs unchanged."""
        started = time.perf_counter()
        values = context or {}
        regions = values.get("suspicious_regions", [])
        padding = int(values.get("parameters", {}).get("mask_padding", 4))
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        height, width = image.shape[:2]
        for region in regions:
            x = max(0, int(region["x"]) - padding)
            y = max(0, int(region["y"]) - padding)
            right = min(width, int(region["x"]) + int(region["width"]) + padding)
            bottom = min(height, int(region["y"]) + int(region["height"]) + padding)
            mask[y:bottom, x:right] = 255
        if not np.any(mask):
            defended = image.copy()
            applied = False
            explanation = "No suspicious region was available; masking was skipped."
        else:
            defended = cv2.inpaint(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), mask, 3, cv2.INPAINT_TELEA)
            defended = cv2.cvtColor(defended, cv2.COLOR_BGR2RGB)
            applied = True
            explanation = "Inpainted localized suspicious regions using surrounding pixels."
        return DefenseResult(
            defense_name="mask",
            defense_applied=applied,
            defended_image=defended.astype(np.uint8),
            parameters={"mask_padding": padding},
            affected_regions=list(regions),
            change_magnitude=change_magnitude(image, defended),
            processing_time_ms=(time.perf_counter() - started) * 1000,
            explanation=explanation,
            metadata={"mask_area_pixels": int(np.count_nonzero(mask))},
        )
