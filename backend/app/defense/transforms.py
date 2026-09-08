"""Lightweight image-transform defense."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from app.defense.base_defense import BaseDefense, DefenseResult, change_magnitude


class TransformDefense(BaseDefense):
    """Apply a configurable mild blur, median filter, or reconstruction transform."""

    def defend(self, image: np.ndarray, context: dict[str, Any] | None = None) -> DefenseResult:
        """Apply one lightweight transform while preserving image dimensions."""
        started = time.perf_counter()
        parameters = (context or {}).get("parameters", {})
        method = parameters.get("method", "gaussian_blur")
        kernel_size = int(parameters.get("kernel_size", 3))
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer of at least 3")
        if method == "gaussian_blur":
            defended = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif method == "median_filter":
            defended = cv2.medianBlur(image, kernel_size)
        elif method == "resize_reconstruct":
            height, width = image.shape[:2]
            smaller = cv2.resize(image, (max(1, width // 2), max(1, height // 2)), interpolation=cv2.INTER_AREA)
            defended = cv2.resize(smaller, (width, height), interpolation=cv2.INTER_LINEAR)
        else:
            raise ValueError(f"Unknown transform method: {method}")
        return DefenseResult(
            defense_name="transform",
            defense_applied=True,
            defended_image=defended.astype(np.uint8),
            parameters={"method": method, "kernel_size": kernel_size},
            affected_regions=[],
            change_magnitude=change_magnitude(image, defended),
            processing_time_ms=(time.perf_counter() - started) * 1000,
            explanation=f"Applied lightweight {method} transformation.",
        )
