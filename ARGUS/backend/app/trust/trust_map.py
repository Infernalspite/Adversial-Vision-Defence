"""Spatial trust-map representation and numerical utilities."""

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class TrustMap:
    """Spatial perception-confidence representation for one image."""

    values: np.ndarray
    width: int
    height: int
    resolution: tuple[int, int]
    normalization: dict[str, Any]
    generation_time_ms: float
    contributing_detectors: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


class TrustMapBuilder:
    """Build a normalized trust map from spatial anomaly sources."""

    def build(
        self,
        image: np.ndarray,
        evidence: dict[str, np.ndarray],
        weights: dict[str, float] | None = None,
        generation_time_ms: float = 0.0,
    ) -> TrustMap:
        """Fuse spatial anomaly maps and return trust in the [0, 1] range."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected an RGB image")
        target_height, target_width = image.shape[:2]
        configured = weights or {
            "saliency": 0.30,
            "frequency": 0.25,
            "local": 0.20,
            "patch": 0.25,
        }
        maps: dict[str, np.ndarray] = {}
        for name, values in evidence.items():
            normalized = normalize_map(values)
            if normalized.size:
                maps[name] = resize_map(normalized, (target_height, target_width))
        available = {name: configured[name] for name in maps if name in configured and configured[name] > 0}
        if not available:
            anomaly = np.zeros((target_height, target_width), dtype=np.float32)
        else:
            total_weight = sum(available.values())
            anomaly = sum((weight / total_weight) * maps[name] for name, weight in available.items())
        trust = np.clip(1.0 - normalize_map(anomaly), 0.0, 1.0).astype(np.float32)
        return TrustMap(
            values=trust,
            width=target_width,
            height=target_height,
            resolution=(target_height, target_width),
            normalization={"input_range": "[0, 1]", "constant_map": "mapped to zero anomaly"},
            generation_time_ms=generation_time_ms,
            contributing_detectors=tuple(available),
            metadata={"available_sources": tuple(maps), "weights": available},
        )


def normalize_map(values: Any) -> np.ndarray:
    """Convert arbitrary numeric map data into a finite [0, 1] float map."""
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum <= minimum:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - minimum) / (maximum - minimum)).astype(np.float32)


def resize_map(values: Any, target_shape: tuple[int, int], binary: bool = False) -> np.ndarray:
    """Resize a map to ``(height, width)`` with continuous or nearest interpolation."""
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return np.zeros(target_shape, dtype=np.float32)
    interpolation = cv2.INTER_NEAREST if binary else cv2.INTER_LINEAR
    resized = cv2.resize(array, (target_shape[1], target_shape[0]), interpolation=interpolation)
    return np.clip(resized, 0.0, 1.0).astype(np.float32)
