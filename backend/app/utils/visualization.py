"""Trust-map and suspicious-region visualization utilities."""

from typing import Any

import cv2
import numpy as np

from app.trust.trust_map import TrustMap


def serialize_trust_map(trust_map: Any) -> dict[str, Any]:
    """Return compact metadata without embedding the raw floating-point matrix."""
    if not isinstance(trust_map, TrustMap):
        raise TypeError("trust_map must be a TrustMap")
    return {
        "width": trust_map.width,
        "height": trust_map.height,
        "resolution": trust_map.resolution,
        "normalization": trust_map.normalization,
        "generation_time_ms": trust_map.generation_time_ms,
        "contributing_detectors": trust_map.contributing_detectors,
    }


def trust_map_png(trust_map: TrustMap) -> bytes:
    """Encode trust values as a grayscale PNG where white means trusted."""
    image = (np.clip(trust_map.values, 0, 1) * 255).round().astype(np.uint8)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Unable to encode trust map")
    return encoded.tobytes()


def heatmap_overlay(image: np.ndarray, trust_map: TrustMap, regions: list[dict[str, Any]] | None = None) -> bytes:
    """Encode a colored trust heatmap overlay with optional suspicious boxes."""
    base = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    anomaly = ((1.0 - np.clip(trust_map.values, 0, 1)) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(anomaly, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base, 0.55, heatmap, 0.45, 0)
    for region in regions or []:
        x, y, width, height = (int(region[key]) for key in ("x", "y", "width", "height"))
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 255), 2)
    success, encoded = cv2.imencode(".png", overlay)
    if not success:
        raise ValueError("Unable to encode trust overlay")
    return encoded.tobytes()
