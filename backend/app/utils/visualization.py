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


def suspicion_heatmap_png(trust_map: TrustMap) -> bytes:
    """Encode the raw suspicion map (1 - trust) as a jet colormap PNG.

    This is the standalone pixel-trace artifact: which areas look manipulated,
    independent of the underlying photo.
    """
    anomaly = ((1.0 - np.clip(trust_map.values, 0, 1)) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(anomaly, cv2.COLORMAP_JET)
    success, encoded = cv2.imencode(".png", heatmap)
    if not success:
        raise ValueError("Unable to encode suspicion heatmap")
    return encoded.tobytes()


def pixel_trace_summary(trust_map: TrustMap, regions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Compact numeric trace of suspicion across the image."""
    suspicion = 1.0 - np.clip(trust_map.values, 0, 1)
    areas = [float(region.get("area", 0)) for region in (regions or [])]
    total_area = float(trust_map.values.size)
    resolution = trust_map.resolution
    # TrustMap.resolution is a (rows, cols) grid-shape tuple; the API field
    # names are (width, height), so the order is swapped deliberately here.
    cell_height, cell_width = (resolution if isinstance(resolution, tuple) else (resolution, resolution))
    return {
        "suspicion_mean": float(suspicion.mean()),
        "suspicion_max": float(suspicion.max()),
        "suspicious_pixel_fraction": float((suspicion >= 0.5).mean()),
        "num_suspicious_regions": len(regions or []),
        "largest_region_area_fraction": (max(areas) / total_area) if areas else 0.0,
        "trust_resolution": {
            "width": float(trust_map.width),
            "height": float(trust_map.height),
            "cell_width": float(cell_width),
            "cell_height": float(cell_height),
        },
    }


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
