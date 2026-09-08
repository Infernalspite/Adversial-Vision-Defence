"""Suspicious-region extraction from a spatial trust map."""

from dataclasses import dataclass

import cv2
import numpy as np

from app.trust.trust_map import TrustMap


@dataclass(frozen=True, slots=True)
class LocalizationResult:
    """Potential attack regions in image coordinates."""

    regions: list[dict[str, float | int]]
    threshold: float
    suspicious_area_percentage: float


class AttackLocalizer:
    """Locate connected low-trust regions with configurable filtering."""

    def __init__(self, threshold: float = 0.30, minimum_region_area: int = 16, max_regions: int = 50) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("localization threshold must be between 0 and 1")
        if minimum_region_area < 1 or max_regions < 1:
            raise ValueError("minimum_region_area and max_regions must be positive")
        self.threshold = threshold
        self.minimum_region_area = minimum_region_area
        self.max_regions = max_regions

    def locate(self, trust_map: TrustMap | np.ndarray) -> LocalizationResult:
        """Return suspicious connected regions sorted by anomaly score."""
        values = trust_map.values if isinstance(trust_map, TrustMap) else np.asarray(trust_map)
        values = np.nan_to_num(values.astype(np.float32), nan=1.0, posinf=1.0, neginf=1.0)
        suspicious = (values < self.threshold).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(suspicious, connectivity=8)
        height, width = values.shape[:2]
        regions: list[dict[str, float | int]] = []
        for label in range(1, count):
            x, y, region_width, region_height, area = stats[label]
            if int(area) < self.minimum_region_area:
                continue
            region_values = values[labels == label]
            trust_score = float(region_values.mean())
            regions.append({
                "region_id": 0,
                "x": int(x), "y": int(y), "width": int(region_width), "height": int(region_height),
                "area_pixels": int(area), "area_percentage": float(area / (height * width) * 100),
                "anomaly_score": float(1.0 - trust_score), "trust_score": trust_score,
            })
        regions.sort(key=lambda region: float(region["anomaly_score"]), reverse=True)
        for index, region in enumerate(regions[: self.max_regions], start=1):
            region["region_id"] = index
        regions = regions[: self.max_regions]
        total_area = sum(int(region["area_pixels"]) for region in regions)
        return LocalizationResult(regions, self.threshold, float(total_area / (height * width) * 100))
