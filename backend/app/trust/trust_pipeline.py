"""Phase 4 spatial trust-map orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from app.detectors.detection_pipeline import DetectionPipeline, DetectionResult
from app.trust.localization import AttackLocalizer, LocalizationResult
from app.trust.trust_fusion import TrustFusion, TrustFusionConfig
from app.trust.trust_map import TrustMap, TrustMapBuilder, normalize_map


@dataclass(frozen=True, slots=True)
class TrustPipelineResult:
    """Complete Phase 4 result without a final trust state."""

    detection: DetectionResult
    trust_map: TrustMap
    localization: LocalizationResult
    spatial_evidence: dict[str, bool]
    overlay_regions: list[dict[str, float | int]]
    processing_time_ms: float


class TrustPipeline:
    """Reuse Phase 3 detector output and produce spatial perception confidence."""

    def __init__(
        self,
        detection_pipeline: DetectionPipeline | None = None,
        trust_fusion: TrustFusion | None = None,
    ) -> None:
        self.detection_pipeline = detection_pipeline or DetectionPipeline()
        self.trust_fusion = trust_fusion or TrustFusion()

    def analyze(
        self,
        image: np.ndarray,
        model: object,
        detector_names: list[str] | None = None,
        threshold: float | None = None,
        trust_weights: dict[str, float] | None = None,
        localization_threshold: float = 0.30,
        minimum_region_area: int = 16,
        patch_mask: np.ndarray | None = None,
        detection: DetectionResult | None = None,
    ) -> TrustPipelineResult:
        """Run or reuse Phase 3, fuse spatial evidence, and localize low-trust regions."""
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Expected an RGB uint8 image")
        started = time.perf_counter()
        detection_result = detection or self.detection_pipeline.analyze(image, model, detector_names, threshold)
        if not detection_result.attack_detected and patch_mask is None:
            height, width = image.shape[:2]
            trust_map = TrustMap(
                values=np.ones((height, width), dtype=np.float32),
                width=width,
                height=height,
                resolution=(height, width),
                normalization={"input_range": "[0, 1]", "policy": "no attack detected"},
                generation_time_ms=(time.perf_counter() - started) * 1000,
                contributing_detectors=(),
                metadata={"available_sources": (), "weights": {}},
            )
            localization = AttackLocalizer(localization_threshold, minimum_region_area).locate(trust_map)
            return TrustPipelineResult(detection_result, trust_map, localization, {"saliency_available": False, "frequency_available": False, "local_anomaly_available": False, "patch_mask_available": False}, [], (time.perf_counter() - started) * 1000)
        maps = self._spatial_maps(image, detection_result, patch_mask)
        fusion = TrustFusion(TrustFusionConfig(trust_weights or self.trust_fusion.config.weights)).fuse(
            maps, image.shape[:2]
        )
        builder = TrustMapBuilder()
        trust_map = builder.build(
            image,
            {"fused": fusion.anomaly_map},
            weights={"fused": 1.0},
            generation_time_ms=(time.perf_counter() - started) * 1000,
        )
        localizer = AttackLocalizer(localization_threshold, minimum_region_area)
        localization = localizer.locate(trust_map)
        return TrustPipelineResult(
            detection=detection_result,
            trust_map=trust_map,
            localization=localization,
            spatial_evidence={
                "saliency_available": "saliency" in maps,
                "frequency_available": "frequency" in maps,
                "local_anomaly_available": "local" in maps,
                "patch_mask_available": "patch" in maps,
            },
            overlay_regions=localization.regions,
            processing_time_ms=(time.perf_counter() - started) * 1000,
        )

    @staticmethod
    def _spatial_maps(
        image: np.ndarray,
        detection: DetectionResult,
        patch_mask: np.ndarray | None,
    ) -> dict[str, np.ndarray]:
        """Build spatial evidence maps from Phase 3 output and image-local statistics."""
        maps: dict[str, np.ndarray] = {}
        saliency = detection.detectors.get("saliency_analysis")
        if saliency is not None and "saliency_map" in saliency.evidence:
            maps["saliency"] = np.asarray(saliency.evidence["saliency_map"], dtype=np.float32)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        maps["frequency"] = normalize_map(laplacian)
        local_mean = cv2.blur(gray, (9, 9))
        local_deviation = np.abs(gray - local_mean)
        maps["local"] = normalize_map(local_deviation)
        if patch_mask is not None:
            if patch_mask.ndim == 3:
                patch_mask = cv2.cvtColor(patch_mask, cv2.COLOR_RGB2GRAY)
            maps["patch"] = (patch_mask > 0).astype(np.float32)
        return maps