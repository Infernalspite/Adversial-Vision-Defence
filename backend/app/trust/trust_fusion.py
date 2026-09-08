"""Spatial trust-map fusion configuration."""

from dataclasses import dataclass, field

import cv2
import numpy as np

from app.trust.trust_map import normalize_map


DEFAULT_TRUST_WEIGHTS = {"saliency": 0.30, "frequency": 0.25, "local": 0.20, "patch": 0.25}
DEFAULT_VERIFICATION_WEIGHTS = {"object": 0.40, "geometry": 0.30, "scene": 0.30}


@dataclass(frozen=True, slots=True)
class TrustFusionConfig:
    """MVP spatial evidence weights."""

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TRUST_WEIGHTS))

    def __post_init__(self) -> None:
        if not self.weights or any(value < 0 for value in self.weights.values()):
            raise ValueError("trust weights must be non-empty and non-negative")
        if sum(self.weights.values()) <= 0:
            raise ValueError("trust weights must have a positive sum")


@dataclass(frozen=True, slots=True)
class SpatialFusionResult:
    """Fused anomaly map and source bookkeeping."""

    anomaly_map: np.ndarray
    available_sources: tuple[str, ...]
    effective_weights: dict[str, float]


@dataclass(frozen=True, slots=True)
class VerificationFusionConfig:
    """Configurable semantic verification weights."""

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_VERIFICATION_WEIGHTS))

    def __post_init__(self) -> None:
        if not self.weights or any(value < 0 for value in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError("verification weights must be non-empty and non-negative")


@dataclass(frozen=True, slots=True)
class VerificationFusionResult:
    """Fused semantic consistency score."""

    score: float
    effective_weights: dict[str, float]


class TrustFusion:
    """Fuse available spatial anomaly maps with dynamic weight renormalization."""

    def __init__(self, config: TrustFusionConfig | None = None) -> None:
        self.config = config or TrustFusionConfig()

    def fuse(self, maps: dict[str, np.ndarray], target_shape: tuple[int, int]) -> SpatialFusionResult:
        """Return anomaly evidence; missing sources are excluded then renormalized."""
        available = {
            name: normalize_map(values)
            for name, values in maps.items()
            if values is not None and np.asarray(values).size
        }
        available = {name: values for name, values in available.items() if name in self.config.weights}
        if not available:
            return SpatialFusionResult(np.zeros(target_shape, dtype=np.float32), (), {})
        effective_total = sum(self.config.weights[name] for name in available)
        effective_weights = {name: self.config.weights[name] / effective_total for name in available}
        anomaly = np.zeros(target_shape, dtype=np.float32)
        for name, values in available.items():
            resized = cv2.resize(values, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
            anomaly += effective_weights[name] * resized
        return SpatialFusionResult(np.clip(anomaly, 0.0, 1.0), tuple(available), effective_weights)


class VerificationFusion:
    """Fuse semantic consistency evidence separately from spatial trust fusion."""

    def __init__(self, config: VerificationFusionConfig | None = None) -> None:
        self.config = config or VerificationFusionConfig()

    def fuse(self, scores: dict[str, float]) -> VerificationFusionResult:
        """Return a bounded semantic verification score."""
        available = {name: max(0.0, min(1.0, float(value))) for name, value in scores.items() if name in self.config.weights}
        if not available:
            return VerificationFusionResult(0.0, {})
        total = sum(self.config.weights[name] for name in available)
        weights = {name: self.config.weights[name] / total for name in available}
        score = sum(weights[name] * available[name] for name in available)
        return VerificationFusionResult(max(0.0, min(1.0, score)), weights)
