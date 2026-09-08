"""Frequency-domain evidence detector."""

import time
from typing import Any

import cv2
import numpy as np

from app.detectors.base_detector import BaseDetector, DetectorResult, clip_score, detector_context


class FrequencyAnalysisDetector(BaseDetector):
    """Measure high-frequency energy as evidence, not proof, of manipulation."""

    name = "frequency_analysis"

    def detect(self, image: np.ndarray, context: dict[str, Any] | None = None) -> DetectorResult:
        """Calculate low/high FFT energy and a bounded anomaly score."""
        started = time.perf_counter()
        _, threshold = detector_context(context)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        spectrum = np.fft.fftshift(np.fft.fft2(gray))
        power = np.abs(spectrum) ** 2
        height, width = gray.shape
        y, x = np.ogrid[:height, :width]
        radius = np.sqrt((x - width / 2) ** 2 + (y - height / 2) ** 2)
        cutoff = min(height, width) * 0.18
        low_energy = float(power[radius <= cutoff].mean())
        high_energy = float(power[radius > cutoff].mean())
        ratio = high_energy / (low_energy + 1e-8)
        score = clip_score(ratio / (ratio + 1.0))
        elapsed = (time.perf_counter() - started) * 1000
        return DetectorResult(
            detector_name=self.name,
            score=score,
            detected=score >= threshold,
            confidence=score,
            evidence={"high_frequency_energy": high_energy, "low_frequency_energy": low_energy, "high_low_ratio": ratio},
            processing_time_ms=elapsed,
            metadata={"cutoff_fraction": 0.18, "spectrum_mean": float(power.mean())},
        )
