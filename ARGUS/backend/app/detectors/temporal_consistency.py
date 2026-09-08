"""Temporal consistency detector placeholder."""

from typing import Any

from app.detectors.base_detector import BaseDetector, DetectorResult


class TemporalConsistencyDetector(BaseDetector):
    """Future detector for frame-to-frame consistency."""

    def detect(self, image: Any, context: dict[str, Any] | None = None) -> DetectorResult:
        """Return temporal evidence."""
        raise NotImplementedError
