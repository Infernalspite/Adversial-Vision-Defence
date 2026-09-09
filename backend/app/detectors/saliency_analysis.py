"""Gradient-based saliency evidence detector."""

import time
from typing import Any

import cv2
import numpy as np
import torch

from app.detectors.base_detector import BaseDetector, DetectorResult, clip_score, detector_context


class SaliencyAnalysisDetector(BaseDetector):
    """Create a lightweight input-gradient saliency map for the current class."""

    name = "saliency_analysis"

    def detect(self, image: np.ndarray, context: dict[str, Any] | None = None) -> DetectorResult:
        """Return saliency statistics and a map suitable for later trust-map use."""
        started = time.perf_counter()
        model, threshold = detector_context(context)
        source = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255).to(model.device)
        was_training = model.model.training
        model.model.eval()
        try:
            source.requires_grad_(True)
            with torch.enable_grad():
                logits = model.model(model.pixel_to_model_input(source).unsqueeze(0))
                class_id = int(logits.argmax(dim=1).item())
                gradient = torch.autograd.grad(logits[0, class_id], source)[0]
            saliency = gradient.abs().amax(dim=0).detach().cpu().numpy()
        finally:
            model.model.train(was_training)
        maximum = float(saliency.max())
        mean = float(saliency.mean())
        baseline = float(np.percentile(saliency, 50))
        scale = float(np.percentile(saliency, 95) - baseline)
        normalized = np.clip((saliency - baseline) / (scale + 1e-8), 0.0, 1.0)
        high_ratio = float((normalized >= 0.8).mean())
        concentration = float(np.percentile(saliency, 99) / (saliency.mean() + 1e-8))
        score = clip_score(0.5 * min(1.0, max(0.0, concentration - 1.0) / 8.0) + 0.5 * min(1.0, high_ratio * 20))
        elapsed = (time.perf_counter() - started) * 1000
        return DetectorResult(
            detector_name=self.name,
            score=score,
            detected=score >= threshold,
            confidence=score,
            evidence={
                "saliency_map": normalized.tolist(),
                "concentration": high_ratio,
                "maximum_saliency": maximum,
                "mean_saliency": mean,
                "high_saliency_area_ratio": high_ratio,
            },
            processing_time_ms=elapsed,
            metadata={"predicted_class_id": class_id, "map_height": int(normalized.shape[0]), "map_width": int(normalized.shape[1])},
        )
