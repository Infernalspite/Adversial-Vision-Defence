"""Lightweight image-structure consistency evidence."""

from typing import Any

import cv2
import numpy as np


class GeometryConsistencyVerifier:
    """Compare edge structure and normalized pixel similarity."""

    def verify(self, original: np.ndarray, defended: np.ndarray) -> dict[str, Any]:
        """Return structural similarity evidence in the [0, 1] range."""
        if original.shape != defended.shape:
            defended = cv2.resize(defended, (original.shape[1], original.shape[0]))
        original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        defended_gray = cv2.cvtColor(defended, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        pixel_similarity = float(1 - np.abs(original_gray - defended_gray).mean())
        original_edges = cv2.Canny((original_gray * 255).astype(np.uint8), 80, 160) > 0
        defended_edges = cv2.Canny((defended_gray * 255).astype(np.uint8), 80, 160) > 0
        union = np.logical_or(original_edges, defended_edges).sum()
        edge_similarity = float(np.logical_and(original_edges, defended_edges).sum() / union) if union else 1.0
        score = max(0.0, min(1.0, 0.65 * pixel_similarity + 0.35 * edge_similarity))
        return {
            "structural_similarity": score,
            "pixel_similarity": pixel_similarity,
            "edge_similarity": edge_similarity,
            "edge_difference": float(np.logical_xor(original_edges, defended_edges).mean()),
        }
