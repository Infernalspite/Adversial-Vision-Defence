"""Semantic Reality Anchor orchestration."""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.models.base_model import ModelPrediction
from app.trust.trust_fusion import VerificationFusion
from app.verification.geometry_consistency import GeometryConsistencyVerifier
from app.verification.object_consistency import ObjectConsistencyVerifier
from app.verification.scene_consistency import SceneConsistencyVerifier


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Independent semantic and visual verification evidence."""

    verification_score: float
    object_consistency: dict[str, Any]
    geometry_consistency: dict[str, Any]
    scene_consistency: dict[str, Any]
    explanation: str
    processing_time_ms: float


class SemanticRealityAnchor:
    """Coordinate object, geometry, and scene evidence without final decisions."""

    def __init__(
        self,
        object_verifier: ObjectConsistencyVerifier | None = None,
        geometry_verifier: GeometryConsistencyVerifier | None = None,
        scene_verifier: SceneConsistencyVerifier | None = None,
        fusion: VerificationFusion | None = None,
    ) -> None:
        self.object_verifier = object_verifier or ObjectConsistencyVerifier()
        self.geometry_verifier = geometry_verifier or GeometryConsistencyVerifier()
        self.scene_verifier = scene_verifier or SceneConsistencyVerifier()
        self.fusion = fusion or VerificationFusion()

    def verify(
        self,
        original: np.ndarray,
        defended: np.ndarray,
        original_prediction: ModelPrediction,
        defended_prediction: ModelPrediction,
        context: dict[str, Any] | None = None,
    ) -> VerificationResult:
        """Compare interpretations and structure, then produce evidence-based explanation."""
        started = time.perf_counter()
        object_result = self.object_verifier.verify(original_prediction, defended_prediction)
        geometry_result = self.geometry_verifier.verify(original, defended)
        scene_result = self.scene_verifier.verify(original, defended, original_prediction, defended_prediction)
        score = self.fusion.fuse({
            "object": object_result["object_consistency_score"],
            "geometry": geometry_result["structural_similarity"],
            "scene": scene_result["scene_similarity_score"],
        }).score
        clauses = [
            "The defended interpretation remained semantically consistent." if object_result["object_consistency_score"] >= 0.6 else "Object interpretation changed or became less consistent.",
            "Image structure remained stable." if geometry_result["structural_similarity"] >= 0.6 else "Image structure changed materially.",
            "Coarse scene characteristics remained stable." if scene_result["scene_similarity_score"] >= 0.6 else "Coarse scene characteristics changed materially.",
        ]
        return VerificationResult(
            verification_score=max(0.0, min(1.0, score)),
            object_consistency=object_result,
            geometry_consistency=geometry_result,
            scene_consistency=scene_result,
            explanation=" ".join(clauses),
            processing_time_ms=(time.perf_counter() - started) * 1000,
        )
