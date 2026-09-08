import numpy as np
import pytest

from app.models.base_model import TopPrediction, ModelPrediction
from app.trust.trust_fusion import VerificationFusion, VerificationFusionConfig
from app.verification.geometry_consistency import GeometryConsistencyVerifier
from app.verification.object_consistency import ObjectConsistencyVerifier
from app.verification.scene_consistency import SceneConsistencyVerifier
from app.verification.semantic_anchor import SemanticRealityAnchor
from app.verification.verification_pipeline import VerificationPipeline


def prediction(class_id: int, confidence: float, top_ids: tuple[int, ...] = (1, 2, 3)) -> ModelPrediction:
    return ModelPrediction(class_id, str(class_id), confidence, tuple(TopPrediction(item, str(item), confidence) for item in top_ids), 1.0)


def test_object_consistency_scores_identical_predictions_higher():
    verifier = ObjectConsistencyVerifier()
    same = verifier.verify(prediction(1, 0.9), prediction(1, 0.8))
    different = verifier.verify(prediction(1, 0.9), prediction(4, 0.8, (4, 5, 6)))
    assert same["object_consistency_score"] > different["object_consistency_score"]
    assert same["top_k_overlap"] == 1


def test_geometry_identical_images_are_consistent():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    result = GeometryConsistencyVerifier().verify(image, image.copy())
    altered = GeometryConsistencyVerifier().verify(image, np.full_like(image, 255))
    assert result["structural_similarity"] == pytest.approx(1.0)
    assert result["structural_similarity"] > altered["structural_similarity"]
    assert 0 <= altered["structural_similarity"] <= 1


def test_scene_identical_images_are_consistent():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    result = SceneConsistencyVerifier().verify(image, image.copy(), prediction(1, 0.9), prediction(1, 0.8))
    altered = SceneConsistencyVerifier().verify(image, np.full_like(image, 255), prediction(1, 0.9), prediction(4, 0.8))
    assert result["scene_similarity_score"] > altered["scene_similarity_score"]
    assert 0 <= altered["scene_similarity_score"] <= 1


def test_verification_fusion_is_configurable_and_bounded():
    fusion = VerificationFusion(VerificationFusionConfig({"object": 0.5, "geometry": 0.5}))
    result = fusion.fuse({"object": 1.0, "geometry": 0.5})
    assert result.score == pytest.approx(0.75)
    assert result.effective_weights == {"object": 0.5, "geometry": 0.5}


def test_semantic_anchor_and_pipeline_return_structured_evidence():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    before = prediction(1, 0.9)
    after = prediction(1, 0.8)
    result = SemanticRealityAnchor().verify(image, image.copy(), before, after)
    assert 0 <= result.verification_score <= 1
    assert result.explanation
    pipeline = VerificationPipeline().run(image, image.copy(), before, after)
    assert pipeline.prediction_changed is False
    assert pipeline.top_k_overlap == 1
