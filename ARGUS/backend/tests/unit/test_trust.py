import numpy as np
import pytest

from app.trust.evaluation import evaluate_localization
from app.trust.localization import AttackLocalizer
from app.trust.trust_fusion import TrustFusion, TrustFusionConfig
from app.trust.trust_map import TrustMap, normalize_map, resize_map


def test_normalize_map_handles_constant_nonfinite_and_empty_inputs():
    assert np.array_equal(normalize_map(np.ones((2, 2))), np.zeros((2, 2), dtype=np.float32))
    normalized = normalize_map(np.array([[np.nan, np.inf], [0.0, 2.0]]))
    assert np.isfinite(normalized).all()
    assert normalized.min() >= 0 and normalized.max() <= 1
    assert normalize_map(np.array([])).size == 0


def test_resize_map_reaches_target_shape_and_binary_mode():
    resized = resize_map(np.zeros((2, 2), dtype=np.float32), (8, 6))
    assert resized.shape == (8, 6)
    binary = resize_map(np.array([[0, 1], [1, 0]], dtype=np.float32), (8, 8), binary=True)
    assert set(np.unique(binary)) <= {0.0, 1.0}


def test_fusion_respects_weights_and_renormalizes_missing_sources():
    fusion = TrustFusion(TrustFusionConfig({"saliency": 0.75, "patch": 0.25}))
    result = fusion.fuse({"saliency": np.ones((2, 2))}, (4, 4))
    assert np.allclose(result.anomaly_map, 0)
    assert result.effective_weights == {"saliency": 1.0}


def test_localization_finds_and_sorts_suspicious_region():
    values = np.ones((20, 20), dtype=np.float32)
    values[5:10, 7:14] = 0.1
    trust_map = TrustMap(values, 20, 20, (20, 20), {}, 0, ("synthetic",))
    result = AttackLocalizer(threshold=0.3, minimum_region_area=4).locate(trust_map)
    assert len(result.regions) == 1
    region = result.regions[0]
    assert (region["x"], region["y"], region["width"], region["height"]) == (7, 5, 7, 5)
    assert region["anomaly_score"] == pytest.approx(0.9)


def test_localization_minimum_area_filters_small_regions():
    values = np.ones((10, 10), dtype=np.float32)
    values[2, 2] = 0
    trust_map = TrustMap(values, 10, 10, (10, 10), {}, 0, ())
    assert not AttackLocalizer(minimum_region_area=2).locate(trust_map).regions


def test_localization_evaluation_uses_ground_truth_mask():
    values = np.ones((4, 4), dtype=np.float32)
    values[1:3, 1:3] = 0.1
    trust_map = TrustMap(values, 4, 4, (4, 4), {}, 0, ())
    metrics = evaluate_localization(trust_map, (values < 0.3).astype(np.uint8))
    assert metrics.intersection_over_union == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
