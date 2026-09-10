from pathlib import Path

from app.learning.splits import assert_disjoint, build_split_manifest, load_split_manifest

TEST_CLASS_MAP = {f"n0144076{i}": f"class_{i}" for i in range(3)}


def make_source_tree(root: Path, classes: int = 3, images: int = 30) -> Path:
    for class_index in range(classes):
        class_dir = root / f"n0144076{class_index}"
        class_dir.mkdir(parents=True)
        for image_index in range(images):
            # Unique bytes per image: the md5-based disjointness check must
            # never see duplicate content across splits.
            (class_dir / f"img_{image_index:03d}.jpg").write_bytes(f"fake-image-{class_index}-{image_index}".encode())
    return root


def test_split_manifest_counts_and_materialization(tmp_path):
    source = make_source_tree(tmp_path / "source")
    output = tmp_path / "splits"
    manifest = build_split_manifest(source, output, per_class={"train": 8, "val": 3, "test": 3}, seed=7, class_names=TEST_CLASS_MAP)
    assert manifest["counts"]["train"]["total"] == 24
    assert manifest["counts"]["val"]["total"] == 9
    assert manifest["counts"]["test"]["total"] == 9
    assert all(count == 8 for count in manifest["counts"]["train"]["per_class"].values())
    for split in ("train", "val", "test"):
        split_dir = output / split
        files = list(split_dir.rglob("*.jpg"))
        assert len(files) == manifest["counts"][split]["total"]
        assert all(path.exists() for path in files)
    assert_disjoint(manifest)


def test_split_manifest_is_deterministic(tmp_path):
    source = make_source_tree(tmp_path / "source")
    first = build_split_manifest(source, tmp_path / "splits_a", per_class={"train": 8, "val": 3, "test": 3}, seed=7, class_names=TEST_CLASS_MAP)
    second = build_split_manifest(source, tmp_path / "splits_b", per_class={"train": 8, "val": 3, "test": 3}, seed=7, class_names=TEST_CLASS_MAP)
    assert first["assignments"] == second["assignments"]
    loaded = load_split_manifest(tmp_path / "splits_a" / "manifest.json")
    assert loaded["seed"] == 7


def test_disjointness_detects_leakage():
    manifest = {"assignments": [
        {"md5": "a", "split": "train"},
        {"md5": "a", "split": "test"},
    ]}
    try:
        assert_disjoint(manifest)
    except ValueError as error:
        assert "train" in str(error) and "test" in str(error)
    else:
        raise AssertionError("Leakage must be detected")


def test_unknown_class_is_rejected(tmp_path):
    source = tmp_path / "source"
    (source / "n99999999").mkdir(parents=True)
    (source / "n99999999" / "x.jpg").write_bytes(b"x")
    try:
        build_split_manifest(source, tmp_path / "out", per_class={"train": 1, "val": 1, "test": 1}, class_names=TEST_CLASS_MAP)
    except ValueError as error:
        assert "Unknown classes" in str(error)
    else:
        raise AssertionError("Unknown classes must be rejected")
