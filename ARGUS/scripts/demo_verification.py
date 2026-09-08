"""Run Detection -> Trust -> Defense -> Verification."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.models.model_registry import get_model
from app.pipeline.defense_pipeline import ArgusDefensePipeline
from app.verification.verification_pipeline import VerificationPipeline
from app.utils.image import decode_image, encode_image
from app.utils.visualization import heatmap_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="ARGUS-AEGIS semantic verification demo")
    parser.add_argument("--image", required=True, type=Path)
    args = parser.parse_args()

    image = decode_image(args.image.read_bytes())
    model = get_model("resnet18")
    defense = ArgusDefensePipeline().run(image, model)
    defended_image = defense.orchestration.defense.defended_image
    verification = VerificationPipeline().run(
        image,
        defended_image,
        defense.original_prediction,
        defense.defended_prediction,
        defense.trust.trust_map,
        defense.trust.localization.regions,
        defense.detection,
    )
    output_dir = ROOT / "data" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    defended_path = output_dir / f"{stem}_verified_defended.png"
    overlay_path = output_dir / f"{stem}_verified_overlay.png"
    result_path = output_dir / f"{stem}_verification.json"
    defended_path.write_bytes(encode_image(defended_image))
    overlay_path.write_bytes(heatmap_overlay(image, defense.trust.trust_map, defense.trust.localization.regions))
    result_path.write_text(json.dumps({
        "verification_score": verification.verification.verification_score,
        "object_consistency": verification.verification.object_consistency,
        "geometry_consistency": verification.verification.geometry_consistency,
        "scene_consistency": verification.verification.scene_consistency,
        "explanation": verification.verification.explanation,
    }, indent=2), encoding="utf-8")

    anchor = verification.verification
    print("ARGUS-AEGIS Verification Demo")
    print("=============================")
    print()
    print(f"Original Prediction:\n    {defense.original_prediction.class_name}")
    print(f"Original Confidence:\n    {defense.original_prediction.confidence:.2f}")
    print(f"Attack Score:\n    {defense.detection.attack_score:.2f}")
    print(f"Global Trust:\n    {defense.trust.trust_map.values.mean():.2f}")
    print(f"Defense:\n    {defense.orchestration.decision.selected_defense}")
    print(f"Defended Prediction:\n    {defense.defended_prediction.class_name}")
    print(f"Defended Confidence:\n    {defense.defended_prediction.confidence:.2f}")
    print("\n--------------------------------")
    print("SEMANTIC REALITY ANCHOR")
    print(f"\nObject Consistency:\n    {anchor.object_consistency['object_consistency_score']:.2f}")
    print(f"Geometry Consistency:\n    {anchor.geometry_consistency['structural_similarity']:.2f}")
    print(f"Scene Consistency:\n    {anchor.scene_consistency['scene_similarity_score']:.2f}")
    print(f"Verification Score:\n    {anchor.verification_score:.2f}")
    print(f"\nPrediction Recovered:\n    {'YES' if not verification.prediction_changed else 'NO'}")
    print(f"\nDefended image saved to:\n    {defended_path}")
    print(f"Overlay saved to:\n    {overlay_path}")
    print(f"Verification saved to:\n    {result_path}")


if __name__ == "__main__":
    main()
