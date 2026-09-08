"""Run the complete ARGUS-AEGIS Phase 3-7 analysis."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.models.model_registry import get_model
from app.pipeline.pipeline import ArgusPipeline, PipelineContext
from app.utils.image import decode_image, encode_image
from app.utils.visualization import heatmap_overlay, trust_map_png


def main() -> None:
    parser = argparse.ArgumentParser(description="ARGUS-AEGIS complete analysis")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--mode", choices=["standard", "evaluation"], default="standard")
    parser.add_argument("--attack-type", default=None)
    args = parser.parse_args()

    image = decode_image(args.image.read_bytes())
    result = ArgusPipeline().run(PipelineContext(
        request_id=args.image.stem,
        image=image,
        metadata={"model": get_model("resnet18"), "mode": args.mode, "attack_type": args.attack_type},
    ))
    defense = result.defense
    verification = result.verification.verification
    output_dir = ROOT / "data" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    (output_dir / f"{stem}_analysis_defended.png").write_bytes(encode_image(defense.orchestration.defense.defended_image))
    (output_dir / f"{stem}_analysis_trust.png").write_bytes(trust_map_png(defense.trust.trust_map))
    (output_dir / f"{stem}_analysis_overlay.png").write_bytes(heatmap_overlay(image, defense.trust.trust_map, defense.trust.localization.regions))

    print("=" * 50)
    print("             ARGUS-AEGIS ANALYSIS")
    print("=" * 50)
    print(f"\nINPUT\nImage: {args.image.name}")
    print("\n--------------------------------------------------\nBASE MODEL\n--------------------------------------------------")
    print(f"\nPrediction:        {defense.original_prediction.class_name}\nConfidence:        {defense.original_prediction.confidence:.2f}")
    print("\n--------------------------------------------------\nADVERSARIAL DETECTION\n--------------------------------------------------")
    print(f"\nAttack Score:      {defense.detection.attack_score:.2f}\nAttack Detected:   {'YES' if defense.detection.attack_detected else 'NO'}")
    print("\n--------------------------------------------------\nSPATIAL TRUST\n--------------------------------------------------")
    print(f"\nGlobal Trust:      {defense.trust.trust_map.values.mean():.2f}\nSuspicious Areas:  {len(defense.trust.localization.regions)}")
    print("\n--------------------------------------------------\nDEFENSE\n--------------------------------------------------")
    print(f"\nDefense Applied:   {'YES' if defense.orchestration.defense.defense_applied else 'NO'}\nStrategy:          {defense.orchestration.decision.selected_defense}\nReason:            {defense.orchestration.decision.reason}")
    print("\n--------------------------------------------------\nRE-INFERENCE\n--------------------------------------------------")
    print(f"\nOriginal:          {defense.original_prediction.class_name} ({defense.original_prediction.confidence:.2f})\nDefended:          {defense.defended_prediction.class_name} ({defense.defended_prediction.confidence:.2f})")
    print("\n--------------------------------------------------\nSEMANTIC VERIFICATION\n--------------------------------------------------")
    print(f"\nObject:            {verification.object_consistency['object_consistency_score']:.2f}\nGeometry:          {verification.geometry_consistency['structural_similarity']:.2f}\nScene:             {verification.scene_consistency['scene_similarity_score']:.2f}\n\nVerification:      {verification.verification_score:.2f}")
    print("\n--------------------------------------------------\nFINAL DECISION\n--------------------------------------------------")
    print(f"\nSTATE:             {result.decision.final_state.value}\nPrediction:        {result.decision.final_prediction or 'NONE'}\nConfidence:        {result.decision.final_confidence if result.decision.final_confidence is not None else 'NONE'}")
    print("\nReason:")
    for reason in result.decision.decision_reasons:
        print(f"- {reason}")
    if result.fallback:
        print(f"\nFallback: {result.fallback.action} - {result.fallback.reason}")
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
