"""Run the ARGUS-AEGIS Phase 5 defense demonstration."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.models.model_registry import get_model
from app.pipeline.defense_pipeline import ArgusDefensePipeline
from app.utils.image import decode_image, encode_image
from app.utils.visualization import heatmap_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="ARGUS-AEGIS autonomous defense demo")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--attack-type", choices=["fgsm", "pgd", "patch", "unknown"], default="unknown")
    parser.add_argument("--patch-mask", type=Path, default=None)
    args = parser.parse_args()

    image = decode_image(args.image.read_bytes())
    patch_mask = None
    if args.patch_mask:
        patch_mask = cv2.cvtColor(decode_image(args.patch_mask.read_bytes()), cv2.COLOR_RGB2GRAY)
    result = ArgusDefensePipeline().run(image, get_model("resnet18"), patch_mask=patch_mask)
    output_dir = ROOT / "data" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    defended_path = output_dir / f"{stem}_defended.png"
    overlay_path = output_dir / f"{stem}_defense_overlay.png"
    trace_path = output_dir / f"{stem}_defense_trace.json"
    defended_path.write_bytes(encode_image(result.orchestration.defense.defended_image))
    overlay_path.write_bytes(heatmap_overlay(image, result.trust.trust_map, result.trust.localization.regions))
    trace_path.write_text(json.dumps(asdict(result.orchestration.trace), indent=2), encoding="utf-8")

    print("ARGUS-AEGIS Defense Demo")
    print("========================")
    print()
    print("Original Prediction:")
    print(f"    {result.original_prediction.class_name}")
    print("Original Confidence:")
    print(f"    {result.original_prediction.confidence:.2f}")
    print()
    print(f"Attack Score:\n    {result.detection.attack_score:.2f}")
    print(f"Global Trust:\n    {result.trust.trust_map.values.mean():.2f}")
    print(f"Suspicious Regions:\n    {len(result.trust.localization.regions)}")
    print("\n--------------------------------")
    print(f"Selected Defense:\n    {result.orchestration.decision.selected_defense}")
    print(f"Reason:\n    {result.orchestration.decision.reason}")
    print("\n--------------------------------")
    print(f"Defended Prediction:\n    {result.defended_prediction.class_name}")
    print(f"Defended Confidence:\n    {result.defended_prediction.confidence:.2f}")
    print(f"Prediction Recovered:\n    {'YES' if result.original_prediction.class_id == result.defended_prediction.class_id else 'NO'}")
    print(f"Defense Time:\n    {result.orchestration.defense.processing_time_ms:.2f} ms")
    print(f"\nDefended image saved to:\n    {defended_path}")
    print(f"Overlay saved to:\n    {overlay_path}")
    print(f"Trace saved to:\n    {trace_path}")


if __name__ == "__main__":
    main()
