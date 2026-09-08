"""Run the ARGUS-AEGIS Phase 4 spatial trust-map demo."""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.models.model_registry import get_model
from app.trust.trust_pipeline import TrustPipeline
from app.utils.image import decode_image
from app.utils.visualization import heatmap_overlay, trust_map_png


def main() -> None:
    parser = argparse.ArgumentParser(description="ARGUS-AEGIS spatial trust-map demo")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--attack-type", choices=["clean", "fgsm", "pgd", "patch"], default="clean")
    parser.add_argument("--patch-mask", type=Path, default=None)
    args = parser.parse_args()

    image = decode_image(args.image.read_bytes())
    model = get_model("resnet18")
    mask = None
    if args.patch_mask:
        mask = cv2.cvtColor(decode_image(args.patch_mask.read_bytes()), cv2.COLOR_RGB2GRAY)
    result = TrustPipeline().analyze(image, model, patch_mask=mask)

    output_dir = ROOT / "data" / "adversarial"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.image.stem}_trust_map"
    trust_path = output_dir / f"{stem}.png"
    overlay_path = output_dir / f"{stem}_overlay.png"
    trust_path.write_bytes(trust_map_png(result.trust_map))
    overlay_path.write_bytes(heatmap_overlay(image, result.trust_map, result.localization.regions))

    print("ARGUS-AEGIS Spatial Trust Map")
    print("=============================")
    print()
    print(f"Global Trust Score: {result.trust_map.values.mean():.2f}")
    print(f"Suspicious Regions: {len(result.localization.regions)}")
    for index, region in enumerate(result.localization.regions, start=1):
        print(f"\nRegion {index}:")
        print(f"  Location: ({region['x']}, {region['y']})")
        print(f"  Size: {region['width']}x{region['height']}")
        print(f"  Area: {region['area_percentage']:.2f}%")
        print(f"  Trust: {region['trust_score']:.2f}")
        print(f"  Anomaly: {region['anomaly_score']:.2f}")
    print(f"\nTrust map saved to: {trust_path}")
    print(f"Overlay saved to: {overlay_path}")


if __name__ == "__main__":
    main()
