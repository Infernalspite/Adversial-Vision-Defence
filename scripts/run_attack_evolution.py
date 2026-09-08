"""Run a bounded offline attack evolution experiment."""

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "backend"))
from app.learning.evolution_engine import EvolutionEngine
from app.models.model_registry import get_model
from app.utils.image import decode_image

parser = argparse.ArgumentParser(); parser.add_argument("--image", type=Path, required=True); parser.add_argument("--attack", choices=["fgsm", "pgd", "patch"], default="pgd"); parser.add_argument("--generations", type=int, default=3); parser.add_argument("--population", type=int, default=20); parser.add_argument("--seed", type=int, default=42); args = parser.parse_args()
image = decode_image(args.image.read_bytes()); report = EvolutionEngine(args.seed).run(image, get_model("resnet18"), args.attack, args.generations, args.population); print({"experiment_id": report["experiment_id"], "status": report["status"], "production_modified": report["production_modified"]})
