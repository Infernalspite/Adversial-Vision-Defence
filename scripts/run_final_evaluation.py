"""Final held-out evaluation: one pass on the test split, full defense chain.

Runs classify -> detect -> localize -> purify -> re-detect -> re-classify for
every test image (clean + every attacked config), then aggregates metrics by
attack family and strength and renders docs/robustness_report.md.

The test split is consumed exactly once: nothing in this script tunes
weights, thresholds, or models.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.evaluation.defense_metrics import defense_metrics  # noqa: E402
from app.evaluation.decision_metrics import decision_metrics  # noqa: E402
from app.evaluation.detection_metrics import detection_metrics  # noqa: E402
from app.evaluation.localization_metrics import localization_metrics  # noqa: E402
from app.evaluation.performance_metrics import latency_metrics  # noqa: E402
from app.evaluation.report import render_robustness_report  # noqa: E402
from app.learning.splits import load_split_manifest  # noqa: E402
from app.models.model_registry import get_model  # noqa: E402
from app.pipeline.pipeline import ArgusPipeline, PipelineContext  # noqa: E402
from app.detectors.detection_pipeline import DetectionPipeline  # noqa: E402
from app.utils.image import decode_image  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def iter_class_images(root: Path):
    if not root.exists():
        return
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for image_path in sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
            yield class_dir.name, image_path


def resolve_clean_reference(splits_root: Path, class_name: str, stem: str) -> Path | None:
    """Find the clean test-split source image for an attacked file stem."""
    class_dir = splits_root / "test" / class_name
    if not class_dir.exists():
        return None
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = class_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def run_record(pipeline: ArgusPipeline, model, image, clean_image, request_id: str, patch_mask) -> dict[str, Any]:
    """Run the exact production composite pipeline (detection, trust, defense,
    verification, decision) on one image and flatten the evidence the report
    aggregates. Uses the same stages as POST /api/v1/analyze so the report
    reflects deployed behavior, not a parallel implementation.
    """
    clean_prediction = model.predict(clean_image)

    started = time.perf_counter()
    context = PipelineContext(request_id, image, {"model": model, "mode": "evaluation"})
    result = pipeline.run(context)
    total_ms = (time.perf_counter() - started) * 1000
    defense = result.defense
    adversarial_prediction = defense.original_prediction
    defended_prediction = defense.defended_prediction
    detection = defense.detection
    decision = result.decision
    return {
        "record": {
            "clean_prediction": clean_prediction.class_name,
            "clean_confidence": clean_prediction.confidence,
            "adversarial_prediction": adversarial_prediction.class_name,
            "adversarial_confidence": adversarial_prediction.confidence,
            "defended_prediction": defended_prediction.class_name,
            "defended_confidence": defended_prediction.confidence,
            "attack_score": detection.attack_score,
            "attack_detected": detection.attack_detected,
            "detector_scores": {name: detector.score for name, detector in detection.detectors.items()},
            "final_state": decision.final_state.value if hasattr(decision.final_state, "value") else str(decision.final_state),
            "defense_method": defense.orchestration.decision.selected_defense,
            "verification_score": result.verification.verification.verification_score,
            "latency_ms": total_ms,
        },
        "prediction_changed": adversarial_prediction.class_id != clean_prediction.class_id,
        "defended_correct": defended_prediction.class_id == clean_prediction.class_id,
        "trust_values": defense.trust.trust_map.values,
        "trust_shape": defense.trust.trust_map.values.shape,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="One-pass final evaluation on the test split.")
    parser.add_argument("--model-tag", choices=["baseline", "robust"], default="robust")
    parser.add_argument("--splits-root", type=Path, default=ROOT / "data" / "robust" / "splits")
    parser.add_argument("--attacks-root", type=Path, default=ROOT / "data" / "robust" / "attacks")
    parser.add_argument("--results-root", type=Path, default=ROOT / "data" / "robust" / "final_results")
    parser.add_argument("--max-clean-per-class", type=int, default=25)
    parser.add_argument("--max-attack-per-class", type=int, default=None)
    parser.add_argument("--skip-zoo", action="store_true", help="Exclude unseen zoo attack families")
    parser.add_argument("--skip-clean", action="store_true")
    args = parser.parse_args()

    model = get_model("resnet18" if args.model_tag == "baseline" else "robust")
    pipeline = ArgusPipeline()
    manifest = load_split_manifest(args.splits_root / "manifest.json")
    test_stems = {
        (entry["class_name"], Path(entry["source_path"]).stem)
        for entry in manifest["assignments"] if entry["split"] == "test"
    }

    results_root = args.results_root
    results_root.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []

    # ---- clean side (accuracy + FPR) ----
    if not args.skip_clean:
        count: dict[str, int] = {}
        for class_name, image_path in iter_class_images(args.splits_root / "test"):
            if count.get(class_name, 0) >= args.max_clean_per_class:
                continue
            image = decode_image(image_path.read_bytes())
            outcome = run_record(pipeline, model, image, image, f"clean-{image_path.stem}", None)
            record = outcome["record"] | {
                "sample": image_path.name, "class_name": class_name, "attack_family": "clean",
                "strength": "-", "is_adversarial": False, "attack_success": False,
                "prediction_changed": False, "defense_recovered": False, "defense_acceptable": True,
                "clean_correct": outcome["record"]["clean_prediction"] == class_name,
                "robust_correct": outcome["record"]["adversarial_prediction"] == class_name,
                "post_defense_correct": outcome["defended_correct"],
                "ground_truth_mask": None,
            }
            all_records.append(record)
            count[class_name] = count.get(class_name, 0) + 1

    # ---- attacked side ----
    attack_root = args.attacks_root / args.model_tag / "test"
    if not attack_root.exists():
        raise SystemExit(f"No test attack suite for '{args.model_tag}' at {attack_root}; run generate_attack_suite.py first")
    attack_count: dict[str, dict[str, int]] = {}
    for config_dir in sorted(p for p in attack_root.iterdir() if p.is_dir()):
        family = config_dir.name.split("_e")[0].split("_s")[0]
        strength = config_dir.name
        for class_name, image_path in iter_class_images(config_dir):
            per_config = attack_count.setdefault(config_dir.name, {})
            if args.max_attack_per_class is not None and per_config.get(class_name, 0) >= args.max_attack_per_class:
                continue
            image = decode_image(image_path.read_bytes())
            clean_reference = resolve_clean_reference(args.splits_root, class_name, image_path.stem)
            clean_image = decode_image(clean_reference.read_bytes()) if clean_reference else image
            mask_path = image_path.with_name(f"{image_path.stem}_mask.png")
            patch_mask = None
            ground_truth_mask = None
            if mask_path.exists():
                ground_truth_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
            outcome = run_record(pipeline, model, image, clean_image, f"{config_dir.name}-{image_path.stem}", patch_mask)
            record = outcome["record"] | {
                "sample": image_path.name, "class_name": class_name, "attack_family": family,
                "strength": strength, "is_adversarial": True,
                "attack_success": outcome["prediction_changed"],
                "prediction_changed": outcome["prediction_changed"],
                "defense_recovered": outcome["defended_correct"] and outcome["prediction_changed"],
                "defense_acceptable": outcome["record"]["final_state"] != "ABSTAIN",
                "clean_correct": outcome["record"]["clean_prediction"] == class_name,
                "robust_correct": outcome["record"]["adversarial_prediction"] == class_name,
                "post_defense_correct": outcome["defended_correct"],
                "ground_truth_mask": ground_truth_mask,
            }
            if ground_truth_mask is not None:
                # The trust grid is coarser than the image; upsample the
                # predicted suspicion mask to the ground-truth resolution
                # before scoring IoU (otherwise shapes never match and
                # localization silently reports nothing).
                predicted_grid = (1.0 - outcome["trust_values"]) >= 0.30
                resized = cv2.resize(
                    predicted_grid.astype(np.uint8),
                    (ground_truth_mask.shape[1], ground_truth_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                predicted_mask = resized.astype(bool)
                record["predicted_mask"] = predicted_mask.astype(np.uint8).tolist()
            all_records.append(record)
            per_config[class_name] = per_config.get(class_name, 0) + 1

    results_path = results_root / "test_results.jsonl"
    results_path.write_text("".join(json.dumps(record, default=str) + "\n" for record in all_records), encoding="utf-8")

    # ---- aggregation ----
    clean_records = [r for r in all_records if not r["is_adversarial"]]
    attack_records = [r for r in all_records if r["is_adversarial"]]
    calibrated = [r for r in attack_records if r["attack_family"] not in {"cw_l2", "deepfool", "mim", "square", "hopskipjump", "autoattack"}]
    zoo = [r for r in attack_records if r["attack_family"] in {"cw_l2", "deepfool", "mim", "square", "hopskipjump", "autoattack"}]

    def robust_accuracy(records: list[dict[str, Any]]) -> float:
        eligible = [r for r in records if r.get("attack_success")]
        if not eligible:
            return 1.0
        return sum(bool(r["robust_correct"]) for r in eligible) / len(eligible)

    def post_defense_accuracy(records: list[dict[str, Any]]) -> float:
        eligible = [r for r in records if r.get("attack_success")]
        if not eligible:
            return 1.0
        return sum(bool(r["post_defense_correct"]) for r in eligible) / len(eligible)

    abstaining = [r for r in all_records if r["final_state"] == "ABSTAIN"]
    abstention_precision = (
        sum(r["is_adversarial"] for r in abstaining) / len(abstaining) if abstaining else 0.0
    )

    by_family: dict[str, list[dict[str, Any]]] = {}
    for record in calibrated:
        by_family.setdefault(record["attack_family"], []).append(record)
    families: dict[str, list[dict[str, Any]]] = {}
    for family, records in by_family.items():
        by_strength: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            by_strength.setdefault(record["strength"], []).append(record)
        families[family] = [
            {
                "strength": strength,
                "samples": len(group),
                "attack_success_rate": sum(bool(r["attack_success"]) for r in group) / len(group),
                "robust_accuracy": robust_accuracy(group),
                "post_defense_accuracy": post_defense_accuracy(group),
                "detection": detection_metrics(group),
                "defense": defense_metrics(group),
                "decisions": decision_metrics(group),
            }
            for strength, group in sorted(by_strength.items())
        ]

    zoo_summary = {
        family: {
            "samples": len(records),
            "attack_success_rate": sum(bool(r["attack_success"]) for r in records) / len(records),
            "robust_accuracy": robust_accuracy(records),
            "post_defense_accuracy": post_defense_accuracy(records),
            "detection": detection_metrics(records),
            "defense": defense_metrics(records),
        }
        for family, records in sorted({r["attack_family"]: [x for x in zoo if x["attack_family"] == r["attack_family"]] for r in zoo}.items())
    } if zoo else {}

    summary = {
        "meta": {
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "model": f"{args.model_tag} tier model",
            "calibration": os.environ.get("ARGUS_CALIBRATION_PATH", "data/models/detector_calibration.json"),
            "clean_count": len(clean_records),
            "attacked_count": len(attack_records),
        },
        "clean": {
            "clean_accuracy": sum(bool(r["clean_correct"]) for r in clean_records) / len(clean_records) if clean_records else 0.0,
            "fpr": detection_metrics(clean_records)["fpr"] if clean_records else 0.0,
            "abstain_rate": sum(r["final_state"] == "ABSTAIN" for r in clean_records) / len(clean_records) if clean_records else 0.0,
        },
        "robust_accuracy_before_defense": robust_accuracy(calibrated),
        "robust_accuracy_after_defense": post_defense_accuracy(calibrated),
        "detection": detection_metrics(calibrated),
        "defense": defense_metrics(calibrated),
        "decisions": decision_metrics(calibrated),
        "abstention_precision": abstention_precision,
        "localization": localization_metrics([r for r in calibrated if r.get("ground_truth_mask") is not None]),
        "latency": latency_metrics(r["latency_ms"] for r in all_records),
        "by_attack_family": families,
        "unseen_attack_families": zoo_summary,
    }
    (results_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    report_path = ROOT / "docs" / "robustness_report.md"
    report_path.write_text(render_robustness_report(summary), encoding="utf-8")
    print(f"Wrote {results_path}")
    print(f"Wrote {results_root / 'summary.json'}")
    print(f"Wrote {report_path}")
    print(json.dumps({k: summary[k] for k in ("clean", "robust_accuracy_before_defense", "robust_accuracy_after_defense", "abstention_precision")}, indent=2))


if __name__ == "__main__":
    main()
