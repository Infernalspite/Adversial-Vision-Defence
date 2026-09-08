"""End-to-end evaluation runner over the existing ARGUS pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2

from app.attacks import get_attack
from app.evaluation.attack_metrics import attack_success_rate
from app.evaluation.config import EvaluationConfig
from app.evaluation.defense_metrics import defense_metrics
from app.evaluation.decision_metrics import decision_metrics
from app.evaluation.detection_metrics import detection_metrics
from app.evaluation.localization_metrics import localization_metrics
from app.evaluation.performance_metrics import latency_metrics
from app.models.model_registry import get_model
from app.pipeline.pipeline import ArgusPipeline, PipelineContext
from app.utils.image import decode_image


class EvaluationRunner:
    """Process evaluation samples and preserve raw per-sample results."""

    def __init__(self, config: EvaluationConfig | None = None) -> None:
        self.config = config or EvaluationConfig()
        self.config.ensure_directories()
        self.model = get_model("resnet18")
        self.pipeline = ArgusPipeline()

    def run(self) -> dict[str, Any]:
        """Evaluate all available categories and write JSONL plus summary JSON."""
        all_records: list[dict[str, Any]] = []
        by_category: dict[str, list[dict[str, Any]]] = {}
        for category in self.config.categories:
            records = self._run_category(category)
            by_category[category] = records
            all_records.extend(records)
            path = self.config.results_root / f"{category}_results.jsonl"
            path.write_text("\n".join(json.dumps(record, default=str) for record in records) + ("\n" if records else ""), encoding="utf-8")
        summary = self.summarize(by_category, all_records)
        (self.config.results_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return summary

    def _run_category(self, category: str) -> list[dict[str, Any]]:
        directory = self.config.dataset_root / category
        if not directory.exists(): return []
        paths = sorted(path for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        if self.config.max_samples_per_category: paths = paths[: self.config.max_samples_per_category]
        records = []
        for path in paths:
            metadata = self._metadata(path)
            image = decode_image(path.read_bytes())
            clean_image = image
            clean_prediction = self.model.predict(clean_image)
            if metadata.get("source_image"):
                source_path = Path(metadata["source_image"])
                if not source_path.is_absolute(): source_path = self.config.dataset_root / source_path
                if source_path.exists():
                    clean_image = decode_image(source_path.read_bytes()); clean_prediction = self.model.predict(clean_image)
            started = time.perf_counter()
            result = self.pipeline.run(PipelineContext(path.stem, image, {"model": self.model, "mode": "evaluation", "attack_type": metadata.get("attack_type")}))
            total_ms = (time.perf_counter() - started) * 1000
            original = result.defense.original_prediction; defended = result.defense.defended_prediction
            ground_truth = metadata.get("ground_truth_class")
            clean_correct = bool(ground_truth is not None and clean_prediction.class_name == ground_truth) if ground_truth else category == "clean"
            record: dict[str, Any] = {
                "sample": path.name, "category": category, "attack_type": metadata.get("attack_type"), "is_adversarial": category != "clean", "ground_truth_class": ground_truth,
                "clean_correct": clean_correct, "clean_prediction": clean_prediction.class_name, "original_prediction": original.class_name, "defended_prediction": defended.class_name,
                "clean_confidence": clean_prediction.confidence, "adversarial_confidence": original.confidence, "defended_confidence": defended.confidence,
                "prediction_changed": original.class_id != clean_prediction.class_id, "attack_success": original.class_id != clean_prediction.class_id,
                "attack_score": result.defense.detection.attack_score, "attack_detected": result.defense.detection.attack_detected,
                "global_trust_score": float(result.defense.trust.trust_map.values.mean()), "suspicious_regions": len(result.defense.trust.localization.regions),
                "defense_method": result.defense.orchestration.decision.selected_defense, "defense_applied": result.defense.orchestration.defense.defense_applied,
                "defense_recovered": defended.class_id == clean_prediction.class_id, "defense_acceptable": result.decision.final_state.value != "ABSTAIN" if category != "clean" else True,
                "verification_score": result.verification.verification.verification_score, "object_consistency": result.verification.verification.object_consistency["object_consistency_score"],
                "geometry_consistency": result.verification.verification.geometry_consistency["structural_similarity"], "scene_consistency": result.verification.verification.scene_consistency["scene_similarity_score"],
                "final_state": result.decision.final_state.value, "decision_score": result.decision.decision_score,
                "latency_ms": {"complete": total_ms, "detection": result.defense.detection.processing_time_ms, "defense": result.defense.orchestration.defense.processing_time_ms, "verification": result.verification.processing_time_ms},
            }
            records.append(record)
        return records

    def _metadata(self, image_path: Path) -> dict[str, Any]:
        metadata_path = image_path.with_suffix(".json")
        if not metadata_path.exists(): return {}
        try: return json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): return {}

    @staticmethod
    def summarize(by_category: dict[str, list[dict[str, Any]]], all_records: list[dict[str, Any]]) -> dict[str, Any]:
        summary: dict[str, Any] = {"status": "ok" if all_records else "no_samples", "dataset": {category: len(records) for category, records in by_category.items()}, "categories": {}, "combined": {}}
        for category, records in by_category.items():
            summary["categories"][category] = {
                "attack": {"attack_success_rate": attack_success_rate(records)},
                "detection": detection_metrics(records), "defense": defense_metrics(records), "decisions": decision_metrics(records), "localization": localization_metrics(records),
                "latency": latency_metrics(record["latency_ms"]["complete"] for record in records),
            }
        summary["combined"] = {"detection": detection_metrics(all_records), "defense": defense_metrics([record for record in all_records if record["is_adversarial"]]), "decisions": decision_metrics(all_records), "localization": localization_metrics(all_records)}
        return summary
