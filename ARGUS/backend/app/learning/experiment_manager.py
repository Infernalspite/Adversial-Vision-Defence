"""Persistence for bounded offline learning experiments."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ExperimentManager:
    """Store experiment metadata and generations outside production artifacts."""

    def __init__(self, root: str | Path = "data/experiments") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, config: dict[str, Any]) -> tuple[str, Path]:
        experiment_id = f"experiment_{uuid.uuid4().hex[:10]}"
        path = self.root / experiment_id
        path.mkdir(parents=True, exist_ok=False)
        payload = {**config, "experiment_id": experiment_id, "timestamp": datetime.now(timezone.utc).isoformat()}
        (path / "config.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return experiment_id, path

    def save_generation(self, path: Path, generation: int, records: list[dict[str, Any]]) -> None:
        (path / f"generation_{generation:02d}.json").write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")

    def save_report(self, path: Path, report: dict[str, Any]) -> None:
        (path / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    def list(self) -> list[dict[str, Any]]:
        output = []
        for path in sorted(self.root.glob("experiment_*")):
            config = path / "config.json"
            if config.exists():
                output.append(json.loads(config.read_text(encoding="utf-8")))
        return output

    def get(self, experiment_id: str) -> dict[str, Any]:
        path = self.root / experiment_id
        if not path.exists(): raise FileNotFoundError(experiment_id)
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        report_path = path / "report.json"
        config["report"] = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
        return config
