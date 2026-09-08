"""Build offline candidate metadata without modifying production data."""

import json
from pathlib import Path
from typing import Iterable

from app.learning.schemas import HardNegative


class CandidateDatasetBuilder:
    """Persist hard-negative metadata as a candidate offline dataset."""

    def build(self, hard_negatives: Iterable[HardNegative], output: str | Path) -> Path:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([item.as_dict() for item in hard_negatives], indent=2), encoding="utf-8")
        return path
