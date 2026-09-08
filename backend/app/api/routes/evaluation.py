"""Read-only evaluation summary endpoint for the judge dashboard."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/evaluation", tags=["evaluation"])
ROOT = Path(__file__).resolve().parents[4]


@router.get("/summary")
def evaluation_summary(dataset: str = Query(default="evaluation_test", pattern=r"^[A-Za-z0-9_-]+$")) -> dict:
    """Return the latest generated evaluation summary."""
    summary_path = ROOT / "data" / dataset / "results" / "summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="No evaluation summary has been generated")
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail="Evaluation summary is unreadable") from error
