"""Read-only evaluation summary endpoint for the judge dashboard."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/evaluation", tags=["evaluation"])
SUMMARY_PATH = Path("data/evaluation/results/summary.json")


@router.get("/summary")
def evaluation_summary() -> dict:
    """Return the latest generated evaluation summary."""
    if not SUMMARY_PATH.exists():
        raise HTTPException(status_code=404, detail="No evaluation summary has been generated")
    try:
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail="Evaluation summary is unreadable") from error
