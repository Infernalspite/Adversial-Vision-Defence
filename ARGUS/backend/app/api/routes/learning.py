"""Bounded offline learning and attack-evolution API."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.audit.audit_logger import AuditLogger
from app.learning.experiment_manager import ExperimentManager
from app.learning.hard_negative_mining import HardNegativeMiner
from app.learning.learning_pipeline import run_learning_cycle

router = APIRouter(prefix="/learning", tags=["learning"])


class EvolutionRequest(BaseModel):
    attack: str = "pgd"
    generations: int = Field(default=3, ge=1, le=5)
    population: int = Field(default=20, ge=1, le=50)
    seed: int = 42


@router.post("/mine")
def mine_hard_negatives() -> dict:
    records = AuditLogger().recent(1000)
    mined = HardNegativeMiner().mine(records)
    return {"count": len(mined), "hard_negatives": [item.as_dict() for item in mined]}


@router.post("/evolve")
def evolve(request: EvolutionRequest) -> dict:
    """Return an offline-experiment instruction; large runs belong in CLI scripts."""
    if request.generations > 3 or request.population > 20:
        raise HTTPException(status_code=422, detail="Large evolution runs must be started offline with the CLI")
    return {"status": "OFFLINE_EXPERIMENT_REQUIRED", "configuration": request.model_dump(), "message": "Run scripts/run_attack_evolution.py for the bounded offline experiment."}


@router.post("/validate")
def validate() -> dict:
    return {"status": "OFFLINE_VALIDATION_REQUIRED", "message": "Use scripts/validate_candidate.py with persisted candidate metrics."}


@router.get("/experiments")
def experiments() -> list[dict]:
    return ExperimentManager().list()


@router.get("/experiments/{experiment_id}")
def experiment(experiment_id: str) -> dict:
    try: return ExperimentManager().get(experiment_id)
    except FileNotFoundError as error: raise HTTPException(status_code=404, detail="Experiment not found") from error


@router.get("/hard-negatives")
def hard_negatives() -> dict:
    records = AuditLogger().recent(1000)
    mined = HardNegativeMiner().mine(records)
    return {"count": len(mined), "items": [item.as_dict() for item in mined]}
