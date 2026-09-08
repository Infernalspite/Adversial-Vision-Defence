# Phase 10 Learning Loop

ARGUS-AEGIS Phase 10 is an offline self-healing experiment loop:

```text
Audit Records
→ Failure Analysis
→ Hard-Negative Mining
→ Candidate Dataset
→ Offline Attack Evolution
→ Evaluation
→ Validation Gate
→ Candidate Artifact
→ Explicit Promotion (future)
```

## Hard Negatives

`HardNegativeMiner` reads persisted audit metadata and categorizes review cases such as unsafe acceptance, clean rejection, defense failure, verification failure, localization disagreement, and detector disagreement. It does not assume every failure is adversarial and it does not feed ground-truth attack labels into production decisions.

## Attack Evolution

`AttackConfig` and `AttackMutator` reuse existing FGSM, PGD, and adversarial-patch implementations. Mutation is deterministic when seeded and bounded by explicit parameter limits. `DifficultyScorer` is an engineering ranking metric, not a scientific robustness measure.

Evolution runs only through offline scripts or bounded API instructions. Normal inference never launches an experiment.

## Validation

`ModelValidationGate` compares candidate metrics against configurable safety thresholds. A candidate is `VALIDATED` only when unsafe acceptance, clean rejection, detection TPR, and recovery gates pass. Otherwise it is `REJECTED` with reasons. No model, threshold, or production policy is overwritten.

## Artifacts

Experiments are stored in `data/experiments/<experiment_id>/` with configuration, per-generation JSON, and report files. Hard negatives and learning-cycle summaries are stored under `data/learning/`. These are candidate/offline artifacts, not production model state.

## Commands

```powershell
python scripts/mine_hard_negatives.py
python scripts/run_learning_cycle.py
python scripts/run_attack_evolution.py --image data/samples/test.jpg --attack pgd --generations 3 --population 20 --seed 42
python scripts/validate_candidate.py --metrics candidate_metrics.json
python scripts/demo_learning_cycle.py
```

## Current MVP / Future Production

Current MVP: bounded local experiments, JSON artifacts, seeded mutation, transparent rules, explicit validation, and manual promotion boundary.

Future production: dataset versioning, stronger statistical analysis, isolated workers, signed candidate artifacts, review workflows, and explicit deployment approval. Kubernetes, distributed training, online learning, and automatic production deployment are intentionally out of scope.
