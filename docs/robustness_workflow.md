# Robustness Workflow

This document describes the end-to-end workflow that backs the numbers in
[robustness_report.md](robustness_report.md): dataset splits, adversarial
training, attack generation, detector calibration, and the one-pass held-out
evaluation. Every stage is a script; nothing depends on manual steps.

## Person-detection tier

A YOLOv8n detector (`backend/app/models/person_detector.py`) runs alongside the
baseline classifier tier. ImageNet-1000 has no "person" class, so the old
color-heuristic experiment was replaced with a real object detector: when YOLO
finds a person with confidence >= 0.45 and no robust-tier routing applies, the
analysis reports `person` (with per-instance bounding boxes exposed via the
`person_detection` response field). The label is applied identically before and
after defense, so verification and the final decision stay consistent.
Defended-class routing is unaffected. If ultralytics or its weights are
missing, the feature degrades to plain classifier labels — it never breaks the
pipeline. Person counts are *detection*, not classification: adversarial-patch
evaluation does not apply to this tier, and `person` predictions never enter
the 10-class robust metrics.

## Fixed class set

The 10 defended classes are pinned in `backend/app/learning/classes.py` and
mirrored to `data/robust/classes.json`. They are the ten Imagenette classes
(tench, English springer spaniel, cassette player, chain saw, church,
French horn, garbage truck, gas pump, golf ball, parachute). Changing the
class list means changing that one table and re-running every stage below.

Recognition scope is two-tier:

- **Tier 1** — frozen ImageNet-1000 ResNet-18 (`get_model("resnet18")`)
  classifies any uploaded image; it is never retrained.
- **Tier 2** — robust specialist (`get_model("robust")`) serves the 10
  defended classes inside the defense loop. `backend/app/models/tier_router.py`
  routes each request: if the baseline's top predictions include a defended
  class, the specialist analyzes and re-classifies; otherwise the baseline
  serves the label. Both tiers run the same detection/purification chain.

## Stage 1 — Splits

```bash
python scripts/make_splits.py                 # defaults: 250/75/75 per class
```

Stratified, seeded, deterministic. Writes `data/robust/splits/{train,val,test}/`
(symlinks or copies) plus `data/robust/splits/manifest.json` with per-image
md5s. `assert_disjoint` proves no image — or any attacked variant derived
from it — appears in two splits. The test split must be consumed exactly
once, by Stage 5.

## Stage 2 — Attack generation (two rounds)

```bash
# Round A: white-box against the frozen baseline (calibration + comparison)
python scripts/generate_attack_suite.py --model-tag baseline \
    --splits val --attacks fgsm,pgd,patch --max-per-class 8
python scripts/generate_attack_suite.py --model-tag baseline \
    --splits test --attacks fgsm,pgd,patch --max-per-class 8

# Round B: white-box against the trained robust model (after Stage 3)
python scripts/generate_attack_suite.py --model-tag robust \
    --splits val,test --attacks fgsm,pgd,patch --max-per-class 8

# Unseen attack families (evaluation only, NEVER calibration)
python scripts/generate_attack_suite.py --model-tag robust --splits test \
    --attacks cw_l2,deepfool,mim,square,hopskipjump,autoattack --zoo-max-per-class 2
```

Grid: FGSM/PGD at epsilon {2, 4, 8, 16}/255; patches at size {0.15, 0.30}
with seeded random locations and saved ground-truth masks. Only successful
attacks are written as images; every attempt (success or failure) is recorded
in `data/robust/attacks/<tag>/run_manifest.json` so robust accuracy stays
computable. Attacks are always regenerated against the model they test —
Round A attacks are not valid robust-accuracy numbers for the robust model.

Zoo attacks (`app/attacks/zoo.py`) wrap torchattacks (PGD cross-validation)
and ART (C&W, DeepFool, MIM, Square, HopSkipJump, AutoAttack). They are
deliberately excluded from calibration so detection rates on them measure
generalization to unseen attack families.

## Stage 3 — Adversarial training

```bash
python scripts/train_robust_model.py \
    --clean-root data/robust/splits/train --val-root data/robust/splits/val \
    --epochs 8 --batch-size 32 --attack-mode fgsm   # or pgd2 / none
```

Each batch is trained on a mix of clean loss and white-box adversarial loss
(FGSM at 8/255 by default — Wong et al. "fast is better than free"; PGD-2
available). Attacks run on the in-training model, so examples are never
stale. Every epoch tracks **clean val accuracy and robust val accuracy**
(PGD-10 on a fixed per-class val subset); checkpoints are selected on their
harmonic mean, never clean accuracy alone. Output: `data/models/robust_model.pt`
plus `robust_training_summary.json` with both curves.

## Stage 4 — Detector calibration (val split only)

```bash
python scripts/calibrate_detectors.py --model-tag baseline   # or robust
```

Collects the four detector scores on val clean images (FPR side) and the
matching attack suite (TPR side), then grid-searches thresholds 0.05–0.95 x
weights in 0.25 steps summing to 1, plus a fitted logistic fusion, under the
hard constraint FPR <= 0.20 and the objective max(F1 - 0.25 * FPR). Writes
`data/models/detector_calibration.json` in the schema `DetectionPipeline`
loads at startup — deployment is the file appearing. Saliency stays computed
for localization regardless of its fused weight (golden test in
`tests/unit/test_calibration.py`).

## Stage 5 — Final held-out evaluation (one pass)

```bash
python scripts/run_final_evaluation.py --model-tag robust
```

Full chain (classify -> detect -> localize -> purify -> re-detect ->
re-classify) on the test split only: clean records for accuracy/FPR, plus
every attacked config. Aggregates by attack family and strength; renders
`docs/robustness_report.md` with clean and robust accuracy together, per-
strength detection, patch IoU, recovery, abstention precision, latency,
the unseen-family section, and the threat-model caveats. If anything is
retuned after looking at these numbers, the test split is burned: re-split
or collect fresh test data before claiming a new result.

## Reproducing from scratch

```bash
python scripts/make_splits.py
python scripts/generate_attack_suite.py --model-tag baseline --splits val --attacks fgsm,pgd,patch --max-per-class 8
python scripts/train_robust_model.py --clean-root data/robust/splits/train --val-root data/robust/splits/val
python scripts/generate_attack_suite.py --model-tag robust --splits val,test --attacks fgsm,pgd,patch --max-per-class 8
python scripts/generate_attack_suite.py --model-tag robust --splits test --attacks cw_l2,deepfool,mim,square,hopskipjump,autoattack --zoo-max-per-class 2
python scripts/calibrate_detectors.py --model-tag robust
python scripts/run_final_evaluation.py --model-tag robust
```
