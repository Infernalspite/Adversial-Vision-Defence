# ARGUS-AEGIS Hackathon Pitch Script

## 1. Opening: The Problem

"Modern vision models are very confident, but confidence is not the same as trust. A small, carefully designed perturbation can make a classifier change its prediction while the image still looks normal to a person. In a safety-sensitive workflow, silently returning the wrong answer is worse than admitting uncertainty.

ARGUS-AEGIS addresses that gap. It is an autonomous adversarial defense and recovery fabric that sits between an image and a vision model. It does not blindly trust the first prediction. It detects suspicious evidence, localizes where perception is less trustworthy, chooses a defense, re-runs inference, verifies the result, and abstains when trust cannot be established."

## 2. One-Sentence Solution

"ARGUS-AEGIS turns image classification into a measured security decision: DETECT, LOCALIZE, DEFEND, RE-INFER, VERIFY, TRUST or ABSTAIN."

## 3. Why This Matters

"A normal classifier gives us a label and a confidence score. ARGUS-AEGIS gives us a decision with evidence:

- What the base model predicted.
- Which independent detectors raised concern.
- How much of the image appears suspicious.
- Which defense was selected.
- What changed after defense.
- Whether semantic and spatial checks agreed.
- Why the system trusted the result or refused to force one."

The key design principle is conservative behavior. When evidence is unresolved, the system returns `ABSTAIN` instead of presenting an unsafe prediction as reliable.

## 4. What We Built

"The implementation is a complete CPU-compatible vertical slice, not just a model or a dashboard. It includes:

1. A pretrained ResNet-18 base vision model.
2. An Attack Lab with FGSM, PGD, and localized adversarial patch attacks.
3. Four adversarial triage detectors.
4. A transparent weighted attack score.
5. A spatial perception trust map.
6. Suspicious-region localization.
7. A defense orchestrator.
8. Image transformations, masking/inpainting, and recovery paths.
9. Re-inference after defense.
10. Semantic, geometric, and scene consistency verification.
11. Final `TRUSTED`, `DEFENDED`, or `ABSTAIN` states.
12. Structured audit logging.
13. A FastAPI backend and React dashboard.
14. Reproducible validation and final-test evaluation."

## 5. Architecture Walkthrough

Use this sequence while showing the architecture diagram:

"The request enters through zero-trust image validation. The base ResNet-18 produces the initial prediction. Four detectors then inspect different kinds of evidence:

- Feature squeezing compares the original image with reduced-bit-depth versions.
- Frequency analysis measures high-frequency image behavior.
- Confidence instability checks whether controlled transformations cause unstable model behavior.
- Saliency analysis examines gradient-based visual concentration and produces spatial evidence.

Their outputs are normalized and fused into one attack score between zero and one. The score is not treated as proof by itself. It feeds the trust-map stage, where spatial evidence is converted into regions of lower perception trust.

The defense orchestrator then chooses the least expensive appropriate response. Depending on the evidence, it may transform the image, mask or inpaint a suspicious region, or escalate to a stronger recovery path. The model is run again on the defended image.

Finally, the verification layer compares the original and defended interpretations and checks object, geometry, scene, and spatial consistency. The decision engine returns exactly one state: TRUSTED, DEFENDED, or ABSTAIN. Every decision is logged with its evidence and timing."

## 6. Dataset Story

"We used Imagenette because it is a compact, recognized ImageNet-derived dataset with ten classes and a practical size for a CPU-friendly hackathon evaluation."

Classes used:

- tench
- English springer
- cassette player
- chain saw
- church
- French horn
- garbage truck
- gas pump
- golf ball
- parachute

Source:

- Official repository: https://github.com/fastai/imagenette
- 320px archive: https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz

The 320px archive was downloaded and extracted once under `data/imagenette/imagenette2-320/`. We did not use an external pre-generated adversarial dataset.

From the extracted training images, we created three disjoint source pools using a deterministic seed:

- `data/sources/baseline`: 200 images
- `data/sources/validation`: 200 images
- `data/sources/test`: 200 images

The manifests record source image identity, split, class ID, class name, attack type, parameters, seed, and mask path where applicable. An audit confirmed zero overlap between baseline, validation, and test pools.

## 7. How the Adversarial Data Was Made

"The adversarial samples were generated against this project's actual ResNet-18 model. That matters because an attack generated against another model would not reliably test this pipeline."

Validation and test generation used the same controlled parameters:

- FGSM: epsilon `0.01`
- PGD: epsilon `0.03`, step size `0.005`, 10 iterations, deterministic no-random-start mode
- Patch: 20% patch size, centered location, 10 optimization iterations
- Seed: `7`
- 200 samples per category

The generated layouts are:

```text
data/evaluation_validation/
  clean/   200
  fgsm/    200
  pgd/     200
  patch/   200

data/evaluation_test/
  clean/   200
  fgsm/    200
  pgd/      200
  patch/   200
```

Patch masks are stored beside the corresponding patch image as `*_mask.png`. The evaluator explicitly excludes mask files from sample counts and uses them only for offline localization metrics, never for live detection.

## 8. What Counts as Attack Success

"We do not call an attack successful merely because pixels changed. Attack success means that the clean image was correctly classified and the adversarial image changed the model prediction away from the clean class."

We also record:

- L-infinity perturbation magnitude.
- Mean absolute perturbation.
- Percentage of changed pixels.
- Patch mask area ratio.
- Patch localization IoU, precision, and recall.

## 9. Calibration Method

"The untouched test set was not used to tune the system. We first evaluated the validation set, saved individual detector scores, calculated score distributions, and searched a small interpretable grid of thresholds and detector weights."

The calibration procedure:

- Candidate thresholds: `0.05` through `0.95`.
- Candidate detector weights: non-negative values in 0.25 increments summing to 1.
- Constraint: false-positive rate at most 20%.
- Selection criterion: maximize F1 minus 0.25 times FPR.
- Calibration data: validation only.

Selected validation configuration:

- Threshold: `0.05`
- Feature squeezing: `0.00`
- Frequency analysis: `0.25`
- Confidence instability: `0.75`
- Saliency analysis: `0.00`

This result is not hidden or oversold. Validation ROC-AUC was approximately `0.519`, showing that the current detectors have limited separation on this attack setup.

## 10. Evaluation Results

"The final test set was evaluated only after calibration was frozen. These numbers are measurements from the pipeline, not hard-coded demo claims."

Final test results:

| Category | Attack success | Detection TPR | FPR | Recovery | Abstain |
|---|---:|---:|---:|---:|---:|
| Clean | 0.0% | 0.0% | 23.5% | 0.0% | 32.5% |
| FGSM | 96.2% | 45.5% | 0.0% | 1.0% | 67.0% |
| PGD | 100.0% | 4.0% | 0.0% | 0.0% | 4.0% |
| Patch | 98.1% | 61.5% | 0.0% | 10.2% | 62.5% |
| Combined | - | 37.0% | 23.5% | 3.7% | 41.5% |

Combined unsafe acceptance was `55.5%`. Patch localization achieved:

- IoU: `25.4%`
- Precision: `45.8%`
- Recall: `40.1%`

Average end-to-end latency was approximately 0.6 seconds per image on CPU.

## 11. How to Explain the Results Honestly

"The evaluation demonstrates that the full loop works and that the system can abstain, localize some patch evidence, and detect a meaningful portion of FGSM and patch cases. It also clearly shows where the current MVP is weak: PGD detection and defense recovery are not strong enough for a production claim.

That is useful evidence, not a failure to report. The system exposes unsafe acceptance and false positives instead of hiding them. The next research step would be improving detector features and calibration, especially for iterative PGD attacks, then returning to validation before any new test evaluation."

Do not say:

- "We solved adversarial attacks."
- "The system detects every attack."
- "The model is certified secure."
- "The test set was used to improve the threshold."

Say:

- "We built and measured a complete adversarial-defense loop."
- "The system is conservative and explainable."
- "The benchmark exposes both strengths and failure modes."
- "Abstention is a designed safety behavior, not an error."

## 12. Live Demo Script

### Demo A: Clean Input

1. Open `http://localhost:5173/`.
2. Upload a clean image from `data/sources/test/`.
3. Click `RUN ANALYSIS`.
4. Point out:
   - Base prediction and confidence.
   - Detector scores.
   - Trust-map overlay.
   - Defense decision.
   - Verification scores.
   - Final `TRUSTED` or `ABSTAIN` state.
   - Audit event created.

Say:

"Even for a clean image, ARGUS shows its reasoning chain. It does not only return a label; it records why the input was accepted or rejected."

### Demo B: Attack Lab

1. Open `Attack Lab`.
2. Upload the same clean image.
3. Select `FGSM`, `PGD`, or `Adversarial Patch`.
4. Adjust bounded parameters if desired.
5. Click `GENERATE ATTACK`.
6. Show original versus adversarial image.
7. Point out whether the prediction changed.
8. Click `RUN ARGUS-AEGIS`.
9. Show the attack score, trust map, defense trace, final state, and audit event.

Say:

"The attack is generated against the same ResNet-18 used by ARGUS, so this is a controlled white-box stress test rather than a random image effect."

### Demo C: Evaluation Page

1. Open `Evaluation`.
2. Show the final test metrics loaded from `data/evaluation_test/results/summary.json`.
3. Explain that calibration used validation data only.
4. Show the clean rejection, unsafe acceptance, detection, recovery, and abstention values.

Say:

"This page is connected to the generated report. The numbers come from the evaluator and include failures, false positives, and abstentions."

### Demo D: Audit Trail

1. Open `Audit Trail`.
2. Expand an event.
3. Show request ID, decision reason, attack evidence, selected defense, verification, and timing.

Say:

"Every decision is correlated to an audit event, which makes the pipeline inspectable rather than opaque."

## 13. Technical Demo Commands

Start the backend with calibrated live scoring:

```powershell
Set-Location C:\Users\itsta\Downloads\ARGUS\backend
$env:ARGUS_CALIBRATION_PATH = "..\data\evaluation_validation\calibration.json"
..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Start the frontend in a second terminal:

```powershell
Set-Location C:\Users\itsta\Downloads\ARGUS\frontend
npm run dev -- --host 0.0.0.0
```

Open:

- Dashboard: http://localhost:5173/
- API documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/api/v1/health

## 14. Reproducibility Commands

Verify manifests and disjoint source pools:

```powershell
Set-Location C:\Users\itsta\Downloads\ARGUS
.\.venv\Scripts\python.exe .\scripts\build_dataset_manifests.py `
  --evaluation-root .\data\evaluation_validation `
  --evaluation-root .\data\evaluation_test
```

Run validation calibration:

```powershell
.\.venv\Scripts\python.exe .\scripts\calibrate_detectors.py `
  --results .\data\evaluation_validation\results `
  --output .\data\evaluation_validation\calibration.json
```

Run final test evaluation without changing calibration:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_evaluation.py `
  --dataset .\data\evaluation_test `
  --max-samples 200 `
  --calibration .\data\evaluation_validation\calibration.json
```

Run checks:

```powershell
.\.venv\Scripts\python.exe -m pytest backend\tests -q
.\.venv\Scripts\python.exe .\scripts\security_check.py
.\.venv\Scripts\python.exe .\scripts\demo.py
Set-Location frontend
npm run test:run
npm run build
```

## 15. Likely Judge Questions

### Why not use a separate adversarial dataset?

"Because FGSM and PGD are model-dependent. We generate them against our own ResNet-18 so the evaluation measures this system's actual threat model."

### Did you tune on the test set?

"No. Validation was used for detector distributions, threshold selection, and weight selection. The test configuration was then frozen before evaluating the untouched test set."

### Is the model retrained?

"No. The pretrained ResNet-18 remains the vulnerable base perception model. That isolates the defense problem and keeps the MVP reproducible and CPU-compatible."

### Why is abstention important?

"A wrong confident prediction can be unsafe. When defense and verification cannot establish trust, abstention is the safer system response."

### Why are the results not higher?

"The benchmark is intentionally honest. The current detectors separate some patch and FGSM cases but struggle on this PGD setup. That identifies the next research target instead of hiding it behind a threshold or hard-coded attack label."

### Is this production-ready?

"No. Production deployment would require stronger detectors, broader datasets, distributed rate limiting, model signing, hardened execution, and extensive red-team validation. Those are roadmap items outside this hackathon MVP."

## 16. Closing

"ARGUS-AEGIS is not just an adversarial image generator and not just a classifier wrapper. It is a complete security loop around perception.

It treats visual input as untrusted, combines independent evidence, reasons spatially, adapts the defense, verifies the result, records the decision, and refuses to force an answer when trust is insufficient.

The immediate MVP contribution is the system-level integration: DETECT, LOCALIZE, DEFEND, RE-INFER, VERIFY, TRUST, ABSTAIN, and LEARN. The evaluation shows exactly what works today and exactly what must improve next. That makes the project reproducible, explainable, and ready for the next iteration."

## 17. Final Presenter Checklist

- [ ] Backend is running on port 8000.
- [ ] Frontend is running on port 5173.
- [ ] Health endpoint returns status `ok`.
- [ ] Calibration file exists at `data/evaluation_validation/calibration.json`.
- [ ] Final test summary exists at `data/evaluation_test/results/summary.json`.
- [ ] Have one clean image ready.
- [ ] Have one image ready for Attack Lab.
- [ ] Know the final metrics and limitations.
- [ ] Do not claim `DEFENDED` recovery unless the live result actually shows it.
- [ ] Keep the evaluation page open as evidence of reproducibility.
