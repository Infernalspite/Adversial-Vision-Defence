# ARGUS-AEGIS Judge Demo Script

Target duration: 5–7 minutes.

## 1. Clean Input

Open the Dashboard and upload a clean image. Explain:

> The image passes through zero-trust ingestion and the base vision model before any trust decision is made.

Show the original prediction, confidence, attack score, trust map, verification, and final state.

## 2. Attack Lab

Open Attack Lab, choose FGSM, PGD, or Adversarial Patch, and generate an adversarial image. Show the actual original-versus-adversarial prediction comparison.

Click `RUN ARGUS-AEGIS` to send the generated output through the canonical `/api/v1/analyze` endpoint.

## 3. Defense and Verification

Show the attack score, suspicious regions, backend trust overlay, selected defense, defended prediction, and object/geometry/scene verification.

Emphasize that the system may return `ABSTAIN`; it does not force recovery when evidence is weak.

## 4. Audit

Open Audit Trail and expand the decision event. Show request ID, timestamps, evidence summaries, decision reasons, and tamper-evident record hash.

## 5. Evaluation

Open Evaluation to show measured detection, recovery, safety, decision-distribution, and latency metrics from generated evaluation artifacts.

## 6. Learning

Open Learning. Point out the explicit `OFFLINE RED TEAM / CANDIDATE WORKBENCH` label. Show hard negatives, experiment IDs, candidate status, and the fact that production remains unchanged.

## 7. One-Click CLI

```powershell
python scripts/demo.py
```

The script runs clean, FGSM, PGD, and patch cases through the real model and pipeline. It uses an existing generated image when available, otherwise a deterministic small local sample.

Closing statement:

> ARGUS-AEGIS does not assume visual input is trustworthy. It measures trust, defends when possible, verifies recovery, and abstains when it cannot establish safety.
