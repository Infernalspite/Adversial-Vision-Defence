# Phase 12 Test Matrix

| Scenario | Expected |
| --- | --- |
| Clean image | TRUSTED or ABSTAIN based on measured evidence |
| FGSM | DEFENDED or ABSTAIN |
| PGD | DEFENDED or ABSTAIN |
| Patch | DEFENDED or ABSTAIN |
| Failed defense | ABSTAIN |
| Failed verification | ABSTAIN |
| Invalid image | Structured rejection |
| Oversized image | Structured rejection |
| Excessive attack parameters | Validation rejection |
| Backend failure | Safe structured failure |
| Audit failure | Explicit degraded/error handling |

Exact prediction labels and final states are not hard-coded because model outputs and calibrated thresholds can vary.
