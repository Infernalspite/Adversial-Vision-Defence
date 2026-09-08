# Evaluation and Benchmarking

Phase 9 evaluates the existing ARGUS-AEGIS pipeline without changing its detectors, defenses, thresholds, or decision logic.

## Dataset Layout

```text
data/evaluation/
├── clean/
├── fgsm/
├── pgd/
├── patch/
├── results/
└── plots/
```

Each sample may have a same-stem JSON metadata file. Adversarial metadata records the attack type, source clean image, and ground-truth clean class. Patch masks are evaluated only when a real mask is present.

## Build Samples

```powershell
python scripts/build_evaluation_dataset.py --input data/raw/clean --output data/evaluation
```

Optional controls:

```powershell
python scripts/build_evaluation_dataset.py --input data/raw/clean --output data/evaluation --number 25 --attack-type fgsm --seed 7
```

Existing generated samples are not overwritten.

## Calibration Boundary

Development and validation data may be used by `scripts/calibrate_thresholds.py` to inspect attack-threshold tradeoffs. The final test set must not be used to tune detector weights, attack thresholds, trust thresholds, verification thresholds, or defense policy thresholds.

```powershell
python scripts/calibrate_thresholds.py --results data/evaluation/results
```

The current script reports a validation-only tradeoff and does not modify production configuration.

## Final Evaluation

```powershell
python scripts/run_evaluation.py --dataset data/evaluation
```

Outputs:

- `data/evaluation/results/*_results.jsonl`: raw sample records.
- `data/evaluation/results/summary.json`: aggregate machine-readable report.
- `data/evaluation/results/summary.csv`: chart-friendly aggregate rows.
- `data/evaluation/results/run_metadata.json`: timestamp, seed, attack parameters, thresholds, and optional git commit.

## Metrics

- **Attack Success Rate:** prediction changes divided by correctly classified clean source samples.
- **Detection TPR/Recall:** adversarial samples detected.
- **False Positive Rate:** clean samples incorrectly detected.
- **Precision/F1/ROC-AUC:** thresholded and raw-score detection quality.
- **Defense Recovery Rate:** successful attacks whose defended prediction returns to the clean class.
- **Confidence Recovery:** adversarial-to-defended confidence movement.
- **TRUSTED/DEFENDED/ABSTAIN rates:** final state distribution by category.
- **Unsafe Acceptance Rate:** adversarial samples marked `TRUSTED`.
- **Safe Rejection Rate:** adversarial samples marked `DEFENDED` or `ABSTAIN`.
- **Clean Rejection Rate:** clean samples marked `ABSTAIN`.
- **IoU/precision/recall:** patch localization only where a ground-truth mask exists.
- **MAE/MSE/PSNR:** image distortion metrics, not security metrics.
- **Latency:** mean, median, p95, minimum, and maximum where stage timings are available.

## Plots and Ablation

```powershell
python scripts/generate_evaluation_plots.py --results data/evaluation/results --output data/evaluation/plots
python scripts/run_ablation.py --results data/evaluation/results
```

The ablation script does not fabricate baseline or detection-only outputs. Those configurations should be run and recorded separately before comparison.

## Limitations

Small or synthetic datasets are demonstration evidence, not benchmark proof. No statistical significance claims are made. Results can vary with pretrained-weight availability, CPU/GPU hardware, image distribution, and calibration choices. Failed attacks, failed defenses, false positives, and abstentions remain in the raw output.

Phase 10 consumes these evaluation artifacts as offline candidate evidence. It does not use final-test data to tune thresholds and does not automatically promote a candidate.
