#!/usr/bin/env bash
# Evaluation chain: waits for the post-training chain to finish, then runs the
# final held-out evaluation for baseline (frozen calibration) and robust
# (recalibrated) models. One pass each over the test split.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
LOG=data/robust/logs/evaluation_chain.log

echo "[evalchain] waiting for post-training chain ..." | tee -a "$LOG"
while ! grep -q "\[chain\] done" data/robust/logs/post_training_chain.log 2>/dev/null; do
  sleep 120
done
echo "[evalchain] post-training complete; starting evaluations" | tee -a "$LOG"

echo "[evalchain] baseline final evaluation (baseline calibration)" | tee -a "$LOG"
ARGUS_CALIBRATION_PATH="$(pwd)/data/models/detector_calibration_baseline.json" \
  "$PY" scripts/run_final_evaluation.py --model-tag baseline \
  --results-root data/robust/final_results_baseline >> "$LOG" 2>&1

echo "[evalchain] robust final evaluation (recalibrated)" | tee -a "$LOG"
"$PY" scripts/run_final_evaluation.py --model-tag robust >> "$LOG" 2>&1

echo "[evalchain] done" | tee -a "$LOG"
